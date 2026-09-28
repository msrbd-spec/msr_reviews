import os
import torch
import torch.nn as nn
import timm

from .components import MSDABlock, HFFBlock, MultiScaleHead, OrdinalRegressionHead, AuxHead



class RetiNA_Net(nn.Module):
    """
    RetiNA-Net: Retinal Neural Architecture Network.

    A custom DR classification network combining:
      - SwinV2-Large backbone (pretrained, features_only)
      - MSDA: Multi-Scale Deformable Attention on Stage 3 & 4
      - HFF: Hierarchical Feature Fusion (Stage 1 → Stage 4, all 4 stages)
      - Spatial Attention Pooling head
      - Deep supervision auxiliary head
      - Ordinal regression head

    Architecture:
      - Backbone: SwinV2-Large (pretrained, features_only)
      - MSDA: Multi-Scale Deformable Attention on Stage 3 & 4
      - HFF: Hierarchical Feature Fusion (Stage 1 → Stage 4, all 4 stages)
      - Head: Multi-scale attention pooling + 2-layer MLP
      - Aux head: Deep supervision on stage-3 features
      - Ordinal head: Optional ordinal regression head for ordinal-aware loss

    The model returns a dict with:
      - 'logits': classification logits (B, num_classes)
      - 'aux_logits': auxiliary logits (B, num_classes) or None
      - 'ordinal_logits': ordinal cumulative logits (B, num_classes-1) or None
    """

    def __init__(self, use_msda: bool = True, use_hff: bool = True,
                 num_classes: int = 5, drop_path_rate: float = 0.1,
                 dropout: float = 0.1, use_ordinal: bool = True,
                 use_aux_head: bool = True, use_attention_pool: bool = True,
                 backbone_name: str = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
                 stage_channels=(192, 384, 768, 1536),
                 ssl_pretrained_path: str = None):
        """
        Args:
            ssl_pretrained_path: Path to SSL-pretrained backbone weights.
                                 If provided, loads these instead of ImageNet weights.
                                 Set to None to use default ImageNet pretraining.
        """
        super().__init__()
        self.use_msda = use_msda
        self.use_hff = use_hff
        self.use_ordinal = use_ordinal
        self.use_aux_head = use_aux_head
        self.num_classes = num_classes
        self.stage_channels = list(stage_channels)

        # Backbone: SwinV2-Large with dynamic image size for 512x512 input
        # If ssl_pretrained_path is provided, we load ImageNet weights first
        # (to ensure all parameters exist), then override with SSL-pretrained weights.
        # checkpoint_in_feature_block=True: gradient checkpointing to save ~40% GPU memory
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,  # Always load ImageNet first (ensures all params initialized)
            features_only=True,
            dynamic_img_size=True,
            img_size=512,
            drop_path_rate=drop_path_rate,
            checkpoint_in_feature_block=True
        )

        # Load SSL-pretrained backbone weights if provided
        if ssl_pretrained_path is not None and os.path.exists(ssl_pretrained_path):
            ssl_state = torch.load(ssl_pretrained_path, map_location='cpu')
            # The SSL backbone state dict keys should match self.backbone's parameters
            # (both use the same timm SwinV2 model with features_only=True)
            missing, unexpected = self.backbone.load_state_dict(ssl_state, strict=False)
            if missing:
                print(f"SSL backbone load: {len(missing)} missing keys (expected for new layers)")
            if unexpected:
                print(f"SSL backbone load: {len(unexpected)} unexpected keys")
            print(f"Loaded SSL-pretrained backbone from {ssl_pretrained_path}")
        elif ssl_pretrained_path is not None:
            print(f"Warning: SSL pretrained path '{ssl_pretrained_path}' not found. Using ImageNet weights.")


        if self.use_msda:
            self.msda3 = MSDABlock(in_channels=self.stage_channels[2], kernel_size=3)
            self.msda4 = MSDABlock(in_channels=self.stage_channels[3], kernel_size=3)

        if self.use_hff:
            self.hff = HFFBlock(
                stage_channels=tuple(self.stage_channels),
                target_channels=self.stage_channels[3]
            )

        # Multi-scale classification head (attention pooling + 2-layer MLP)
        self.head = MultiScaleHead(
            in_channels=self.stage_channels[3],
            num_classes=num_classes,
            dropout=dropout,
            use_attention_pool=use_attention_pool
        )

        # Auxiliary head for deep supervision (on stage-3 features)
        if self.use_aux_head:
            self.aux_head = AuxHead(
                in_channels=self.stage_channels[2],
                num_classes=num_classes,
                dropout=dropout
            )

        # Optional ordinal regression head
        if self.use_ordinal:
            self.ordinal_head = OrdinalRegressionHead(
                in_channels=self.stage_channels[3],
                num_classes=num_classes,
                dropout=dropout
            )

    def forward(self, x):
        features = self.backbone(x)
        stage1 = features[0]
        stage2 = features[1]
        stage3 = features[2]
        stage4 = features[3]

        # Swin Transformers return features in (B, H, W, C) format.
        # Permute to (B, C, H, W) if the last dimension matches expected channel count.
        if stage4.shape[-1] == self.stage_channels[3]:
            stage1 = stage1.permute(0, 3, 1, 2).contiguous()
            stage2 = stage2.permute(0, 3, 1, 2).contiguous()
            stage3 = stage3.permute(0, 3, 1, 2).contiguous()
            stage4 = stage4.permute(0, 3, 1, 2).contiguous()

        # Save stage3 before MSDA for aux head
        stage3_pre_msda = stage3

        if self.use_msda:
            stage3 = self.msda3(stage3)
            stage4 = self.msda4(stage4)

        if self.use_hff:
            fused_features = self.hff(stage1, stage2, stage3, stage4)
        else:
            fused_features = stage4

        # Classification logits
        logits = self.head(fused_features)

        # Auxiliary logits (deep supervision on stage-3 pre-MSDA features)
        aux_logits = None
        if self.use_aux_head:
            aux_logits = self.aux_head(stage3_pre_msda)

        # Ordinal logits (if enabled)
        ordinal_logits = None
        if self.use_ordinal:
            ordinal_logits = self.ordinal_head(fused_features)

        return {
            'logits': logits,
            'aux_logits': aux_logits,
            'ordinal_logits': ordinal_logits
        }

    def get_classification_logits(self, x):
        """Convenience method returning only classification logits."""
        return self.forward(x)['logits']

    def predict(self, x):
        """
        Inference method returning softmax probabilities.
        Combines classification and ordinal heads for final prediction.
        Uses 0.7/0.3 weighting (classification dominant).
        """
        out = self.forward(x)
        cls_probs = torch.softmax(out['logits'], dim=1)

        if self.use_ordinal and out['ordinal_logits'] is not None:
            ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(out['ordinal_logits'])
            # 70% classification, 30% ordinal — classification head is more reliable
            final_probs = 0.7 * cls_probs + 0.3 * ord_probs
        else:
            final_probs = cls_probs

        return final_probs
