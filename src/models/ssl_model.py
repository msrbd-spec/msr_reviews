"""
SSL Pretraining Model: Lesion-Aware Contrastive + Multi-Task.

Architecture (per plan.md Phase 3):
  SwinV2-Large backbone (shared, pretrained=ImageNet)
  ├── Projection head (MLP 1536→512→128) → Contrastive loss (InfoNCE)
  ├── Lesion type head (Multi-label BCE, 3 outputs) → from stage-3 features
  ├── Lesion count head (MSE regression, 3 outputs) → from stage-4 features
  └── Lightweight decoder (ConvTranspose → 128×128) → Reconstruction loss

Dimension flow:
  Input: (B, 3, 512, 512)
  Backbone outputs (features_only, in BCHW format after timm internal handling):
    - Stage 1: (B, 192, 128, 128)    — but SwinV2 returns (B, H, W, C)
    - Stage 2: (B, 384, 64, 64)
    - Stage 3: (B, 768, 32, 32)
    - Stage 4: (B, 1536, 16, 16)

  We permute from (B, H, W, C) → (B, C, H, W) for conv operations.

  Projection head: pool stage4 → (B, 1536) → MLP → (B, 128)
  Lesion type head: pool stage3 → (B, 768) → MLP → (B, 3)
  Lesion count head: pool stage4 → (B, 1536) → MLP → (B, 3)
  Decoder: stage3 + stage4 → upsample → (B, 3, 128, 128) lesion masks

Author: RetiNA-Net Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ProjectionHead(nn.Module):
    """
    SimCLR-style projection head for contrastive learning.
    Maps backbone features to a 128-dim normalized embedding space.

    Input: (B, in_channels) — pooled stage-4 features
    Output: (B, 128) — L2-normalized projection
    """

    def __init__(self, in_channels=1536, hidden_dim=512, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: (B, in_channels)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # L2 normalize for contrastive loss
        x = F.normalize(x, dim=1)
        return x  # (B, out_dim)


class LesionTypeHead(nn.Module):
    """
    Multi-label classification head for lesion presence prediction.
    Predicts whether each lesion type (microaneurysm, hemorrhage, exudate)
    is present in the image.

    Input: (B, C, H, W) — stage-3 features (768ch)
    Output: (B, 3) — sigmoid logits for 3 lesion types
    """

    def __init__(self, in_channels=768, num_lesion_types=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_lesion_types)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        pooled = self.avg_pool(x).flatten(1)  # (B, C)
        return self.fc(pooled)  # (B, 3)


class LesionCountHead(nn.Module):
    """
    Regression head for lesion count prediction.
    Predicts the count of each lesion type (normalized to [0, 1]).

    Input: (B, C, H, W) — stage-4 features (1536ch)
    Output: (B, 3) — predicted normalized counts
    """

    def __init__(self, in_channels=1536, num_lesion_types=3, max_count=100):
        super().__init__()
        self.max_count = max_count
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_lesion_types)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        pooled = self.avg_pool(x).flatten(1)  # (B, C)
        counts = self.fc(pooled)  # (B, 3)
        # Sigmoid to normalize to [0, 1] (counts are normalized by max_count)
        return torch.sigmoid(counts)


class LesionDecoder(nn.Module):
    """
    Lightweight decoder for lesion mask reconstruction.
    Takes stage-3 and stage-4 features, fuses and upsamples to produce
    a 3-channel lesion mask at 128×128 resolution.

    Input:
        stage3: (B, 768, 32, 32)
        stage4: (B, 1536, 16, 16)
    Output:
        mask: (B, 3, 128, 128) — reconstructed lesion masks
    """

    def __init__(self, stage3_channels=768, stage4_channels=1536, out_channels=3):
        super().__init__()

        # Project stage4 (16×16) to stage3 resolution (32×32)
        self.stage4_upsample = nn.Sequential(
            nn.Conv2d(stage4_channels, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Fuse stage3 + upsampled stage4
        self.fuse = nn.Sequential(
            nn.Conv2d(768 + 768, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Upsample to 128×128 (32 → 64 → 128)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32→64
        )

        self.upsample2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64→128
        )

        # Final 1×1 conv to produce 3-channel mask
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, stage3, stage4):
        # stage3: (B, 768, 32, 32)
        # stage4: (B, 1536, 16, 16)

        # Upsample stage4 to stage3 resolution
        stage4_up = self.stage4_upsample(stage4)  # (B, 768, 32, 32)

        # Fuse
        fused = torch.cat([stage3, stage4_up], dim=1)  # (B, 1536, 32, 32)
        fused = self.fuse(fused)  # (B, 256, 32, 32)

        # Upsample to 128×128
        x = self.upsample1(fused)    # (B, 128, 64, 64)
        x = self.upsample2(x)        # (B, 64, 128, 128)
        mask = self.output_conv(x)   # (B, 3, 128, 128)

        return mask


class SSLModel(nn.Module):
    """
    Full SSL pretraining model combining contrastive learning + multi-task.

    Forward pass:
      1. Pass both views through the shared backbone
      2. Compute projection embeddings for contrastive loss
      3. Predict lesion type, count, and reconstruct masks

    Args:
        backbone_name: timm model name for SwinV2 backbone
        stage_channels: Channel dims for 4 SwinV2 stages
        projection_dim: Output dim for contrastive projection
        use_contrastive: Whether to include contrastive branch (for ablation)
        use_multitask: Whether to include multi-task branch (for ablation)
    """

    def __init__(self,
                 backbone_name='swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
                 stage_channels=(192, 384, 768, 1536),
                 projection_dim=128,
                 use_contrastive=True,
                 use_multitask=True):
        super().__init__()
        self.use_contrastive = use_contrastive
        self.use_multitask = use_multitask
        self.stage_channels = list(stage_channels)

        # Shared backbone (same as RetiNA-Net for compatibility)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            dynamic_img_size=True,
            img_size=512,

            drop_path_rate=0.1,
        )


        # Enable gradient checkpointing to reduce memory usage
        # Trades compute for memory (~40% activation memory reduction)
        self.backbone.set_grad_checkpointing(enable=True)

        in_ch_stage3 = self.stage_channels[2]  # 768

        in_ch_stage4 = self.stage_channels[3]   # 1536

        # Contrastive branch: projection head on stage-4 features
        if self.use_contrastive:
            self.projection_head = ProjectionHead(
                in_channels=in_ch_stage4,
                hidden_dim=512,
                out_dim=projection_dim
            )

        # Multi-task branches
        if self.use_multitask:
            self.lesion_type_head = LesionTypeHead(
                in_channels=in_ch_stage3,
                num_lesion_types=3
            )
            self.lesion_count_head = LesionCountHead(
                in_channels=in_ch_stage4,
                num_lesion_types=3
            )
            self.lesion_decoder = LesionDecoder(
                stage3_channels=in_ch_stage3,
                stage4_channels=in_ch_stage4,
                out_channels=3
            )

    def _extract_features(self, x):
        """
        Extract multi-stage features from backbone.
        Handles SwinV2's (B, H, W, C) → (B, C, H, W) conversion.

        Returns:
            stage3: (B, 768, 32, 32)
            stage4: (B, 1536, 16, 16)
        """
        features = self.backbone(x)
        # SwinV2 returns (B, H, W, C) format
        stage3 = features[2]
        stage4 = features[3]

        if stage3.shape[-1] == self.stage_channels[2]:
            stage3 = stage3.permute(0, 3, 1, 2).contiguous()
            stage4 = stage4.permute(0, 3, 1, 2).contiguous()

        return stage3, stage4

    def forward(self, view1, view2):
        """
        Forward pass for SSL pretraining.

        Args:
            view1: (B, 3, 512, 512) — first augmented view
            view2: (B, 3, 512, 512) — second augmented view

        Returns:
            dict with:
              - 'proj1': (B, projection_dim) or None — L2-normalized projection for view1
              - 'proj2': (B, projection_dim) or None — L2-normalized projection for view2
              - 'lesion_type1': (B, 3) or None — lesion type logits for view1
              - 'lesion_type2': (B, 3) or None — lesion type logits for view2
              - 'lesion_count1': (B, 3) or None — lesion count predictions for view1
              - 'lesion_count2': (B, 3) or None — lesion count predictions for view2
              - 'lesion_mask1': (B, 3, 128, 128) or None — reconstructed masks for view1
              - 'lesion_mask2': (B, 3, 128, 128) or None — reconstructed masks for view2
        """
        results = {
            'proj1': None, 'proj2': None,
            'lesion_type1': None, 'lesion_type2': None,
            'lesion_count1': None, 'lesion_count2': None,
            'lesion_mask1': None, 'lesion_mask2': None
        }

        # Process view1
        stage3_1, stage4_1 = self._extract_features(view1)

        if self.use_contrastive:
            pooled1 = F.adaptive_avg_pool2d(stage4_1, (1, 1)).flatten(1)  # (B, 1536)
            results['proj1'] = self.projection_head(pooled1)  # (B, 128)

        if self.use_multitask:
            results['lesion_type1'] = self.lesion_type_head(stage3_1)     # (B, 3)
            results['lesion_count1'] = self.lesion_count_head(stage4_1)   # (B, 3)
            results['lesion_mask1'] = self.lesion_decoder(stage3_1, stage4_1)  # (B, 3, 128, 128)

        # Process view2
        stage3_2, stage4_2 = self._extract_features(view2)

        if self.use_contrastive:
            pooled2 = F.adaptive_avg_pool2d(stage4_2, (1, 1)).flatten(1)  # (B, 1536)
            results['proj2'] = self.projection_head(pooled2)  # (B, 128)

        if self.use_multitask:
            results['lesion_type2'] = self.lesion_type_head(stage3_2)     # (B, 3)
            results['lesion_count2'] = self.lesion_count_head(stage4_2)   # (B, 3)
            results['lesion_mask2'] = self.lesion_decoder(stage3_2, stage4_2)  # (B, 3, 128, 128)

        return results

    def get_backbone_state_dict(self):
        """
        Extract only the backbone weights for transfer to RetiNA-Net.
        This is the key output of SSL pretraining — the learned backbone
        weights are loaded into RetiNA-Net for fine-tuning.
        """
        backbone_state = {}
        for name, param in self.backbone.named_parameters():
            backbone_state[name] = param.data.clone()
        return backbone_state

    def save_backbone(self, path):
        """Save backbone weights for use in RetiNA-Net fine-tuning."""
        torch.save(self.get_backbone_state_dict(), path)
        print(f"SSL pretrained backbone saved to {path}")


def info_nce_loss(proj1, proj2, temperature=0.07):
    """
    InfoNCE (NT-Xent) contrastive loss for SimCLR-style training.

    For each image, view1 and view2 form a positive pair.
    All other views in the batch are negatives.

    Args:
        proj1: (B, D) — L2-normalized projections for view1
        proj2: (B, D) — L2-normalized projections for view2
        temperature: Temperature scaling factor

    Returns:
        loss: scalar InfoNCE loss
    """
    batch_size = proj1.shape[0]

    # Concatenate: [view1_0, view1_1, ..., view1_{B-1}, view2_0, ..., view2_{B-1}]
    projections = torch.cat([proj1, proj2], dim=0)  # (2B, D)

    # Similarity matrix: (2B, 2B)
    sim = torch.matmul(projections, projections.T) / temperature

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=projections.device)
    sim.masked_fill_(mask, float('-inf'))


    # Labels: for i in [0, B), positive is i+B; for i in [B, 2B), positive is i-B
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=projections.device),  # view1 → view2
        torch.arange(0, batch_size, device=projections.device)                 # view2 → view1
    ])

    # Cross-entropy: each row should predict its positive pair
    loss = F.cross_entropy(sim, labels)

    return loss


class SSLCombinedLoss(nn.Module):
    """
    Combined SSL loss: Contrastive (InfoNCE) + Multi-Task (Lesion).

    L_pretrain = w_contrastive * L_contrastive
               + w_type * L_lesion_type (BCE)
               + w_count * L_lesion_count (MSE)
               + w_recon * L_reconstruction (L1)

    Weights can be set to 0 for ablation studies.
    """

    def __init__(self,
                 temperature=0.07,
                 w_contrastive=1.0,
                 w_lesion_type=0.5,
                 w_lesion_count=0.3,
                 w_reconstruction=0.2,
                 max_lesion_count=100.0,
                 use_contrastive=True,
                 use_multitask=True):
        super().__init__()
        self.temperature = temperature
        self.w_contrastive = w_contrastive if use_contrastive else 0.0
        self.w_lesion_type = w_lesion_type if use_multitask else 0.0
        self.w_lesion_count = w_lesion_count if use_multitask else 0.0
        self.w_reconstruction = w_reconstruction if use_multitask else 0.0
        self.max_lesion_count = max_lesion_count
        self.use_contrastive = use_contrastive
        self.use_multitask = use_multitask

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, lesion_mask, lesion_count, lesion_presence):
        """
        Args:
            outputs: dict from SSLModel.forward()
            lesion_mask: (B, 3, 128, 128) — ground truth lesion masks
            lesion_count: (B, 3) — ground truth lesion counts
            lesion_presence: (B, 3) — ground truth lesion presence (binary)

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        total_loss = torch.tensor(0.0, device=lesion_mask.device)
        loss_dict = {}

        # Contrastive loss
        if self.use_contrastive and outputs['proj1'] is not None:
            contrastive_loss = info_nce_loss(
                outputs['proj1'], outputs['proj2'], self.temperature
            )
            total_loss = total_loss + self.w_contrastive * contrastive_loss
            loss_dict['contrastive'] = contrastive_loss.item()
        else:
            loss_dict['contrastive'] = 0.0

        if self.use_multitask:
            # Lesion type loss (BCE on both views)
            if outputs['lesion_type1'] is not None:
                type_loss = self.bce_loss(outputs['lesion_type1'], lesion_presence)
                type_loss += self.bce_loss(outputs['lesion_type2'], lesion_presence)
                type_loss = type_loss / 2
                total_loss = total_loss + self.w_lesion_type * type_loss
                loss_dict['lesion_type'] = type_loss.item()
            else:
                loss_dict['lesion_type'] = 0.0

            # Lesion count loss (MSE, normalized to [0, 1])
            if outputs['lesion_count1'] is not None:
                count_norm = lesion_count / self.max_lesion_count  # (B, 3) in [0, 1]
                count_loss = self.mse_loss(outputs['lesion_count1'], count_norm)
                count_loss += self.mse_loss(outputs['lesion_count2'], count_norm)
                count_loss = count_loss / 2
                total_loss = total_loss + self.w_lesion_count * count_loss
                loss_dict['lesion_count'] = count_loss.item()
            else:
                loss_dict['lesion_count'] = 0.0

            # Reconstruction loss (L1 on lesion masks)
            if outputs['lesion_mask1'] is not None:
                recon_loss = self.l1_loss(outputs['lesion_mask1'], lesion_mask)
                recon_loss += self.l1_loss(outputs['lesion_mask2'], lesion_mask)
                recon_loss = recon_loss / 2
                total_loss = total_loss + self.w_reconstruction * recon_loss
                loss_dict['reconstruction'] = recon_loss.item()
            else:
                loss_dict['reconstruction'] = 0.0
        else:
            loss_dict['lesion_type'] = 0.0
            loss_dict['lesion_count'] = 0.0
            loss_dict['reconstruction'] = 0.0

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict
