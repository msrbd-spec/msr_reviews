import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


class MSDABlock(nn.Module):
    """
    Multi-Scale Deformable Attention Block wrapping DeformConv2d.

    Uses deform_groups=4 independent offset fields — enough spatial
    flexibility to capture distinct lesion-shape deformation patterns
    while keeping the offset predictor lightweight.
    """

    def __init__(self, in_channels: int, kernel_size: int = 3, deform_groups: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.deform_groups = deform_groups

        # Offset (2 channels) + mask (1 channel) per deform group, per kernel tap.
        num_offset_channels = 3 * (kernel_size ** 2) * deform_groups

        self.offset_conv = nn.Conv2d(
            in_channels,
            num_offset_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True
        )

        # Initialize offsets to 0
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.deform_conv = ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )

        # LayerNorm for stabilising deformable conv output
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x):
        # Predict offsets and mask
        offset_and_mask = self.offset_conv(x)

        # Split into offsets and mask
        o1 = 2 * (self.kernel_size ** 2) * self.deform_groups
        o2 = 1 * (self.kernel_size ** 2) * self.deform_groups

        offsets, mask = torch.split(offset_and_mask, [o1, o2], dim=1)
        mask = torch.sigmoid(mask)

        out = self.deform_conv(x, offsets, mask=mask)

        # Residual connection + LayerNorm
        out = out + x
        # out shape: (B, C, H, W) → permute for LayerNorm → back
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out


class HFFBlock(nn.Module):
    """
    Hierarchical Feature Fusion Block with learnable gating across all 4 stages.

    Updated for SwinV2-Large channel dims:
      - Stage 1 (192ch)  → stride-8 projection → 1536ch
      - Stage 2 (384ch)  → stride-4 projection → 1536ch
      - Stage 3 (768ch)  → stride-2 projection → 1536ch (post-MSDA)
      - Stage 4 (1536ch) → identity

    Pipeline:
      1. Project each stage to 1536ch at target spatial resolution
      2. LayerNorm on each projected stage
      3. Learnable softmax gates (4 gates, deeper stages initialised higher)
      4. Gated fusion: fused = Σ gate_i * stage_i_normed
      5. Residual: fused = fused + stage4 (preserve strongest semantic signal)
      6. Output projection: 2-layer MLP per spatial location
    """

    def __init__(self, stage_channels=(192, 384, 768, 1536), target_channels=1536):
        super().__init__()
        self.target_channels = target_channels

        # Projection convs: each stage → target_channels, downsample to stage4 resolution
        # Stage 1: 192ch → 1536ch (stride 8)
        self.proj_stage1 = nn.Conv2d(stage_channels[0], target_channels, kernel_size=8, stride=8)
        # Stage 2: 384ch → 1536ch (stride 4)
        self.proj_stage2 = nn.Conv2d(stage_channels[1], target_channels, kernel_size=4, stride=4)
        # Stage 3: 768ch → 1536ch (stride 2)
        self.proj_stage3 = nn.Conv2d(stage_channels[2], target_channels, kernel_size=2, stride=2)
        # Stage 4: already target_channels — no projection needed

        # LayerNorm on channel dimension for each stage
        self.stage1_norm = nn.LayerNorm(target_channels)
        self.stage2_norm = nn.LayerNorm(target_channels)
        self.stage3_norm = nn.LayerNorm(target_channels)
        self.stage4_norm = nn.LayerNorm(target_channels)

        # Learnable gates (4 stages) — initialised so deeper stages get more weight
        self.gates = nn.Parameter(torch.tensor([-1.0, -0.5, 0.5, 1.5]))

        # Output projection after fusion (per spatial location)
        self.output_proj = nn.Sequential(
            nn.Linear(target_channels, target_channels),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(target_channels, target_channels),
        )

    def _project_and_norm(self, x, proj_conv, norm_layer):
        """Project spatial features to target dims and apply LayerNorm on channel dim."""
        projected = proj_conv(x)  # (B, C, H, W)
        projected = projected.permute(0, 2, 3, 1)  # (B, H, W, C)
        projected = norm_layer(projected)
        projected = projected.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return projected

    def forward(self, stage1_features, stage2_features, stage3_features, stage4_features):
        # Project each stage to target_channels and normalise
        proj1 = self._project_and_norm(stage1_features, self.proj_stage1, self.stage1_norm)
        proj2 = self._project_and_norm(stage2_features, self.proj_stage2, self.stage2_norm)
        proj3 = self._project_and_norm(stage3_features, self.proj_stage3, self.stage3_norm)

        # Stage 4: just LayerNorm (no projection needed — already target_channels)
        proj4 = stage4_features.permute(0, 2, 3, 1)
        proj4 = self.stage4_norm(proj4)
        proj4 = proj4.permute(0, 3, 1, 2).contiguous()

        # Softmax gates: 4 weights summing to 1
        gate_weights = torch.softmax(self.gates, dim=0)

        # Gated fusion
        fused = (gate_weights[0] * proj1 +
                 gate_weights[1] * proj2 +
                 gate_weights[2] * proj3 +
                 gate_weights[3] * proj4)

        # Residual connection with stage4 (strongest semantic signal)
        fused = fused + stage4_features

        # Output projection (applied per spatial location)
        fused_flat = fused.permute(0, 2, 3, 1)  # (B, H, W, C)
        fused_flat = self.output_proj(fused_flat)
        fused = fused_flat.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return fused


class SpatialAttentionPooling(nn.Module):
    """
    Spatial Attention Pooling for classification heads.

    Instead of uniform average/max pooling, this module learns a spatial
    attention map that highlights lesion-relevant regions. A learnable
    query vector attends over spatial locations, producing a weighted
    sum of features. This is particularly useful for DR grading where
    small lesions (microaneurysms, hemorrhages) occupy only a few
    spatial locations.

    Produces both attention-pooled and max-pooled features (concatenated)
    for a richer representation.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Spatial attention: 1x1 conv → sigmoid → spatial weights
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.attention_conv.weight, 0.)
        nn.init.constant_(self.attention_conv.bias, 0.)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        # Compute spatial attention
        attn = self.attention_conv(x)  # (B, 1, H, W)
        attn = torch.sigmoid(attn)

        # Attention-weighted pooling
        weighted = x * attn  # (B, C, H, W)
        attn_pooled = self.avg_pool(weighted)  # (B, C, 1, 1)
        attn_pooled = torch.flatten(attn_pooled, 1)  # (B, C)

        # Max pooling (captures peak activations — small lesions)
        max_pooled = self.max_pool(x)  # (B, C, 1, 1)
        max_pooled = torch.flatten(max_pooled, 1)  # (B, C)

        # Concatenate attention-pooled + max-pooled → (B, 2C)
        pooled = torch.cat([attn_pooled, max_pooled], dim=1)
        return pooled


class MultiScaleHead(nn.Module):
    """
    Multi-scale classification head with optional spatial attention pooling.

    When use_attention_pool=True:
      1. Spatial attention pooling → 2*in_channels-dim
      2. 2-layer MLP with BatchNorm, GELU, Dropout
      3. Linear → num_classes

    When use_attention_pool=False (legacy):
      1. Dual pooling (AvgPool + MaxPool) → 2*in_channels-dim
      2. Same MLP
    """

    def __init__(self, in_channels: int = 1536, num_classes: int = 5,
                 dropout: float = 0.1, use_attention_pool: bool = True):
        super().__init__()
        self.use_attention_pool = use_attention_pool

        if use_attention_pool:
            self.pool = SpatialAttentionPooling(in_channels)
            pooled_dim = in_channels * 2
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            pooled_dim = in_channels * 2

        # 2-layer MLP head
        self.fc1 = nn.Linear(pooled_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.use_attention_pool:
            pooled = self.pool(x)
        else:
            avg_out = self.avg_pool(x)
            max_out = self.max_pool(x)
            pooled = torch.cat([avg_out, max_out], dim=1)
            pooled = torch.flatten(pooled, 1)

        # MLP
        out = self.fc1(pooled)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.fc2(out)

        return out


class OrdinalRegressionHead(nn.Module):
    """
    Ordinal regression head for DR grading.

    Predicts K-1 cumulative probabilities P(y > k) for k=0..K-2.
    The final class prediction is derived from these cumulative probs.
    This head enforces the ordinal structure of DR grades (0<1<2<3<4).
    """

    def __init__(self, in_channels: int = 1536, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # K-1 binary classifiers for cumulative probabilities
        self.ordinal_layers = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_classes - 1)
        ])

    def forward(self, x):
        pooled = self.avg_pool(x)
        pooled = torch.flatten(pooled, 1)
        features = self.fc(pooled)

        # K-1 cumulative logits
        cum_logits = torch.cat([layer(features) for layer in self.ordinal_layers], dim=1)
        return cum_logits

    @staticmethod
    def ordinal_logits_to_class_probs(cum_logits):
        """
        Convert K-1 cumulative logits to K class probabilities.

        P(y > k) = sigmoid(cum_logit_k)
        P(y = k) = P(y > k-1) - P(y > k)
        """
        cum_probs = torch.sigmoid(cum_logits)  # (B, K-1)

        # P(y > -1) = 1, P(y > K-1) = 0
        ones = torch.ones(cum_probs.shape[0], 1, device=cum_probs.device)
        zeros = torch.zeros(cum_probs.shape[0], 1, device=cum_probs.device)
        cum_probs_padded = torch.cat([ones, cum_probs, zeros], dim=1)  # (B, K+1)

        # P(y = k) = P(y > k-1) - P(y > k)
        class_probs = cum_probs_padded[:, :-1] - cum_probs_padded[:, 1:]
        return class_probs


class AuxHead(nn.Module):
    """
    Auxiliary classification head for deep supervision.

    Applied to stage-3 features (before HFF fusion) to force intermediate
    features to be discriminative. This improves gradient flow to the
    backbone and acts as a regularizer.
    """

    def __init__(self, in_channels: int = 768, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        pooled = self.avg_pool(x)
        pooled = torch.flatten(pooled, 1)
        out = self.fc1(pooled)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out
