import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.

    Focal Loss downweights well-classified examples and focuses gradient
    signal on hard, minority-class examples — critical for the heavily
    imbalanced APTOS-2019 dataset.

    gamma reduced from 2.0 to 1.5 — with reduced regularization, the model
    needs to learn ALL examples well, not just hard ones.
    """

    def __init__(self, gamma=1.5, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, weights=None):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=weights, reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class OrdinalLoss(nn.Module):
    """
    Ordinal regression loss using cumulative link model.

    For K classes, predicts K-1 cumulative logits P(y > k).
    Loss = sum of binary cross-entropy on each cumulative threshold.

    This enforces the ordinal structure of DR grades: misclassifying
    class 0 as class 4 is penalized more than class 2→3.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, cum_logits, targets):
        """
        Args:
            cum_logits: (B, K-1) cumulative logits from ordinal head
            targets: (B,) integer labels in [0, K-1]
        """
        # Create binary targets: for each threshold k, target is 1 if y > k
        # targets shape: (B,) → (B, K-1)
        targets_expanded = targets.unsqueeze(1)  # (B, 1)
        thresholds = torch.arange(self.num_classes - 1, device=targets.device).unsqueeze(0)  # (1, K-1)
        binary_targets = (targets_expanded > thresholds).float()  # (B, K-1)

        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(cum_logits, binary_targets, reduction='mean')
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal Loss (weighted) + Ordinal Loss + Auxiliary Loss.

    The classification head is trained with Focal Loss + class weights.
    The ordinal head is trained with OrdinalLoss.
    The auxiliary head (deep supervision) is trained with Focal Loss.
    Total loss = focal_loss + ordinal_weight * ordinal_loss + aux_weight * aux_loss.

    The model forward() returns a dict with 'logits', 'aux_logits', and 'ordinal_logits'.
    """

    def __init__(self, class_weights, device, label_smoothing=0.0,
                 use_ordinal=True, ordinal_loss_weight=0.3, num_classes=5,
                 focal_gamma=1.5, use_aux=True, aux_loss_weight=0.2):
        super(CombinedLoss, self).__init__()

        # Device placement for class weights
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.FloatTensor(class_weights)
        self.class_weights = class_weights.to(device)

        self.focal_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        self.use_ordinal = use_ordinal
        self.use_aux = use_aux

        if use_ordinal:
            self.ordinal_loss = OrdinalLoss(num_classes=num_classes)
            self.ordinal_loss_weight = ordinal_loss_weight

        if use_aux:
            self.aux_loss_weight = aux_loss_weight

    def forward(self, pred, targets):
        """
        Args:
            pred: dict with 'logits' (B, K), 'aux_logits' (B, K) or None,
                  and 'ordinal_logits' (B, K-1) or None
            targets: (B,) integer labels

        Returns:
            Total combined loss
        """
        # Handle both dict (new model) and tensor (old model) predictions
        if isinstance(pred, dict):
            logits = pred['logits']
            aux_logits = pred.get('aux_logits', None)
            ordinal_logits = pred.get('ordinal_logits', None)
        else:
            logits = pred
            aux_logits = None
            ordinal_logits = None

        # Focal loss with class weights (main classification head)
        cls_loss = self.focal_loss(logits, targets, weights=self.class_weights)

        total_loss = cls_loss

        # Auxiliary loss (deep supervision)
        if self.use_aux and aux_logits is not None:
            aux_loss = self.focal_loss(aux_logits, targets, weights=self.class_weights)
            total_loss = total_loss + self.aux_loss_weight * aux_loss

        # Ordinal loss
        if self.use_ordinal and ordinal_logits is not None:
            ord_loss = self.ordinal_loss(ordinal_logits, targets)
            total_loss = total_loss + self.ordinal_loss_weight * ord_loss

        return total_loss
