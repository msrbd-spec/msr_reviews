import torch
import numpy as np


def mixup_data(x, y, alpha=0.4):
    """
    Mixup augmentation: linearly interpolates between two images and their labels.

    Args:
        x: Input tensors (B, C, H, W)
        y: Labels (B,)
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation: cuts a rectangular region from one image and pastes
    it onto another. Labels are mixed proportionally to the area ratio.

    Args:
        x: Input tensors (B, C, H, W)
        y: Labels (B,)
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Generate random bounding box
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # Uniform random center
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    # Clamp bounding box to image boundaries
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    # Apply cutmix
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to actual area ratio (may differ due to clamping)
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for mixed labels.

    Args:
        criterion: Loss function that takes (pred, target) and returns scalar loss
        pred: Model predictions (dict with 'logits' and 'ordinal_logits')
        y_a: Labels from first image
        y_b: Labels from second image
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b


def apply_mixup_or_cutmix(x, y, mixup_alpha=0.4, cutmix_alpha=1.0,
                          mix_prob=0.5, cutmix_prob=0.5):
    """
    Randomly apply Mixup, CutMix, or no mixing to a batch.

    Args:
        x: Input tensors (B, C, H, W)
        y: Labels (B,)
        mixup_alpha: Beta parameter for Mixup
        cutmix_alpha: Beta parameter for CutMix
        mix_prob: Overall probability of applying any mixing
        cutmix_prob: Within mixing, probability of CutMix (rest is Mixup)

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if np.random.random() > mix_prob:
        # No mixing — return original
        return x, y, y, 1.0

    if np.random.random() < cutmix_prob:
        return cutmix_data(x, y, alpha=cutmix_alpha)
    else:
        return mixup_data(x, y, alpha=mixup_alpha)
