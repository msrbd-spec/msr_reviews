import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm

from .tta import predict_with_tta, predict_multiscale_tta
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models (e.g., K-fold models).

    Averages softmax probabilities across all models, then takes argmax.
    This reduces prediction variance and typically yields 2-5% accuracy
    improvement over a single model.
    """

    def __init__(self, models, weights=None):
        """
        Args:
            models: List of trained models
            weights: Optional list of weights for each model (default: equal)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def forward(self, x):
        """
        Forward pass through all models, averaging probabilities.
        Returns dict with 'logits' (averaged) and 'ordinal_logits' (averaged).
        """
        all_probs_cls = []
        all_ord_probs = []

        for model in self.models:
            out = model(x)
            if isinstance(out, dict):
                cls_probs = torch.softmax(out['logits'], dim=1)
                all_probs_cls.append(cls_probs)

                if out.get('ordinal_logits') is not None:
                    from ..models.components import OrdinalRegressionHead
                    ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(out['ordinal_logits'])
                    all_ord_probs.append(ord_probs)

        # Weighted average of classification probabilities
        avg_cls_probs = sum(w * p for w, p in zip(self.weights, all_probs_cls))

        # Weighted average of ordinal probabilities (if available)
        if all_ord_probs:
            avg_ord_probs = sum(w * p for w, p in zip(self.weights, all_ord_probs))
            # Combine classification and ordinal — 0.7/0.3 (classification dominant)
            final_probs = 0.7 * avg_cls_probs + 0.3 * avg_ord_probs
        else:
            final_probs = avg_cls_probs

        # Convert back to logits for compatibility
        final_logits = torch.log(final_probs + 1e-8)

        return {
            'logits': final_logits,
            'ordinal_logits': None,
            'probs': final_probs
        }

    def predict(self, x):
        """Return final probabilities."""
        out = self.forward(x)
        return out['probs']


@torch.no_grad()
def ensemble_predict(models, dataloader, device, use_tta=True,
                     use_multiscale=False, scales=(0.9, 1.0, 1.1),
                     use_flips=True, use_rotations=True, weights=None,
                     center_crops=(0.9, 0.95)):
    """
    Run ensemble inference on a dataloader.

    Args:
        models: List of trained models
        dataloader: DataLoader to evaluate on
        device: torch.device
        use_tta: Whether to use TTA
        use_multiscale: Whether to use multi-scale TTA
        scales: Scales for multi-scale TTA
        use_flips: Include flips in TTA
        use_rotations: Include rotations in TTA
        weights: Optional weights for each model

    Returns:
        all_preds, all_targets, all_probs
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    all_preds = []
    all_targets = []
    all_probs = []

    for model in models:
        model.to(device)
        model.eval()

    for inputs, targets in tqdm(dataloader, desc="Ensemble inference"):
        inputs = inputs.to(device)
        batch_probs = torch.zeros(inputs.size(0), 5, device=device)

        for model, weight in zip(models, weights):
            if use_multiscale:
                probs, _ = predict_multiscale_tta(
                    model, inputs, scales=scales,
                    use_flips=use_flips, use_rotations=use_rotations,
                    center_crops=center_crops
                )
            else:
                probs, _ = predict_with_tta(
                    model, inputs, use_tta=use_tta,
                    use_flips=use_flips, use_rotations=use_rotations
                )
            batch_probs += weight * probs

        preds = torch.argmax(batch_probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_probs.extend(batch_probs.cpu().numpy())

    return all_preds, all_targets, all_probs


def load_ensemble_models(model_class, model_paths, device, **model_kwargs):
    """
    Load multiple model checkpoints into separate model instances.

    Args:
        model_class: Model class (e.g., RetiNA_Net)
        model_paths: List of checkpoint paths
        device: torch.device
        **model_kwargs: Additional model constructor arguments

    Returns:
        List of loaded models
    """
    models = []
    for path in model_paths:
        model = model_class(**model_kwargs).to(device)
        state_dict = torch.load(path, map_location=device)

        # Handle SWA models that may have different BN stats
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        logger.info(f"Loaded model from {path}")

    return models


def evaluate_ensemble(models, dataloader, device, use_tta=True,
                      use_multiscale=False, scales=(0.9, 1.0, 1.1),
                      use_flips=True, use_rotations=True, weights=None,
                      center_crops=(0.9, 0.95)):
    """
    Evaluate ensemble and return metrics.

    Args:
        models: List of trained models
        dataloader: DataLoader to evaluate on
        device: torch.device
        use_tta: Whether to use TTA
        use_multiscale: Whether to use multi-scale TTA
        scales: Scales for multi-scale TTA
        use_flips: Include flips in TTA
        use_rotations: Include rotations in TTA
        weights: Optional weights for each model

    Returns:
        metrics dict
    """
    all_preds, all_targets, all_probs = ensemble_predict(
        models, dataloader, device,
        use_tta=use_tta, use_multiscale=use_multiscale, scales=scales,
        use_flips=use_flips, use_rotations=use_rotations, weights=weights,
        center_crops=center_crops
    )

    metrics = compute_metrics(all_targets, all_preds)
    return metrics, all_preds, all_targets, all_probs
