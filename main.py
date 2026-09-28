import os
# Force HuggingFace to use local cache only — no network requests (cluster has no internet)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# Reduce CUDA memory fragmentation for large models on 40GB A100
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import yaml
import torch
import logging
import datetime
import numpy as np

from src.utils.logger import setup_logger
from src.data.datamodule import get_dataloaders
from src.models.dr_model import RetiNA_Net
from src.models.components import OrdinalRegressionHead
from src.training.loss import CombinedLoss
from src.training.trainer import DRTrainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import plot_training_curves, plot_confusion_matrix, plot_roc_curve
from src.evaluation.xai import generate_heatmap
from src.evaluation.tta import predict_with_tta, predict_multiscale_tta
from src.evaluation.ensemble import load_ensemble_models, evaluate_ensemble
import cv2

# SSL pretraining imports (Phase 2-3 of plan.md)
from src.preprocessing.lesion_detection import detect_lesions_batch
from src.data.eyepacs_dataset import get_ssl_dataloader
from src.models.ssl_model import SSLModel
from src.training.ssl_trainer import SSLTrainer

# Tables & charts generation imports (Phase 8 of plan.md)
from src.evaluation.generate_tables import (
    compute_all_metrics, generate_classification_report_text,
    generate_architecture_ablation_table,
    generate_ssl_ablation_table, generate_training_ablation_table,
    generate_sota_comparison_table,
    generate_per_class_metrics_table,
    generate_external_validation_table
)
from src.evaluation.generate_charts import (
    plot_confusion_matrix as plot_cm_v2,
    plot_roc_curves, plot_pr_curves, plot_ablation_bar_chart,
    plot_ssl_pretraining_curves, plot_lesion_correlation,
    plot_radar_chart, plot_xai_grid
)





def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_ablation_flags(ablation):
    if ablation == 'baseline':
        return False, False
    elif ablation == 'msda_only':
        return True, False
    elif ablation == 'hff_only':
        return False, True
    elif ablation == 'proposed':
        return True, True
    else:
        raise ValueError(f"Unknown ablation mode: {ablation}")


def create_model(config, ablation, device):
    """Create model with config parameters."""
    use_msda, use_hff = get_ablation_flags(ablation)
    use_ordinal = config.get("use_ordinal_loss", True)
    drop_path_rate = config.get("drop_path_rate", 0.1)
    dropout = config.get("dropout", 0.1)
    num_classes = config.get("num_classes", 5)
    use_aux_head = config.get("use_aux_head", True)
    use_attention_pool = config.get("use_attention_pool", True)
    backbone_name = config.get("backbone", "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k")
    stage_channels = config.get("backbone_channels", [192, 384, 768, 1536])

    # SSL pretrained backbone path (if available, loads SSL weights instead of ImageNet)
    ssl_pretrained_path = config.get("ssl_pretrained_path", None)

    model = RetiNA_Net(
        use_msda=use_msda,
        use_hff=use_hff,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        dropout=dropout,
        use_ordinal=use_ordinal,
        use_aux_head=use_aux_head,
        use_attention_pool=use_attention_pool,
        backbone_name=backbone_name,
        stage_channels=tuple(stage_channels),
        ssl_pretrained_path=ssl_pretrained_path
    ).to(device)
    return model



def create_criterion(config, class_weights, device):
    """Create loss function with config parameters."""
    use_ordinal = config.get("use_ordinal_loss", True)
    ordinal_weight = config.get("ordinal_loss_weight", 0.3)
    label_smoothing = config.get("label_smoothing", 0.0)
    num_classes = config.get("num_classes", 5)
    focal_gamma = config.get("focal_gamma", 1.5)
    use_aux = config.get("use_aux_head", True)
    aux_loss_weight = config.get("aux_loss_weight", 0.2)

    criterion = CombinedLoss(
        class_weights=class_weights,
        device=device,
        label_smoothing=label_smoothing,
        use_ordinal=use_ordinal,
        ordinal_loss_weight=ordinal_weight,
        num_classes=num_classes,
        focal_gamma=focal_gamma,
        use_aux=use_aux,
        aux_loss_weight=aux_loss_weight
    )
    return criterion


def train_single_fold(config, ablation, device, logger, fold_idx=None):
    """Train a single fold (or standard train/val split)."""
    train_loader, val_loader, _, _, class_weights = get_dataloaders(config, fold_idx=fold_idx)

    model = create_model(config, ablation, device)
    criterion = create_criterion(config, class_weights, device)
    trainer = DRTrainer(
        model, train_loader, val_loader, criterion, device, config,
        ablation=ablation, fold_idx=fold_idx
    )

    train_losses, val_losses, train_accs, val_accs = trainer.train()
    return train_losses, val_losses, train_accs, val_accs


def train_kfold(config, ablation, device, logger, timestamp):
    """Train K-fold cross-validation models."""
    n_folds = config.get("n_folds", 5)
    all_train_losses = []
    all_val_losses = []
    all_val_accs = []

    for fold_idx in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold_idx + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        train_losses, val_losses, train_accs, val_accs = train_single_fold(
            config, ablation, device, logger, fold_idx=fold_idx
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_accs.append(val_accs)

        # Plot training curves for this fold
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            filename=os.path.join('results', f'training_curves_{ablation}_fold{fold_idx}_{timestamp}.png')
        )

    logger.info(f"\nK-Fold training complete. {n_folds} models saved.")
    return all_train_losses, all_val_losses, all_val_accs


def run_test(config, ablation, device, logger, timestamp):
    """Run test evaluation with optional ensemble."""
    _, _, test_loader, _, _ = get_dataloaders(config, fold_idx=None)

    use_kfold = config.get("use_kfold", True)
    n_folds = config.get("n_folds", 5)
    use_tta = config.get("use_tta", True)
    tta_scales = config.get("tta_scales", [0.8, 0.9, 1.0, 1.1, 1.2])
    tta_flips = config.get("tta_flips", True)
    tta_rotations = config.get("tta_rotations", True)
    tta_center_crops = config.get("tta_center_crops", [0.9, 0.95])
    ensemble_folds = config.get("ensemble_folds", True)

    if use_kfold and ensemble_folds:
        # Ensemble inference across K-fold models
        model_paths = [f'checkpoints/best_model_{ablation}_fold{f}.pth' for f in range(n_folds)]
        model_paths = [p for p in model_paths if os.path.exists(p)]

        if len(model_paths) == 0:
            logger.error("No fold models found. Please train with K-fold first.")
            return

        logger.info(f"Loading ensemble of {len(model_paths)} fold models...")

        use_msda, use_hff = get_ablation_flags(ablation)
        use_ordinal = config.get("use_ordinal_loss", True)
        use_aux_head = config.get("use_aux_head", True)
        use_attention_pool = config.get("use_attention_pool", True)
        backbone_name = config.get("backbone", "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k")
        stage_channels = config.get("backbone_channels", [192, 384, 768, 1536])

        model_kwargs = {
            'use_msda': use_msda,
            'use_hff': use_hff,
            'num_classes': config.get("num_classes", 5),
            'drop_path_rate': config.get("drop_path_rate", 0.1),
            'dropout': config.get("dropout", 0.1),
            'use_ordinal': use_ordinal,
            'use_aux_head': use_aux_head,
            'use_attention_pool': use_attention_pool,
            'backbone_name': backbone_name,
            'stage_channels': tuple(stage_channels),
        }

        models = load_ensemble_models(RetiNA_Net, model_paths, device, **model_kwargs)

        logger.info("Running ensemble inference with multi-scale TTA...")
        metrics, all_preds, all_targets, all_probs = evaluate_ensemble(
            models, test_loader, device,
            use_tta=use_tta,
            use_multiscale=True,
            scales=tuple(tta_scales),
            use_flips=tta_flips,
            use_rotations=tta_rotations,
            center_crops=tuple(tta_center_crops)
        )
    else:
        # Single model inference
        model = create_model(config, ablation, device)
        model_path = f'checkpoints/best_model_{ablation}.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                if use_tta:
                    probs, preds = predict_multiscale_tta(
                        model, inputs,
                        scales=tuple(tta_scales),
                        use_flips=tta_flips,
                        use_rotations=tta_rotations,
                        center_crops=tuple(tta_center_crops)
                    )
                else:
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                        ordinal_logits = outputs.get('ordinal_logits')
                        if ordinal_logits is not None:
                            cls_probs = torch.softmax(logits, dim=1)
                            ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(ordinal_logits)
                            probs = 0.7 * cls_probs + 0.3 * ord_probs
                        else:
                            probs = torch.softmax(logits, dim=1)
                    else:
                        probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(all_targets, all_preds)

    logger.info(f"Test Metrics - Acc: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_macro']:.4f}, QWK: {metrics['qwk']:.4f}")
    logger.info(f"Test Classification Report:\n{metrics['classification_report']}")

    plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_{ablation}_{timestamp}.png'))
    plot_roc_curve(all_targets, np.array(all_probs), filename=os.path.join('results', f'roc_multiclass_{ablation}_{timestamp}.png'))

    # Save predictions for later table/chart generation (Phase 8)
    np.savez(
        os.path.join('results', 'test_predictions.npz'),
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        ablation=ablation
    )
    logger.info(f"Test predictions saved to results/test_predictions.npz (for generate_results mode)")



def run_external_validation(config, ablation, device, logger, timestamp):
    """Run external validation on Messidor-2 with optional ensemble."""
    _, _, _, ext_loader, _ = get_dataloaders(config, fold_idx=None)

    if ext_loader is None:
        logger.error("External validation dataset not configured.")
        return

    use_kfold = config.get("use_kfold", True)
    n_folds = config.get("n_folds", 5)
    use_tta = config.get("use_tta", True)
    tta_scales = config.get("tta_scales", [0.8, 0.9, 1.0, 1.1, 1.2])
    tta_flips = config.get("tta_flips", True)
    tta_rotations = config.get("tta_rotations", True)
    tta_center_crops = config.get("tta_center_crops", [0.9, 0.95])
    ensemble_folds = config.get("ensemble_folds", True)

    if use_kfold and ensemble_folds:
        model_paths = [f'checkpoints/best_model_{ablation}_fold{f}.pth' for f in range(n_folds)]
        model_paths = [p for p in model_paths if os.path.exists(p)]

        if len(model_paths) == 0:
            logger.error("No fold models found. Please train with K-fold first.")
            return

        logger.info(f"Loading ensemble of {len(model_paths)} fold models for external validation...")

        use_msda, use_hff = get_ablation_flags(ablation)
        use_ordinal = config.get("use_ordinal_loss", True)
        use_aux_head = config.get("use_aux_head", True)
        use_attention_pool = config.get("use_attention_pool", True)
        backbone_name = config.get("backbone", "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k")
        stage_channels = config.get("backbone_channels", [192, 384, 768, 1536])

        model_kwargs = {
            'use_msda': use_msda,
            'use_hff': use_hff,
            'num_classes': config.get("num_classes", 5),
            'drop_path_rate': config.get("drop_path_rate", 0.1),
            'dropout': config.get("dropout", 0.1),
            'use_ordinal': use_ordinal,
            'use_aux_head': use_aux_head,
            'use_attention_pool': use_attention_pool,
            'backbone_name': backbone_name,
            'stage_channels': tuple(stage_channels),
        }

        models = load_ensemble_models(RetiNA_Net, model_paths, device, **model_kwargs)

        metrics, all_preds, all_targets, all_probs = evaluate_ensemble(
            models, ext_loader, device,
            use_tta=use_tta,
            use_multiscale=True,
            scales=tuple(tta_scales),
            use_flips=tta_flips,
            use_rotations=tta_rotations,
            center_crops=tuple(tta_center_crops)
        )
    else:
        model = create_model(config, ablation, device)
        model_path = f'checkpoints/best_model_{ablation}.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for inputs, targets in ext_loader:
                inputs = inputs.to(device)
                if use_tta:
                    probs, preds = predict_multiscale_tta(
                        model, inputs,
                        scales=tuple(tta_scales),
                        use_flips=tta_flips,
                        use_rotations=tta_rotations,
                        center_crops=tuple(tta_center_crops)
                    )
                else:
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        probs = torch.softmax(outputs['logits'], dim=1)
                    else:
                        probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(all_targets, all_preds)

    logger.info(f"External Validation Metrics - Acc: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_macro']:.4f}, QWK: {metrics['qwk']:.4f}")
    logger.info(f"External Validation Classification Report:\n{metrics['classification_report']}")

    plot_confusion_matrix(all_targets, all_preds, filename=os.path.join('results', f'confusion_matrix_external_{ablation}_{timestamp}.png'))

    # Save external predictions for later table/chart generation (Phase 8)
    np.savez(
        os.path.join('results', 'external_predictions.npz'),
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        ablation=ablation
    )
    logger.info(f"External predictions saved to results/external_predictions.npz (for generate_results mode)")


def run_detect_lesions(config, logger, timestamp):

    """
    Phase 2: Run classical lesion detection on EyePACS images.
    Generates .npz pseudo-label files for SSL pretraining.
    """
    ssl_config_path = config.get('ssl_config', 'configs/config_ssl.yaml')
    if os.path.exists(ssl_config_path):
        ssl_config = load_config(ssl_config_path)
    else:
        logger.error(f"SSL config not found: {ssl_config_path}")
        return

    image_dirs = ssl_config.get('ssl_image_dirs', [])
    lesion_label_dir = ssl_config.get('ssl_lesion_label_dir', 'datasets/EyePACS/lesion_labels')
    mask_size = ssl_config.get('ssl_mask_size', 128)

    if not image_dirs:
        logger.error("No ssl_image_dirs specified in config.")
        return

    logger.info(f"Starting lesion detection on images from: {image_dirs}")
    logger.info(f"Output lesion labels: {lesion_label_dir}")
    logger.info(f"Mask size: {mask_size}")

    detect_lesions_batch(image_dirs, lesion_label_dir, mask_size=mask_size)

    logger.info("Lesion detection complete!")


def run_pretrain(config, device, logger, timestamp):
    """
    Phase 3: Run SSL pretraining on EyePACS dataset.
    Saves pretrained backbone weights for fine-tuning.
    """
    ssl_config_path = config.get('ssl_config', 'configs/config_ssl.yaml')
    if os.path.exists(ssl_config_path):
        ssl_config = load_config(ssl_config_path)
    else:
        logger.error(f"SSL config not found: {ssl_config_path}")
        return

    # Merge SSL config into config for SSLTrainer
    for key, val in ssl_config.items():
        config[key] = val

    image_dirs = ssl_config.get('ssl_image_dirs', [])
    lesion_label_dir = ssl_config.get('ssl_lesion_label_dir', 'datasets/EyePACS/lesion_labels')
    img_size = ssl_config.get('ssl_img_size', 512)
    mask_size = ssl_config.get('ssl_mask_size', 128)
    batch_size = ssl_config.get('ssl_batch_size', 32)
    num_workers = ssl_config.get('ssl_num_workers', 8)

    backbone_name = ssl_config.get('ssl_backbone', 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k')
    projection_dim = ssl_config.get('ssl_projection_dim', 128)
    use_contrastive = ssl_config.get('ssl_use_contrastive', True)
    use_multitask = ssl_config.get('ssl_use_multitask', True)

    # Create dataloader
    logger.info("Creating SSL dataloader...")
    train_loader = get_ssl_dataloader(
        image_dirs=image_dirs,
        lesion_label_dir=lesion_label_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        mask_size=mask_size
    )

    # Create SSL model
    logger.info("Creating SSL model...")
    ssl_model = SSLModel(
        backbone_name=backbone_name,
        projection_dim=projection_dim,
        use_contrastive=use_contrastive,
        use_multitask=use_multitask
    )

    # Create trainer
    trainer = SSLTrainer(
        model=ssl_model,
        train_loader=train_loader,
        config=config,
        device=device,
        logger=logger
    )

    # Run pretraining
    loss_history = trainer.train()

    # Plot SSL pretraining curves
    os.makedirs('results/figures', exist_ok=True)
    plot_ssl_pretraining_curves(loss_history, output_dir='results/figures')

    logger.info("SSL pretraining complete! Backbone saved to checkpoints/ssl_pretrained_backbone.pth")


def run_generate_results(config, logger, timestamp):
    """
    Phase 8: Generate all tables and charts from saved results.
    Reads saved test predictions and SSL loss history, then generates:
      - 6 CSV tables (architecture/SSL/training ablation, per-class, external, SOTA)
      - 7+ chart types (confusion matrix, ROC, PR, ablation bars, SSL curves, radar, etc.)

    Prerequisites:
      - Run `--mode test` first (saves results/test_predictions.npz)
      - Run `--mode pretrain` first (saves results/ssl_loss_history.npz) [optional]
      - Run `--mode external_validation` first (saves results/external_predictions.npz) [optional]
    """
    logger.info("Generating tables and charts...")

    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    CLASS_NAMES = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']
    num_classes = config.get("num_classes", 5)

    # ===================================================================
    # 1. Load test predictions (from run_test)
    # ===================================================================
    pred_path = os.path.join('results', 'test_predictions.npz')
    y_true, y_pred, y_prob = None, None, None
    test_ablation = 'proposed'

    if os.path.exists(pred_path):
        logger.info(f"Loading test predictions from {pred_path}...")
        pred_data = np.load(pred_path, allow_pickle=True)
        y_true = pred_data['y_true']
        y_pred = pred_data['y_pred']
        y_prob = pred_data['y_prob']
        test_ablation = str(pred_data['ablation']) if 'ablation' in pred_data else 'proposed'
        logger.info(f"  Loaded {len(y_true)} predictions (ablation: {test_ablation})")
    else:
        logger.warning(f"No test predictions found at {pred_path}. Run `--mode test` first.")
        logger.warning("  Tables/charts requiring predictions will be skipped.")

    # ===================================================================
    # 2. Generate tables from predictions
    # ===================================================================
    if y_true is not None and y_prob is not None:
        logger.info("Generating per-class metrics table...")
        generate_per_class_metrics_table(
            y_true, y_pred, y_prob=y_prob,
            num_classes=num_classes, class_names=CLASS_NAMES
        )

        logger.info("Generating classification report...")
        generate_classification_report_text(y_true, y_pred, class_names=CLASS_NAMES)

        # Compute full metrics for architecture ablation table
        logger.info("Computing full metrics for architecture ablation table...")
        full_metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob, num_classes=num_classes)
        arch_results = {test_ablation: full_metrics}

        # If there are previously saved ablation results, merge them
        # (This allows accumulating results from multiple --ablation runs)
        existing_arch_csv = os.path.join('results/tables', 'ablation_architecture.csv')
        if os.path.exists(existing_arch_csv):
            import pandas as pd
            try:
                existing_df = pd.read_csv(existing_arch_csv)
                for _, row in existing_df.iterrows():
                    name = row['Model']
                    if name not in arch_results:
                        # Parse saved metrics back from CSV string values
                        arch_results[name] = {
                            'accuracy': float(row.get('Accuracy (%)', 0)) / 100,
                            'precision_macro': float(row.get('Precision (%)', 0)) / 100,
                            'recall_macro': float(row.get('Recall (%)', 0)) / 100,
                            'f1_macro': float(row.get('F1 (macro) (%)', 0)) / 100,
                            'f1_weighted': float(row.get('F1 (weighted) (%)', 0)) / 100,
                            'qwk': float(row.get('QWK', 0)),
                            'kappa': float(row.get('Kappa', 0)),
                            'auc_macro': float(row.get('AUC', 0)),
                        }
            except Exception as e:
                logger.warning(f"  Could not merge existing arch CSV: {e}")

        logger.info("Generating architecture ablation table...")
        generate_architecture_ablation_table(arch_results)

    # ===================================================================
    # 3. Generate charts from predictions
    # ===================================================================
    if y_true is not None:
        logger.info("Generating confusion matrix (raw)...")
        plot_cm_v2(
            y_true, y_pred, class_names=CLASS_NAMES, normalize=False,
            title=f'Confusion Matrix — {test_ablation}',
            output_path=os.path.join('results/figures', 'confusion_matrix_raw.png')
        )

        logger.info("Generating confusion matrix (normalized)...")
        plot_cm_v2(
            y_true, y_pred, class_names=CLASS_NAMES, normalize=True,
            title=f'Normalized Confusion Matrix — {test_ablation}',
            output_path=os.path.join('results/figures', 'confusion_matrix_normalized.png')
        )

    if y_true is not None and y_prob is not None:
        logger.info("Generating ROC curves...")
        plot_roc_curves(
            y_true, y_prob, num_classes=num_classes, class_names=CLASS_NAMES,
            output_path=os.path.join('results/figures', 'roc_multiclass.png')
        )

        logger.info("Generating Precision-Recall curves...")
        plot_pr_curves(
            y_true, y_prob, num_classes=num_classes, class_names=CLASS_NAMES,
            output_path=os.path.join('results/figures', 'pr_curve.png')
        )

        # Per-class radar chart
        logger.info("Generating per-class radar chart...")
        per_class_metrics = {}
        for i, cls_name in enumerate(CLASS_NAMES):
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)
            tp = ((binary_pred == 1) & (binary_true == 1)).sum()
            fn = ((binary_pred == 0) & (binary_true == 1)).sum()
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            tn = ((binary_pred == 0) & (binary_true == 0)).sum()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
            per_class_metrics[cls_name] = {
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'F1': f1,
                'Precision': precision,
            }
        plot_radar_chart(
            per_class_metrics, metric_names=['Sensitivity', 'Specificity', 'F1', 'Precision'],
            class_names=CLASS_NAMES,
            output_path=os.path.join('results/figures', 'radar_per_class.png')
        )

    # ===================================================================
    # 4. Generate SSL pretraining curves (if loss history exists)
    # ===================================================================
    ssl_loss_path = os.path.join('results', 'ssl_loss_history.npz')
    if os.path.exists(ssl_loss_path):
        logger.info("Generating SSL pretraining curves...")
        loss_data = np.load(ssl_loss_path)
        loss_history = {k: loss_data[k].tolist() for k in loss_data.files}
        plot_ssl_pretraining_curves(loss_history, output_dir='results/figures')
    else:
        logger.info("No SSL loss history found. Skipping SSL curves.")

    # ===================================================================
    # 5. Generate ablation bar charts from saved CSV tables
    # ===================================================================
    arch_csv = os.path.join('results/tables', 'ablation_architecture.csv')
    if os.path.exists(arch_csv):
        import pandas as pd
        df = pd.read_csv(arch_csv)
        if len(df) > 0:
            accs = []
            for val in df['Accuracy (%)']:
                try:
                    accs.append(float(val))
                except (ValueError, TypeError):
                    accs.append(0.0)
            highlight_idx = len(accs) - 1
            plot_ablation_bar_chart(
                names=df['Model'].tolist(), values=accs,
                title='Architecture Ablation — Test Accuracy',
                ylabel='Accuracy (%)',
                output_path=os.path.join('results/figures', 'ablation_arch_bar.png'),
                highlight_idx=highlight_idx
            )
            logger.info("Generated architecture ablation bar chart.")

    ssl_csv = os.path.join('results/tables', 'ablation_ssl.csv')
    if os.path.exists(ssl_csv):
        import pandas as pd
        df = pd.read_csv(ssl_csv)
        if len(df) > 0:
            accs = []
            for val in df['Test Acc (%)']:
                try:
                    accs.append(float(val))
                except (ValueError, TypeError):
                    accs.append(0.0)
            highlight_idx = len(accs) - 1
            plot_ablation_bar_chart(
                names=df['Pretraining Method'].tolist(), values=accs,
                title='SSL Pretraining Ablation — Test Accuracy',
                ylabel='Accuracy (%)',
                output_path=os.path.join('results/figures', 'ablation_ssl_bar.png'),
                highlight_idx=highlight_idx
            )
            logger.info("Generated SSL ablation bar chart.")

    train_csv = os.path.join('results/tables', 'ablation_training.csv')
    if os.path.exists(train_csv):
        import pandas as pd
        df = pd.read_csv(train_csv)
        if len(df) > 0:
            accs = []
            for val in df['Test Acc (%)']:
                try:
                    accs.append(float(val))
                except (ValueError, TypeError):
                    accs.append(0.0)
            highlight_idx = len(accs) - 1
            plot_ablation_bar_chart(
                names=df['Strategy'].tolist(), values=accs,
                title='Training Strategy Ablation — Test Accuracy',
                ylabel='Accuracy (%)',
                output_path=os.path.join('results/figures', 'ablation_training_bar.png'),
                highlight_idx=highlight_idx
            )
            logger.info("Generated training strategy ablation bar chart.")

    # ===================================================================
    # 6. Load external validation predictions (if available)
    # ===================================================================
    ext_pred_path = os.path.join('results', 'external_predictions.npz')
    if os.path.exists(ext_pred_path):
        logger.info("Generating external validation table...")
        ext_data = np.load(ext_pred_path, allow_pickle=True)
        ext_true = ext_data['y_true']
        ext_pred = ext_data['y_pred']
        ext_prob = ext_data['y_prob'] if 'y_prob' in ext_data else None
        ext_ablation = str(ext_data['ablation']) if 'ablation' in ext_data else 'proposed'
        ext_metrics = compute_all_metrics(ext_true, ext_pred, y_prob=ext_prob, num_classes=num_classes)
        ext_results = {ext_ablation: ext_metrics}
        generate_external_validation_table(ext_results)

        # External validation confusion matrix
        plot_cm_v2(
            ext_true, ext_pred, class_names=CLASS_NAMES, normalize=True,
            title=f'External Validation (Messidor-2) — {ext_ablation}',
            output_path=os.path.join('results/figures', 'confusion_matrix_external.png')
        )
        logger.info("Generated external validation table and confusion matrix.")
    else:
        logger.info("No external validation predictions found. Skipping external table.")

    # ===================================================================
    # 7. Generate SSL ablation table (Table 2)
    #    Tries to load saved SSL ablation results; falls back to template
    # ===================================================================
    logger.info("Generating SSL pretraining ablation table (Table 2)...")
    ssl_ablation_path = os.path.join('results', 'ssl_ablation_results.npz')
    if os.path.exists(ssl_ablation_path):
        ssl_abl_data = np.load(ssl_ablation_path, allow_pickle=True)
        ssl_abl_results = {}
        for key in ssl_abl_data.files:
            d = ssl_abl_data[key].item()
            ssl_abl_results[key] = d
    else:
        # Template with placeholder values — user fills in after running experiments
        logger.info("  No saved SSL ablation results found. Generating template with placeholders.")
        logger.info("  Run SSL ablation experiments and save to results/ssl_ablation_results.npz for real values.")
        ssl_abl_results = {
            'ImageNet pretrain': {'val_acc': 0.0, 'val_qwk': 0.0, 'test_acc': 0.0, 'test_qwk': 0.0},
            'Contrastive only': {'val_acc': 0.0, 'val_qwk': 0.0, 'test_acc': 0.0, 'test_qwk': 0.0},
            'Multi-task only': {'val_acc': 0.0, 'val_qwk': 0.0, 'test_acc': 0.0, 'test_qwk': 0.0},
            'Full SSL (Proposed)': {'val_acc': 0.0, 'val_qwk': 0.0, 'test_acc': 0.0, 'test_qwk': 0.0},
        }
    generate_ssl_ablation_table(ssl_abl_results)

    # ===================================================================
    # 8. Generate training strategy ablation table (Table 3)
    # ===================================================================
    logger.info("Generating training strategy ablation table (Table 3)...")
    train_ablation_path = os.path.join('results', 'training_ablation_results.npz')
    if os.path.exists(train_ablation_path):
        train_abl_data = np.load(train_ablation_path, allow_pickle=True)
        train_abl_results = {}
        for key in train_abl_data.files:
            train_abl_results[key] = train_abl_data[key].item()
    else:
        logger.info("  No saved training ablation results found. Generating template with placeholders.")
        train_abl_results = {
            'No tricks': {'test_acc': 0.0, 'test_qwk': 0.0, 'test_f1': 0.0, 'test_auc': 0.0},
            '+EMA': {'test_acc': 0.0, 'test_qwk': 0.0, 'test_f1': 0.0, 'test_auc': 0.0},
            '+EMA+SWA': {'test_acc': 0.0, 'test_qwk': 0.0, 'test_f1': 0.0, 'test_auc': 0.0},
            '+TTA': {'test_acc': 0.0, 'test_qwk': 0.0, 'test_f1': 0.0, 'test_auc': 0.0},
            'Full (Proposed)': {'test_acc': 0.0, 'test_qwk': 0.0, 'test_f1': 0.0, 'test_auc': 0.0},
        }
    generate_training_ablation_table(train_abl_results)

    # ===================================================================
    # 9. Generate SOTA comparison table (Table 4)
    #    Hardcoded literature values for common DR methods
    # ===================================================================
    logger.info("Generating SOTA comparison table (Table 4)...")
    sota_results = [
        {'method': 'ResNet50', 'backbone': 'ResNet-50', 'year': '2019', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
        {'method': 'EfficientNet-B5', 'backbone': 'EfficientNet-B5', 'year': '2019', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
        {'method': 'SwinV2-Large (ImageNet)', 'backbone': 'SwinV2-L', 'year': '2024', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
        {'method': 'RetiNA-Net (Proposed)', 'backbone': 'SwinV2-L + SSL', 'year': '2025', 'dataset': 'APTOS-2019', 'acc': 0.0, 'qwk': 0.0, 'auc': 0.0},
    ]
    # If we have test predictions, fill in the proposed method's accuracy
    if y_true is not None:
        from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
        proposed_acc = accuracy_score(y_true, y_pred) * 100
        proposed_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        try:
            proposed_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            proposed_auc = 0.0
        sota_results[-1]['acc'] = proposed_acc
        sota_results[-1]['qwk'] = proposed_qwk
        sota_results[-1]['auc'] = proposed_auc
    generate_sota_comparison_table(sota_results)

    # ===================================================================
    # 10. Generate lesion correlation chart (Chart 5b)
    #     Reads trainLabels.csv + lesion .npz files
    # ===================================================================
    ssl_config_path = config.get('ssl_config', 'configs/config_ssl.yaml')
    if os.path.exists(ssl_config_path):
        ssl_config = load_config(ssl_config_path)
        labels_csv = ssl_config.get('ssl_train_labels_csv', '')
        lesion_label_dir = ssl_config.get('ssl_lesion_label_dir', 'datasets/EyePACS/lesion_labels')

        if labels_csv and os.path.exists(labels_csv) and os.path.exists(lesion_label_dir):
            logger.info("Generating lesion correlation chart (Chart 5b)...")
            import csv as csv_module
            dr_grades = []
            lesion_counts = []
            with open(labels_csv, 'r') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    basename = row.get('image', '')
                    grade = int(row.get('level', 0))
                    npz_path = os.path.join(lesion_label_dir, basename + '.npz')
                    if os.path.exists(npz_path):
                        data = np.load(npz_path)
                        total_count = int(data['count'].sum())
                        dr_grades.append(grade)
                        lesion_counts.append(total_count)
            if len(dr_grades) > 0:
                plot_lesion_correlation(
                    np.array(dr_grades), np.array(lesion_counts),
                    output_dir='results/figures'
                )
                logger.info(f"  Generated lesion correlation chart from {len(dr_grades)} images.")
            else:
                logger.info("  No lesion labels found. Skipping lesion correlation chart.")
        else:
            logger.info(f"  trainLabels.csv or lesion labels not found. Skipping lesion correlation chart.")
    else:
        logger.info("  No SSL config found. Skipping lesion correlation chart.")

    # ===================================================================
    # Summary
    # ===================================================================
    logger.info("=" * 60)
    logger.info("Table and chart generation complete!")
    logger.info(f"  Tables saved to:   results/tables/")
    logger.info(f"  Figures saved to:  results/figures/")
    logger.info("=" * 60)




def run_xai(config, ablation, device, logger, timestamp):
    """
    Generate XAI heatmaps.
    Produces:
      1. A 5×3 grid (5 DR grades × original/EigenCAM/GradCAM) — Chart 6 from plan
      2. A single sample heatmap (legacy, for quick inspection)
    """
    _, _, test_loader, _, _ = get_dataloaders(config, fold_idx=None)

    model = create_model(config, ablation, device)

    # Try to load fold 0 model, fall back to single model
    model_path = f'checkpoints/best_model_{ablation}_fold0.pth'
    if not os.path.exists(model_path):
        model_path = f'checkpoints/best_model_{ablation}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Collect one sample per DR grade (0-4)
    grade_samples = {}  # grade → (input_tensor, target)
    for inputs, targets in test_loader:
        for i in range(len(targets)):
            grade = int(targets[i].item())
            if grade not in grade_samples:
                grade_samples[grade] = (inputs[i:i+1], targets[i])
            if len(grade_samples) == 5:
                break
        if len(grade_samples) == 5:
            break

    if len(grade_samples) < 5:
        logger.warning(f"Only found {len(grade_samples)} DR grades in test set. Grid will have missing rows.")

    # Denormalization params
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    CLASS_NAMES = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']

    # --- Generate 5×3 XAI grid (Chart 6) ---
    logger.info("Generating 5-grade XAI grid (Chart 6)...")
    grid_images = []
    grid_eigencam = []
    grid_gradcam = []

    for grade in range(5):
        if grade not in grade_samples:
            # Use black placeholder for missing grade
            grid_images.append(np.zeros((512, 512, 3)))
            grid_eigencam.append(np.zeros((512, 512)))
            grid_gradcam.append(np.zeros((512, 512)))
            continue


        single_tensor, _ = grade_samples[grade]
        single_tensor = single_tensor.to(device)

        # Denormalize for visualization
        img_np = single_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Generate EigenCAM and GradCAM heatmaps
        try:
            # EigenCAM
            eigencam_heatmap = generate_heatmap(
                single_tensor, model, img_np,
                method='eigencam', return_heatmap=True
            )
        except Exception:
            eigencam_heatmap = np.zeros((512, 512))

        try:
            # GradCAM
            gradcam_heatmap = generate_heatmap(
                single_tensor, model, img_np,
                method='gradcam', return_heatmap=True
            )
        except Exception:
            gradcam_heatmap = np.zeros((512, 512))

        grid_images.append(img_np)
        grid_eigencam.append(eigencam_heatmap)
        grid_gradcam.append(gradcam_heatmap)

    # Save the 5×3 grid
    plot_xai_grid(
        images=grid_images,
        heatmaps_eigencam=grid_eigencam,
        heatmaps_gradcam=grid_gradcam,
        class_names=CLASS_NAMES,
        output_path=os.path.join('results', f'xai_grid_5grades_{ablation}_{timestamp}.png')
    )
    logger.info(f"5-grade XAI grid saved as results/xai_grid_5grades_{ablation}_{timestamp}.png")

    # --- Also generate single sample heatmap (legacy) ---
    if 0 in grade_samples:
        single_tensor, _ = grade_samples[0]
        single_tensor = single_tensor.to(device)
        img_np = single_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        generate_heatmap(single_tensor, model, img_np,
                         out_name=os.path.join('results', f'xai_heatmap_{ablation}_{timestamp}.png'))
        logger.info(f"Single XAI heatmap saved as results/xai_heatmap_{ablation}_{timestamp}.png")



def main():
    parser = argparse.ArgumentParser(description="RetiNA-Net: Retinal DR Classification Project")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test', 'external_validation', 'xai',
                                 'detect_lesions', 'pretrain', 'generate_results'],
                        help="Execution mode.")
    parser.add_argument('--ablation', type=str, default='proposed',
                        choices=['baseline', 'msda_only', 'hff_only', 'proposed'],
                        help="Ablation configuration for the model.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to config.yaml")
    parser.add_argument('--fold', type=int, default=None,
                        help="Train only a specific fold (0-indexed). If not specified, trains all folds.")

    args = parser.parse_args()

    config = load_config(args.config)

    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join('logs', f'{args.mode}_{args.ablation}_{timestamp}.log')

    logger = setup_logger(log_file=log_file_path)
    logger.info(f"Starting execution in mode: {args.mode}, ablation: {args.ablation}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.mode == 'train':
        use_kfold = config.get("use_kfold", True)

        if use_kfold:
            if args.fold is not None:
                # Train single fold
                logger.info(f"Training single fold: {args.fold}")
                train_losses, val_losses, train_accs, val_accs = train_single_fold(
                    config, args.ablation, device, logger, fold_idx=args.fold
                )
                plot_training_curves(
                    train_losses, val_losses, train_accs, val_accs,
                    filename=os.path.join('results', f'training_curves_{args.ablation}_fold{args.fold}_{timestamp}.png')
                )
            else:
                # Train all folds
                train_kfold(config, args.ablation, device, logger, timestamp)
        else:
            # Standard train/val split
            train_losses, val_losses, train_accs, val_accs = train_single_fold(
                config, args.ablation, device, logger, fold_idx=None
            )
            plot_training_curves(
                train_losses, val_losses, train_accs, val_accs,
                filename=os.path.join('results', f'training_curves_{args.ablation}_{timestamp}.png')
            )

        logger.info("Training complete.")

    elif args.mode == 'test':
        run_test(config, args.ablation, device, logger, timestamp)

    elif args.mode == 'external_validation':
        run_external_validation(config, args.ablation, device, logger, timestamp)

    elif args.mode == 'xai':
        run_xai(config, args.ablation, device, logger, timestamp)

    elif args.mode == 'detect_lesions':
        run_detect_lesions(config, logger, timestamp)

    elif args.mode == 'pretrain':
        run_pretrain(config, device, logger, timestamp)

    elif args.mode == 'generate_results':
        run_generate_results(config, logger, timestamp)


if __name__ == '__main__':
    main()

