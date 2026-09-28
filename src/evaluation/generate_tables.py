"""
Auto-generate all tables for the paper (Phase 8 of plan.md).

Generates 6 tables as CSV files:
  1. Architecture ablation table
  2. SSL pretraining ablation table
  3. Training strategy ablation table
  4. SOTA comparison table
  5. Per-class metrics table
  6. External validation results table

All metrics reported to 2 decimal places.

Author: RetiNA-Net Project
"""

import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, confusion_matrix,
                             classification_report, roc_auc_score)


def fmt(x, decimals=2):
    """Format a number to specified decimal places, handling None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '-'
    return f"{x:.{decimals}f}"


def compute_all_metrics(y_true, y_pred, y_prob=None, num_classes=5):
    """
    Compute all metrics for a set of predictions.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, num_classes) for AUC, or None
        num_classes: Number of classes

    Returns:
        dict of metrics
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['qwk'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred, weights=None)

    if y_prob is not None:
        try:
            metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            metrics['auc_macro'] = 0.0
    else:
        metrics['auc_macro'] = 0.0

    # Per-class sensitivity and specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        metrics[f'sensitivity_class{i}'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'specificity_class{i}'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def generate_architecture_ablation_table(results, output_dir='results/tables'):
    """
    Table 1: Architecture ablation table.

    Args:
        results: dict mapping ablation name → metrics dict
        output_dir: Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, m in results.items():
        rows.append({
            'Model': name,
            'Accuracy (%)': fmt(m.get('accuracy', 0) * 100),
            'Precision (%)': fmt(m.get('precision_macro', 0) * 100),
            'Recall (%)': fmt(m.get('recall_macro', 0) * 100),
            'F1 (macro) (%)': fmt(m.get('f1_macro', 0) * 100),
            'F1 (weighted) (%)': fmt(m.get('f1_weighted', 0) * 100),
            'QWK': fmt(m.get('qwk', 0)),
            'Kappa': fmt(m.get('kappa', 0)),
            'AUC': fmt(m.get('auc_macro', 0)),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'ablation_architecture.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def generate_ssl_ablation_table(results, output_dir='results/tables'):
    """
    Table 2: SSL pretraining ablation table.

    Args:
        results: dict mapping SSL method name → {'val_acc', 'val_qwk', 'test_acc', 'test_qwk'}
        output_dir: Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, m in results.items():
        rows.append({
            'Pretraining Method': name,
            'Val Acc (%)': fmt(m.get('val_acc', 0) * 100),
            'Val QWK': fmt(m.get('val_qwk', 0)),
            'Test Acc (%)': fmt(m.get('test_acc', 0) * 100),
            'Test QWK': fmt(m.get('test_qwk', 0)),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'ablation_ssl.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def generate_training_ablation_table(results, output_dir='results/tables'):
    """
    Table 3: Training strategy ablation table.

    Args:
        results: dict mapping strategy name → metrics dict
        output_dir: Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, m in results.items():
        rows.append({
            'Strategy': name,
            'Test Acc (%)': fmt(m.get('test_acc', 0) * 100),
            'Test QWK': fmt(m.get('test_qwk', 0)),
            'Test F1 (%)': fmt(m.get('test_f1', 0) * 100),
            'Test AUC': fmt(m.get('test_auc', 0)),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'ablation_training.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def generate_sota_comparison_table(results, output_dir='results/tables'):
    """
    Table 4: SOTA comparison table.

    Args:
        results: list of dicts with keys: method, backbone, year, acc, qwk, auc, dataset
        output_dir: Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            'Method': r.get('method', '-'),
            'Backbone': r.get('backbone', '-'),
            'Year': r.get('year', '-'),
            'Dataset': r.get('dataset', '-'),
            'Accuracy (%)': fmt(r.get('acc', 0)),
            'QWK': fmt(r.get('qwk', 0)),
            'AUC': fmt(r.get('auc', 0)),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'sota_comparison.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def generate_per_class_metrics_table(y_true, y_pred, y_prob=None,
                                     num_classes=5, class_names=None,
                                     output_dir='results/tables'):
    """
    Table 5: Per-class metrics table.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for per-class AUC)
        num_classes: Number of classes
        class_names: List of class names
        output_dir: Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    rows = []
    for i in range(num_classes):
        binary_true = (np.array(y_true) == i).astype(int)
        binary_pred = (np.array(y_pred) == i).astype(int)

        tp = ((binary_pred == 1) & (binary_true == 1)).sum()
        fn = ((binary_pred == 0) & (binary_true == 1)).sum()
        fp = ((binary_pred == 1) & (binary_true == 0)).sum()
        tn = ((binary_pred == 0) & (binary_true == 0)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        # Per-class AUC
        auc = 0.0
        if y_prob is not None:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(binary_true, y_prob[:, i])
            except Exception:
                auc = 0.0

        rows.append({
            'Class': class_names[i],
            'Sensitivity (%)': fmt(sensitivity * 100),
            'Specificity (%)': fmt(specificity * 100),
            'Precision (%)': fmt(precision * 100),
            'F1 (%)': fmt(f1 * 100),
            'AUC': fmt(auc),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'per_class_metrics.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def generate_external_validation_table(results, output_dir='results/tables'):
    """
    Table 6: External validation results table.

    Args:
        results: dict mapping model name → metrics dict
        output_dir: Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, m in results.items():
        rows.append({
            'Model': name,
            'Dataset': 'Messidor-2',
            'Accuracy (%)': fmt(m.get('accuracy', 0) * 100),
            'F1 (macro) (%)': fmt(m.get('f1_macro', 0) * 100),
            'QWK': fmt(m.get('qwk', 0)),
            'AUC': fmt(m.get('auc_macro', 0)),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'external_validation.csv')
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def generate_classification_report_text(y_true, y_pred, class_names=None,
                                         output_dir='results/tables'):
    """Generate sklearn classification report as text file."""
    os.makedirs(output_dir, exist_ok=True)

    if class_names is None:
        class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']

    report = classification_report(y_true, y_pred, target_names=class_names, digits=2)
    report += f"\nQuadratic Weighted Kappa: {cohen_kappa_score(y_true, y_pred, weights='quadratic'):.4f}\n"

    path = os.path.join(output_dir, 'classification_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"Saved: {path}")
    return report
