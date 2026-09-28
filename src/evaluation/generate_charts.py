"""
Auto-generate all charts for the paper (Phase 8 of plan.md).

Generates 7 chart types:
  1. Training curves (loss, accuracy, QWK, LR schedule)
  2. Confusion matrices (raw + normalized)
  3. ROC curves (multi-class OVR)
  4. Ablation bar charts (architecture, SSL, training)
  5. SSL pretraining curves (contrastive, multi-task, correlation)
  6. XAI heatmap grids
  7. Per-class radar chart

Author: RetiNA-Net Project
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Try to use a nice font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12


CLASS_NAMES = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']


def plot_training_curves(history, output_dir='results/figures', fold_label='Proposed'):
    """
    Chart 1: Training curves (loss, accuracy, QWK, LR schedule).

    Args:
        history: dict with keys: train_loss, val_loss, train_acc, val_acc, val_qwk, lr
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], label='Train Loss', color=COLORS[0], linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Val Loss', color=COLORS[3], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training & Validation Loss — {fold_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'train_val_loss_curve.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'train_acc' in history:
        ax.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Acc', color=COLORS[0], linewidth=2)
    if 'val_acc' in history:
        ax.plot(epochs, [a * 100 for a in history['val_acc']], label='Val Acc', color=COLORS[1], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Training & Validation Accuracy — {fold_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'train_val_acc_curve.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # QWK curve
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'val_qwk' in history:
        ax.plot(epochs, history['val_qwk'], label='Val QWK', color=COLORS[2], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Quadratic Weighted Kappa')
    ax.set_title(f'Validation QWK — {fold_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'train_val_qwk_curve.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # LR schedule
    if 'lr' in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history['lr'], color=COLORS[4], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, 'lr_schedule.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False,
                          title='Confusion Matrix', output_path='results/figures/cm.png'):
    """
    Chart 2: Confusion matrix (raw or normalized).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: If True, show percentages
        title: Plot title
        output_path: Path to save figure
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = '.1%'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{cm[i, j]:.1%}'
            else:
                text = f'{cm[i, j]}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_roc_curves(y_true, y_prob, num_classes=5, class_names=None,
                    output_path='results/figures/roc_multiclass.png'):
    """
    Chart 3: Multi-class ROC curves (One-vs-Rest).

    Args:
        y_true: Ground truth labels (N,)
        y_prob: Predicted probabilities (N, num_classes)
        num_classes: Number of classes
        class_names: List of class names
        output_path: Path to save figure
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], linewidth=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi-class ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pr_curves(y_true, y_prob, num_classes=5, class_names=None,
                   output_path='results/figures/pr_curve.png'):
    """Chart 3b: Precision-Recall curves."""
    if class_names is None:
        class_names = CLASS_NAMES

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, color=COLORS[i], linewidth=2,
                label=f'{class_names[i]}')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ablation_bar_chart(names, values, title, ylabel, output_path,
                             colors=None, highlight_idx=None):
    """
    Chart 4: Generic ablation bar chart.

    Args:
        names: List of method names
        values: List of metric values
        title: Chart title
        ylabel: Y-axis label
        output_path: Path to save
        colors: List of bar colors
        highlight_idx: Index of bar to highlight (proposed method)
    """
    if colors is None:
        colors = ['#90CAF9'] * len(names)
        if highlight_idx is not None:
            colors[highlight_idx] = '#1565C0'

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='white', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ssl_pretraining_curves(loss_history, output_dir='results/figures'):
    """
    Chart 5: SSL pretraining loss curves.

    Args:
        loss_history: dict with keys: contrastive, lesion_type, lesion_count, reconstruction, total
                      Each maps to list of per-epoch values
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Contrastive loss
    if 'contrastive' in loss_history:
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(loss_history['contrastive']) + 1)
        ax.plot(epochs, loss_history['contrastive'], color=COLORS[0], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('InfoNCE Loss')
        ax.set_title('SSL Contrastive Loss')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, 'ssl_contrastive_loss.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    # Multi-task losses combined
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color, label in [
        ('lesion_type', COLORS[1], 'Lesion Type (BCE)'),
        ('lesion_count', COLORS[2], 'Lesion Count (MSE)'),
        ('reconstruction', COLORS[3], 'Reconstruction (L1)'),
        ('total', COLORS[4], 'Total Loss')
    ]:
        if key in loss_history:
            epochs = range(1, len(loss_history[key]) + 1)
            ax.plot(epochs, loss_history[key], label=label, color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SSL Multi-Task Pretraining Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'ssl_multitask_loss.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lesion_correlation(dr_grades, lesion_counts, output_dir='results/figures'):
    """
    Chart 5b: Lesion detection vs DR grade correlation.

    Args:
        dr_grades: array of DR grades (0-4)
        lesion_counts: array of total lesion counts (same length as dr_grades)
        output_dir: Directory to save
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Box plot: lesion count by DR grade
    grades = sorted(set(dr_grades))
    data_by_grade = []
    for g in grades:
        data_by_grade.append([c for grade, c in zip(dr_grades, lesion_counts) if grade == g])

    bp = ax.boxplot(data_by_grade, labels=[CLASS_NAMES[g] if g < len(CLASS_NAMES) else f'Grade {g}' for g in grades],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS[:len(grades)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('DR Grade')
    ax.set_ylabel('Total Lesion Count')
    ax.set_title('Lesion Detection Count vs DR Grade')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    path = os.path.join(output_dir, 'ssl_lesion_correlation.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_radar_chart(per_class_metrics, metric_names=None, class_names=None,
                      output_path='results/figures/radar_per_class.png'):
    """
    Chart 7: Per-class performance radar chart.

    Args:
        per_class_metrics: dict mapping class_name → dict of metric → value
        metric_names: List of metric names to show
        class_names: List of class names
        output_path: Path to save
    """
    if metric_names is None:
        metric_names = ['Sensitivity', 'Specificity', 'F1', 'Precision']
    if class_names is None:
        class_names = CLASS_NAMES

    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, cls in enumerate(class_names):
        if cls in per_class_metrics:
            values = [per_class_metrics[cls].get(m, 0) * 100 for m in metric_names]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[i], label=cls, markersize=5)
            ax.fill(angles, values, alpha=0.1, color=COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.set_title('Per-Class Performance Radar Chart', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_xai_grid(images, heatmaps_eigencam, heatmaps_gradcam, class_names=None,
                   output_path='results/figures/xai_grid_5grades.png'):
    """
    Chart 6: XAI heatmap grid (5 DR grades × 3 columns: original, EigenCAM, GradCAM).

    Args:
        images: list of 5 original images (H, W, 3)
        heatmaps_eigencam: list of 5 EigenCAM heatmaps (H, W)
        heatmaps_gradcam: list of 5 GradCAM heatmaps (H, W)
        class_names: List of 5 class names
        output_path: Path to save
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    col_titles = ['Original', 'EigenCAM', 'GradCAM']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14, fontweight='bold')

    for i in range(5):
        # Original
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_ylabel(class_names[i], fontsize=12, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # EigenCAM
        axes[i, 1].imshow(images[i])
        axes[i, 1].imshow(heatmaps_eigencam[i], cmap='jet', alpha=0.5)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # GradCAM
        axes[i, 2].imshow(images[i])
        axes[i, 2].imshow(heatmaps_gradcam[i], cmap='jet', alpha=0.5)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

    fig.suptitle('XAI Heatmaps Across DR Grades', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
