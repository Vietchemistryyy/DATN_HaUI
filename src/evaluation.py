"""
Evaluation utilities: metrics, reports, visualizations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Tuple


# ==================== Model Evaluation ====================

def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate model and return predictions, labels, probabilities, and metrics."""

    model.eval()
    predictions = []
    true_labels = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': total_loss / len(dataloader)
    }

    return predictions, true_labels, all_probs, metrics


# ==================== Reports ====================

def print_evaluation_report(predictions: np.ndarray, true_labels: np.ndarray):
    """Print detailed evaluation report."""

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)

    print("\nClassification Report:")
    print(classification_report(
        true_labels, predictions,
        target_names=['Real', 'Fake'],
        digits=4
    ))

    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Real   Fake")
    print(f"Actual Real   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Fake   {cm[1][0]:5d}  {cm[1][1]:5d}")
    print("=" * 70 + "\n")


# ==================== Visualization ====================

def plot_confusion_matrix(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    save_path: str = None,
    title: str = "Confusion Matrix"
):
    """Plot confusion matrix heatmap."""

    cm = confusion_matrix(true_labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()


def plot_training_history(history: Dict, save_path: str = None):
    """Plot training history (loss, accuracy, F1)."""

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 's-', label='Validation')
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'o-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 's-', label='Validation')
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    if 'train_f1' in history:
        axes[2].plot(epochs, history['train_f1'], 'o-', label='Train')
        axes[2].plot(epochs, history['val_f1'], 's-', label='Validation')
        axes[2].set_title('F1 Score', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()


def plot_roc_curve(
    true_labels: np.ndarray,
    probs: np.ndarray,
    save_path: str = None,
    title: str = "ROC Curve"
):
    """Plot ROC curve with AUC score."""

    fpr, tpr, _ = roc_curve(true_labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()

    return roc_auc


def plot_reliability_diagram(
    true_labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
    save_path: str = None,
    title: str = "Reliability Diagram"
):
    """Plot reliability diagram for calibration analysis.
    
    Compares predicted confidence vs actual accuracy in each bin.
    """
    # Get predicted class probabilities
    pred_probs = probs.max(axis=1)
    pred_labels = probs.argmax(axis=1)
    correct = (pred_labels == true_labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (pred_probs > bin_edges[i]) & (pred_probs <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(pred_probs[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax1.bar(bin_centers, bin_accs, width=1.0/n_bins, alpha=0.7,
            edgecolor='black', label='Model')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(title, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.hist(pred_probs, bins=n_bins, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()


def plot_uncertainty_distribution(
    uncertainties: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str = None
):
    """Plot uncertainty distribution: correct vs incorrect predictions."""

    correct_mask = predictions == true_labels
    correct_unc = uncertainties[correct_mask]
    incorrect_unc = uncertainties[~correct_mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(correct_unc, bins=50, alpha=0.6, label=f'Correct (n={len(correct_unc)})',
            color='green', density=True)
    ax.hist(incorrect_unc, bins=50, alpha=0.6, label=f'Incorrect (n={len(incorrect_unc)})',
            color='red', density=True)
    ax.set_xlabel('Uncertainty (MC Dropout Std)')
    ax.set_ylabel('Density')
    ax.set_title('Uncertainty Distribution: Correct vs Incorrect Predictions',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")
    plt.show()


def compare_models_table(results: Dict[str, Dict[str, float]]):
    """Print a comparison table of model results.
    
    Args:
        results: Dict like {'LR+TF-IDF': {'accuracy': 0.85, 'f1': 0.84}, ...}
    """
    print("\n" + "=" * 70)
    print(f"  {'Model':<30s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("─" * 70)

    for name, metrics in results.items():
        print(f"  {name:<30s} "
              f"{metrics.get('accuracy', 0):>10.4f} "
              f"{metrics.get('precision', 0):>10.4f} "
              f"{metrics.get('recall', 0):>10.4f} "
              f"{metrics.get('f1', 0):>10.4f}")

    print("=" * 70 + "\n")


# ==================== Single Prediction ====================

def predict_single(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 256
) -> Dict[str, any]:
    """Standard prediction for a single text (no MC Dropout)."""

    model.eval()

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    from .config import LABEL_MAP_INV

    return {
        'prediction': LABEL_MAP_INV[prediction],
        'confidence': probs[0][prediction].item(),
        'probabilities': {
            'real': probs[0][0].item(),
            'fake': probs[0][1].item()
        }
    }
