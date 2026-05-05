"""
Utility functions: seeding, text cleaning, metrics, early stopping, device management.
"""

import re
import os
import json
import random
import numpy as np
import torch
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ==================== Reproducibility ====================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== Text Cleaning ====================

def clean_text(text: str) -> str:
    """Clean and normalize text for model input."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove mentions & hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # Remove special characters (keep letters, numbers, basic punctuation)
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==================== Metrics ====================

def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, F1."""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    print(f"\n{prefix} Metrics:")
    for key, value in metrics.items():
        print(f"  {key.capitalize():12s}: {value:.4f}")


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] Metrics saved to: {filepath}")


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== Time Formatting ====================

def format_time(elapsed: float) -> str:
    """Format elapsed seconds to human-readable string."""
    elapsed_rounded = int(round(elapsed))
    return f"{elapsed_rounded // 60}m {elapsed_rounded % 60}s"


# ==================== Early Stopping ====================

class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 3, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


# ==================== Device Management ====================

def get_device() -> torch.device:
    """Get available compute device (CUDA/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            print(f"  VRAM: {total_mem / 1e9:.2f} GB")
        except Exception:
            print("  VRAM: (unable to query)")
    else:
        device = torch.device("cpu")
        print("[WARNING] Using CPU (training will be slow)")
    return device


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU Memory — Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
