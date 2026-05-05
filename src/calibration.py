"""
Temperature Scaling for probability calibration.
Learns a single scalar parameter T on the validation set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict


class TemperatureScaling(nn.Module):
    """Post-hoc calibration via temperature scaling.
    T > 1 softens probabilities, T < 1 sharpens them.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, model, val_dataloader: DataLoader, device: torch.device,
            lr: float = 0.01, max_iter: int = 50) -> float:
        """Learn optimal T on validation set using NLL loss."""
        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Collecting logits", leave=False):
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                all_logits.append(outputs.logits.cpu())
                all_labels.append(batch['labels'])

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)
        self.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"[OK] Temperature Scaling fitted: T = {self.temperature.item():.4f}")
        return self.temperature.item()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(logits), dim=-1)


def compute_ece(true_labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error. Lower is better."""
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true_labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(correct[mask].mean() - confidences[mask].mean())
    return ece / len(true_labels)


def compute_brier_score(true_labels: np.ndarray, probs: np.ndarray) -> float:
    """Brier Score. Lower is better."""
    true_onehot = np.eye(probs.shape[1])[true_labels]
    return np.mean(np.sum((probs - true_onehot) ** 2, axis=1))


def calibration_report(true_labels, probs_before, probs_after) -> Dict:
    """Compare calibration metrics before/after temperature scaling."""
    before = {'ECE': compute_ece(true_labels, probs_before),
              'Brier': compute_brier_score(true_labels, probs_before)}
    after = {'ECE': compute_ece(true_labels, probs_after),
             'Brier': compute_brier_score(true_labels, probs_after)}

    print("\n" + "=" * 55)
    print("  CALIBRATION REPORT")
    print("=" * 55)
    print(f"  {'Metric':<15s} {'Before':>10s} {'After':>10s} {'Δ':>10s}")
    print("─" * 55)
    for m in ['ECE', 'Brier']:
        d = after[m] - before[m]
        print(f"  {m:<15s} {before[m]:>10.4f} {after[m]:>10.4f} {'↓' if d<0 else '↑'} {abs(d):.4f}")
    print("=" * 55)
    return {'before': before, 'after': after}
