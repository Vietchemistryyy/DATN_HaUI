"""
Training loop with validation, early stopping, and history tracking.
"""

import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from typing import Dict, Any

from .utils import format_time, calculate_metrics, print_metrics, EarlyStopping


class Trainer:
    """Trainer for RoBERTa-based fake news detection model."""

    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm

        # Move model to device
        self.model.to(self.device)

        # Optimizer (use torch.optim.AdamW instead of deprecated transformers.AdamW)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        progress_bar = tqdm(self.train_dataloader, desc="Training", leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            # Track
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_dataloader)
        metrics = calculate_metrics(np.array(predictions), np.array(true_labels))
        metrics['loss'] = avg_loss

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_dataloader)
        metrics = calculate_metrics(np.array(predictions), np.array(true_labels))
        metrics['loss'] = avg_loss

        return metrics

    def train(self) -> Dict[str, Any]:
        """Full training loop with validation and early stopping."""
        print(f"\n{'=' * 70}")
        print(f"  STARTING TRAINING — {self.num_epochs} epochs")
        print(f"  Train batches: {len(self.train_dataloader)}")
        print(f"  Val batches:   {len(self.val_dataloader)}")
        print(f"  Device:        {self.device}")
        print(f"{'=' * 70}\n")

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.num_epochs):
            print(f"\n{'─' * 70}")
            print(f"  Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'─' * 70}")

            # Train
            start = time.time()
            train_metrics = self.train_epoch()
            print(f"\n  [Train] {format_time(time.time() - start)}")
            print_metrics(train_metrics, "  Train")

            # Validate
            start = time.time()
            val_metrics = self.validate()
            print(f"\n  [Val]   {format_time(time.time() - start)}")
            print_metrics(val_metrics, "  Val")

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                print(f"\n  [OK] New best val loss: {best_val_loss:.4f}")

            # Early stopping check
            if self.early_stopping(val_metrics['loss']):
                print(f"\n  [WARNING] Early stopping triggered at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
            print(f"\n  [OK] Restored best model (val_loss={best_val_loss:.4f})")

        print(f"\n{'=' * 70}")
        print(f"  TRAINING COMPLETE")
        print(f"{'=' * 70}\n")

        return self.history
