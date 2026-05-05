"""
Model definitions: RoBERTa loading, saving, MC Dropout inference.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from typing import Dict, Optional, Tuple


# ==================== Model Loading ====================

def load_model_and_tokenizer(
    model_name: str = "roberta-base",
    num_labels: int = 2,
    dropout_rate: float = 0.1
):
    """Load pretrained RoBERTa model and tokenizer."""

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model loaded: {model_name}")
    print(f"  Parameters: {total_params:,} (trainable: {trainable:,})")

    return model, tokenizer


# ==================== Model Save/Load ====================

def save_model(model, tokenizer, save_dir: str):
    """Save model and tokenizer to directory."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[OK] Model saved to: {save_dir}")


def load_trained_model(model_dir: str):
    """Load a previously trained model."""
    print(f"Loading trained model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    print(f"[OK] Model loaded successfully")
    return model, tokenizer


# ==================== Layer Freezing ====================

def freeze_layers(model, num_layers_to_freeze: int = 0):
    """Freeze bottom N transformer layers for transfer learning."""
    if num_layers_to_freeze == 0:
        return model

    # Identify encoder
    if hasattr(model, 'roberta'):
        encoder = model.roberta.encoder
        embeddings = model.roberta.embeddings
    elif hasattr(model, 'bert'):
        encoder = model.bert.encoder
        embeddings = model.bert.embeddings
    else:
        print("[WARNING] Model architecture not recognized for layer freezing")
        return model

    # Freeze embeddings
    for param in embeddings.parameters():
        param.requires_grad = False

    # Freeze specified layers
    for layer in encoder.layer[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Froze embeddings + {num_layers_to_freeze} layers")
    print(f"  Trainable parameters: {trainable:,}")

    return model


# ==================== Model Summary ====================

def get_model_summary(model):
    """Print model parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Non-trainable:        {total - trainable:,}")
    print("=" * 70 + "\n")


# ==================== MC Dropout Inference ====================

def enable_mc_dropout(model):
    """Enable dropout layers during inference for Monte Carlo estimation.
    
    Keeps model in eval mode but activates dropout layers specifically.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_runs: int = 10,
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    """Run Monte Carlo Dropout inference.
    
    Performs N forward passes with dropout enabled to estimate:
    - Mean prediction probabilities
    - Prediction uncertainty (variance)
    
    Args:
        model: RoBERTa model
        input_ids: Tokenized input IDs
        attention_mask: Attention mask
        n_runs: Number of MC forward passes
        device: Compute device
        
    Returns:
        Dict with keys:
        - 'mean_probs': Mean probabilities across runs [batch, num_labels]
        - 'std_probs': Std deviation of probabilities [batch, num_labels]
        - 'predictions': Argmax of mean probs [batch]
        - 'uncertainty': Mean std across labels (scalar per sample) [batch]
        - 'all_probs': All probabilities from each run [n_runs, batch, num_labels]
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Enable MC Dropout
    enable_mc_dropout(model)

    all_probs = []

    for _ in range(n_runs):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.array(all_probs)  # [n_runs, batch, num_labels]

    mean_probs = all_probs.mean(axis=0)     # [batch, num_labels]
    std_probs = all_probs.std(axis=0)       # [batch, num_labels]
    predictions = mean_probs.argmax(axis=1)  # [batch]
    uncertainty = std_probs.mean(axis=1)     # [batch]

    # Restore normal eval mode
    model.eval()

    return {
        'mean_probs': mean_probs,
        'std_probs': std_probs,
        'predictions': predictions,
        'uncertainty': uncertainty,
        'all_probs': all_probs
    }


def mc_dropout_predict_single(
    model,
    tokenizer,
    text: str,
    n_runs: int = 10,
    max_length: int = 256,
    device: torch.device = None
) -> Dict[str, any]:
    """MC Dropout prediction for a single text input.
    
    Returns:
        Dict with prediction, confidence, uncertainty, and probabilities.
    """
    if device is None:
        device = next(model.parameters()).device

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    result = mc_dropout_predict(
        model=model,
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        n_runs=n_runs,
        device=device
    )

    pred_label = int(result['predictions'][0])
    from .config import LABEL_MAP_INV

    return {
        'prediction': LABEL_MAP_INV[pred_label],
        'confidence': float(result['mean_probs'][0][pred_label]),
        'uncertainty': float(result['uncertainty'][0]),
        'probabilities': {
            'real': float(result['mean_probs'][0][0]),
            'fake': float(result['mean_probs'][0][1])
        }
    }
