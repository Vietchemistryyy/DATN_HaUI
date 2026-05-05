"""
Dataset classes and data loading utilities for RoBERTa training.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import pandas as pd


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection with RoBERTa tokenization."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file. Expects 'text' and 'label' columns."""
    df = pd.read_csv(filepath)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    # Drop NaN texts
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    print(f"[OK] Loaded {len(df):,} records from {filepath}")
    return df


def prepare_datasets(
    train_file: str,
    val_file: str,
    test_file: str,
    tokenizer,
    max_length: int = 256
) -> Tuple[FakeNewsDataset, FakeNewsDataset, FakeNewsDataset]:
    """Prepare train, validation, and test datasets."""

    train_df = load_data(train_file)
    val_df = load_data(val_file)
    test_df = load_data(test_file)

    train_dataset = FakeNewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = FakeNewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = FakeNewsDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    print(f"\n[OK] Datasets prepared:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")

    return train_dataset, val_dataset, test_dataset
