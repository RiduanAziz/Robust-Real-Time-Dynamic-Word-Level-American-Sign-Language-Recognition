from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .dataset import SignLanguageDataset


def build_dataloader(dataset: SignLanguageDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create a basic DataLoader for landmark samples."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
