"""Dataset abstractions and split utilities."""

from .dataset import SignLanguageDataset, SignSample, build_synthetic_dataset
from .dataloader import build_dataloader
from .preprocessing import pad_or_truncate_sequence, sequence_to_tensor, temporal_augmentation
from .splitting import signer_aware_split

__all__ = [
    "SignLanguageDataset",
    "SignSample",
    "build_synthetic_dataset",
    "signer_aware_split",
    "build_dataloader",
    "pad_or_truncate_sequence",
    "sequence_to_tensor",
    "temporal_augmentation",
]
