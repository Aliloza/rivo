"""Data package for the fish age prediction benchmark."""

from .dataset import FishAgeDataset
from .augmentation import get_train_transforms, get_eval_transforms
from .splits import get_kfold_splits

__all__ = [
    "FishAgeDataset",
    "get_train_transforms",
    "get_eval_transforms",
    "get_kfold_splits",
]
