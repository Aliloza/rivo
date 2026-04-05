"""Reproducibility helpers: seed setting and deterministic mode."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_determinism(deterministic: bool = True) -> None:
    """Enable or disable deterministic CuDNN behaviour.

    When enabled, training becomes fully reproducible at the cost of a
    minor speed reduction.

    Args:
        deterministic (bool): If ``True``, enable deterministic mode.
    """
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
