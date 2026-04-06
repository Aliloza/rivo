"""Abstract base class for all fish-age prediction models."""

import torch.nn as nn


class BaseModel(nn.Module):
    """Base class that all benchmark models inherit from.

    Enforces a common interface and provides shared utilities.
    """

    def forward(self, images, metadata=None):
        """Forward pass.

        Args:
            images (torch.Tensor): Batch of images ``(B, C, H, W)``.
            metadata (torch.Tensor | None): Optional metadata features
                ``(B, D)``. Ignored by image-only models.

        Returns:
            torch.Tensor: Predictions ``(B,)`` for regression or
            ``(B, num_classes)`` for classification.
        """
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
