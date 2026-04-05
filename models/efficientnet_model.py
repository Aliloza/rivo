"""EfficientNet-B4 image-only baseline model."""

import timm
import torch.nn as nn

from .base_model import BaseModel


class EfficientNetModel(BaseModel):
    """EfficientNet-B4 backbone with a custom regression / classification head.

    Args:
        task_type (str): ``"regression"`` or ``"classification"``.
        num_classes (int): Number of output classes (ignored for regression).
        pretrained (bool): Load ImageNet-pretrained weights.
        dropout (float): Dropout probability in the head.
    """

    def __init__(
        self,
        task_type: str = "regression",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.task_type = task_type

        # Build backbone without a classifier head (num_classes=0 → feature vector)
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,  # removes the classifier
        )
        in_features = self.backbone.num_features

        out_dim = 1 if task_type == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_dim),
        )

    def forward(self, images, metadata=None):
        features = self.backbone(images)
        predictions = self.head(features)
        if self.task_type == "regression":
            predictions = predictions.squeeze(1)
        return predictions
