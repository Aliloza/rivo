"""InceptionV3 image-only baseline model."""

import torch.nn as nn
import torchvision.models as tv_models

from .base_model import BaseModel


class InceptionModel(BaseModel):
    """InceptionV3 backbone with a custom regression / classification head.

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
        dropout: float = 0.3,
    ):
        super().__init__()
        self.task_type = task_type

        weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.inception_v3(weights=weights, aux_logits=True)
        in_features = backbone.fc.in_features

        # Replace the classification head
        backbone.fc = nn.Identity()
        # Also replace aux head to avoid output shape mismatches
        backbone.AuxLogits.fc = nn.Identity()
        self.backbone = backbone

        out_dim = 1 if task_type == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_dim),
        )

    def forward(self, images, metadata=None):
        # InceptionV3 returns (logits, aux_logits) in train mode
        out = self.backbone(images)
        if isinstance(out, tuple):
            features = out[0]
        else:
            features = out

        predictions = self.head(features)
        if self.task_type == "regression":
            predictions = predictions.squeeze(1)
        return predictions
