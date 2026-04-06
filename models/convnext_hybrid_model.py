"""ConvNeXt hybrid model: image encoder + metadata encoder with fusion head."""

import timm
import torch
import torch.nn as nn

from .base_model import BaseModel


class MetadataEncoder(nn.Module):
    """Metadata MLP encoder.

    Architecture: Linear → BatchNorm → ReLU → Dropout → Linear

    Args:
        input_dim (int): Number of input metadata features.
        hidden_dim (int): Hidden layer size.
        dropout (float): Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvNeXtHybridModel(BaseModel):
    """ConvNeXt-Small backbone fused with a metadata encoder.

    If ``metadata_dim`` is 0 (no metadata), the model falls back to
    image-only mode.

    Args:
        task_type (str): ``"regression"`` or ``"classification"``.
        num_classes (int): Number of output classes.
        pretrained (bool): Load ImageNet-pretrained weights.
        dropout (float): Dropout probability.
        metadata_dim (int): Dimensionality of the metadata feature vector.
        metadata_hidden_dim (int): Hidden dim for the metadata encoder.
        fusion_hidden_dim (int): Hidden dim for the fusion prediction head.
    """

    def __init__(
        self,
        task_type: str = "regression",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.3,
        metadata_dim: int = 0,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        self.task_type = task_type
        self.use_metadata = metadata_dim > 0

        # ── Image encoder ────────────────────────────────────────────────────
        self.image_encoder = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            num_classes=0,  # removes classifier
        )
        image_features_dim = self.image_encoder.num_features

        # ── Metadata encoder ─────────────────────────────────────────────────
        if self.use_metadata:
            self.meta_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                hidden_dim=metadata_hidden_dim,
                dropout=dropout,
            )
            fusion_in_dim = image_features_dim + metadata_hidden_dim
        else:
            self.meta_encoder = None
            fusion_in_dim = image_features_dim

        # ── Fusion / prediction head ──────────────────────────────────────────
        out_dim = 1 if task_type == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, out_dim),
        )

    def forward(self, images: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        image_features = self.image_encoder(images)

        if self.use_metadata and metadata is not None and metadata.shape[1] > 0:
            meta_features = self.meta_encoder(metadata)
            combined = torch.cat([image_features, meta_features], dim=1)
        else:
            combined = image_features

        predictions = self.head(combined)
        if self.task_type == "regression":
            predictions = predictions.squeeze(1)
        return predictions
