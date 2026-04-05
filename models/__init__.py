"""Models package for the fish age prediction benchmark."""

from .inception_model import InceptionModel
from .efficientnet_model import EfficientNetModel
from .convnext_hybrid_model import ConvNeXtHybridModel


def build_model(cfg: dict):
    """Factory function – instantiate a model from a config dict.

    Args:
        cfg (dict): Full config dictionary (parsed from YAML).

    Returns:
        torch.nn.Module
    """
    model_name = cfg["model"]["name"].lower()
    task_type = cfg["task"]["type"]
    num_classes = cfg["task"].get("num_classes", 1)
    pretrained = cfg["model"].get("pretrained", True)
    dropout = cfg["model"].get("dropout", 0.3)

    if model_name == "inception":
        return InceptionModel(
            task_type=task_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif model_name == "efficientnet":
        return EfficientNetModel(
            task_type=task_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif model_name == "convnext_hybrid":
        metadata_dim = len(cfg["data"].get("metadata_cols", []))
        metadata_hidden_dim = cfg["model"].get("metadata_hidden_dim", 64)
        fusion_hidden_dim = cfg["model"].get("fusion_hidden_dim", 256)
        return ConvNeXtHybridModel(
            task_type=task_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=metadata_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            "Choose from: inception, efficientnet, convnext_hybrid."
        )


__all__ = [
    "InceptionModel",
    "EfficientNetModel",
    "ConvNeXtHybridModel",
    "build_model",
]
