"""Metric computation for regression and classification tasks."""

import numpy as np


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    task_type: str = "regression",
) -> dict:
    """Compute benchmark metrics.

    For **regression**:
    - MAE  (mean absolute error)
    - RMSE (root mean squared error)
    - Exact accuracy  (prediction rounds to exact integer age)
    - ±1 accuracy     (prediction within 1 year of true age)

    For **classification**:
    - Top-1 accuracy
    - MAE (treating class indices as ordinal values)

    Args:
        preds (np.ndarray): Model predictions, shape ``(N,)``.
        targets (np.ndarray): Ground-truth labels, shape ``(N,)``.
        task_type (str): ``"regression"`` or ``"classification"``.

    Returns:
        dict: Metric name → float value.
    """
    preds = np.asarray(preds, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()

    metrics: dict = {}

    if task_type == "regression":
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        rounded = np.round(preds)
        exact_acc = float(np.mean(rounded == targets))
        pm1_acc = float(np.mean(np.abs(rounded - targets) <= 1))

        metrics["mae"] = mae
        metrics["rmse"] = rmse
        metrics["exact_acc"] = exact_acc
        metrics["pm1_acc"] = pm1_acc

    else:  # classification
        pred_classes = np.round(preds).astype(int)
        target_classes = targets.astype(int)
        acc = float(np.mean(pred_classes == target_classes))
        mae = float(np.mean(np.abs(preds - targets)))

        metrics["accuracy"] = acc
        metrics["mae"] = mae

    return metrics
