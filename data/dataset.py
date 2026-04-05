"""Dataset loader for fish age prediction.

Loads segmented ROI images and optional metadata from a CSV file.
Supports regression and classification modes.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class FishAgeDataset(Dataset):
    """PyTorch Dataset for fish age prediction.

    Args:
        image_dir (str): Directory containing segmented ROI images.
        metadata_df (pd.DataFrame): DataFrame with at least image filename
            and label columns. Must already be split into the desired subset.
        image_col (str): Column name for image file names.
        label_col (str): Column name for labels.
        metadata_cols (list[str]): Columns to use as numeric metadata features.
            May be empty.
        transform: Torchvision / albumentations transform applied to images.
        task_type (str): ``"regression"`` or ``"classification"``.
        meta_scaler: Fitted sklearn scaler for metadata normalisation.
            ``None`` means no scaling is applied (e.g. during scaler fitting).
        is_train (bool): Whether this split is for training (informational).
    """

    def __init__(
        self,
        image_dir: str,
        metadata_df: pd.DataFrame,
        image_col: str = "filename",
        label_col: str = "age",
        metadata_cols: list = None,
        transform=None,
        task_type: str = "regression",
        meta_scaler=None,
        is_train: bool = False,
    ):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"Image directory not found: '{image_dir}'. "
                "Please update 'data.image_dir' in your config."
            )

        self.image_dir = image_dir
        self.df = metadata_df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.metadata_cols = metadata_cols if metadata_cols else []
        self.transform = transform
        self.task_type = task_type
        self.meta_scaler = meta_scaler
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Image ─────────────────────────────────────────────────────────────
        img_path = os.path.join(self.image_dir, str(row[self.image_col]))
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # ── Label ─────────────────────────────────────────────────────────────
        label_val = row[self.label_col]
        if self.task_type == "classification":
            label = torch.tensor(int(label_val), dtype=torch.long)
        else:
            label = torch.tensor(float(label_val), dtype=torch.float32)

        # ── Metadata ──────────────────────────────────────────────────────────
        if self.metadata_cols:
            meta_values = []
            for col in self.metadata_cols:
                if col in row.index:
                    val = row[col]
                    meta_values.append(float(val) if pd.notna(val) else 0.0)
                else:
                    meta_values.append(0.0)

            meta = np.array(meta_values, dtype=np.float32).reshape(1, -1)
            if self.meta_scaler is not None:
                meta = self.meta_scaler.transform(meta)
            meta_tensor = torch.tensor(meta.squeeze(0), dtype=torch.float32)
        else:
            meta_tensor = torch.empty(0, dtype=torch.float32)

        return image, meta_tensor, label
