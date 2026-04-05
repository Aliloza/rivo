"""K-fold cross-validation split utilities."""

import os
import pandas as pd
from sklearn.model_selection import KFold


def load_metadata(metadata_csv: str) -> pd.DataFrame:
    """Load the metadata CSV with clear error messages.

    Args:
        metadata_csv (str): Path to the CSV file.

    Returns:
        pd.DataFrame
    """
    if not os.path.isfile(metadata_csv):
        raise FileNotFoundError(
            f"Metadata CSV not found: '{metadata_csv}'. "
            "Please update 'data.metadata_csv' in your config."
        )
    df = pd.read_csv(metadata_csv)
    return df


def get_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 3,
    seed: int = 42,
) -> list:
    """Generate stratified K-fold index splits.

    Args:
        df (pd.DataFrame): Full metadata DataFrame.
        n_folds (int): Number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Each element is
        ``(train_indices, test_indices)`` for one fold.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(df))
    return splits
