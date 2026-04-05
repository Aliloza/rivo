"""CSV and TensorBoard logging helpers."""

import csv
import os
from pathlib import Path


def setup_csv_logger(log_path: str, fieldnames: list) -> tuple:
    """Create and return a CSV DictWriter with a header row.

    The caller is responsible for closing the returned file handle when done.
    Typical usage::

        writer, f = setup_csv_logger(path, cols)
        try:
            ...
            log_metrics_csv(writer, row)
        finally:
            f.close()

    Args:
        log_path (str): Path to the output CSV file.
        fieldnames (list[str]): Column names.

    Returns:
        tuple[csv.DictWriter, IO]: The writer and the open file handle.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", newline="")
    try:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    except Exception:
        f.close()
        raise
    return writer, f


def log_metrics_csv(writer: csv.DictWriter, row: dict) -> None:
    """Write a single metrics row to a CSV logger.

    Args:
        writer (csv.DictWriter): Open DictWriter.
        row (dict): Mapping of column name → value.
    """
    writer.writerow(row)


def setup_tensorboard(log_dir: str, run_name: str):
    """Initialise a TensorBoard SummaryWriter.

    Falls back gracefully if TensorBoard is not installed.

    Args:
        log_dir (str): Root TensorBoard log directory.
        run_name (str): Sub-directory name for this run.

    Returns:
        torch.utils.tensorboard.SummaryWriter | None
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return SummaryWriter(log_dir=run_dir)
    except ImportError:
        print("[WARNING] TensorBoard not available. Skipping TB logging.")
        return None
