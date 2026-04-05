"""Training pipeline for fish age prediction.

Usage:
    python train.py --config configs/inception.yaml [--fold 0] [--seed 42]

The script:
- Loads a merged (base + model-specific) config
- Sets seeds and deterministic mode
- Builds train / val / test DataLoaders for the requested fold
- Trains with mixed-precision, early stopping, and LR scheduling
- Saves the best checkpoint and training CSV log
- Writes TensorBoard events
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data import FishAgeDataset, get_train_transforms, get_eval_transforms, get_kfold_splits
from data.splits import load_metadata
from models import build_model
from utils import (
    set_seed,
    configure_determinism,
    compute_metrics,
    setup_csv_logger,
    log_metrics_csv,
    setup_tensorboard,
    load_config,
    merge_configs,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[INFO] GPU not available – using CPU.")
    return dev


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"].get("weight_decay", 0.0)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "cosine").lower()
    epochs = cfg["training"]["epochs"]
    warmup = sched_cfg.get("warmup_epochs", 0)

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup),
            eta_min=sched_cfg.get("min_lr", 1e-7),
        )
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.5),
        )
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("gamma", 0.5),
            patience=5,
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    else:
        scheduler = None

    return scheduler


def build_criterion(cfg: dict) -> nn.Module:
    task_type = cfg["task"]["type"]
    if task_type == "regression":
        loss_name = cfg["loss"].get("regression", "mse").lower()
        if loss_name == "mae":
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss()
    else:
        return nn.CrossEntropyLoss()


def build_dataloaders(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    meta_scaler=None,
) -> tuple:
    """Build train and validation DataLoaders."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    task_type = cfg["task"]["type"]

    input_size = model_cfg["input_size"]
    image_dir = data_cfg["image_dir"]
    image_col = data_cfg.get("image_col", "filename")
    label_col = data_cfg.get("label_col", "age")
    metadata_cols = data_cfg.get("metadata_cols", []) or []

    train_ds = FishAgeDataset(
        image_dir=image_dir,
        metadata_df=train_df,
        image_col=image_col,
        label_col=label_col,
        metadata_cols=metadata_cols,
        transform=get_train_transforms(input_size),
        task_type=task_type,
        meta_scaler=meta_scaler,
        is_train=True,
    )
    val_ds = FishAgeDataset(
        image_dir=image_dir,
        metadata_df=val_df,
        image_col=image_col,
        label_col=label_col,
        metadata_cols=metadata_cols,
        transform=get_eval_transforms(input_size),
        task_type=task_type,
        meta_scaler=meta_scaler,
        is_train=False,
    )

    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True)
    batch_size = train_cfg["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# One epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, task_type
) -> float:
    model.train()
    total_loss = 0.0
    for images, metadata, labels in loader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            preds = model(images, metadata)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, task_type) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, metadata, labels in loader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            preds = model(images, metadata)
            loss = criterion(preds, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    metrics = compute_metrics(preds_arr, targets_arr, task_type)
    return avg_loss, metrics, preds_arr, targets_arr


# ──────────────────────────────────────────────────────────────────────────────
# Main fold training function
# ──────────────────────────────────────────────────────────────────────────────

def train_fold(
    cfg: dict,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    experiment_dir: str,
    device: torch.device,
) -> dict:
    """Train for one fold and return the best validation metrics."""
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    task_type = cfg["task"]["type"]
    model_name = cfg["model"]["name"]

    fold_dir = Path(experiment_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ── Metadata scaler (fit on train only) ───────────────────────────────────
    metadata_cols = data_cfg.get("metadata_cols", []) or []
    meta_scaler = None
    if metadata_cols:
        available_cols = [c for c in metadata_cols if c in train_df.columns]
        if available_cols:
            meta_scaler = StandardScaler()
            meta_scaler.fit(train_df[available_cols].fillna(0).values)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg, train_df, val_df, meta_scaler)

    # ── Model / optimiser / scheduler / loss ──────────────────────────────────
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = build_criterion(cfg)
    scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir = cfg["logging"].get("log_dir", "runs")
    run_name = f"{model_name}_fold{fold}"
    tb_writer = setup_tensorboard(log_dir, run_name)

    csv_path = str(fold_dir / "training_logs.csv")
    fieldnames = ["epoch", "train_loss", "val_loss", "mae", "rmse", "exact_acc", "pm1_acc", "lr"]
    csv_writer, csv_file = setup_csv_logger(csv_path, fieldnames)

    # ── Early stopping state ──────────────────────────────────────────────────
    patience = train_cfg.get("early_stopping_patience", 10)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_metrics = {}

    epochs = train_cfg["epochs"]
    print(f"\n  [Fold {fold}] Training {model_name} for up to {epochs} epochs …")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, task_type
        )
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, device, task_type
        )

        # LR scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # TensorBoard
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            for k, v in val_metrics.items():
                tb_writer.add_scalar(f"Metrics/{k}", v, epoch)
            tb_writer.add_scalar("LR", current_lr, epoch)

        # CSV
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "mae": val_metrics.get("mae", ""),
            "rmse": val_metrics.get("rmse", ""),
            "exact_acc": val_metrics.get("exact_acc", ""),
            "pm1_acc": val_metrics.get("pm1_acc", ""),
            "lr": current_lr,
        }
        log_metrics_csv(csv_writer, row)
        csv_file.flush()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = dict(val_metrics)
            best_metrics["val_loss"] = val_loss
            epochs_no_improve = 0
            ckpt_path = str(fold_dir / "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mae={val_metrics.get('mae', 0):.3f}  "
            f"±1acc={val_metrics.get('pm1_acc', 0):.3f}  "
            f"lr={current_lr:.2e}  [{elapsed:.1f}s]"
        )

        if epochs_no_improve >= patience:
            print(f"  [Fold {fold}] Early stopping after {epoch} epochs.")
            break

    csv_file.close()
    if tb_writer:
        tb_writer.close()

    print(f"  [Fold {fold}] Best val_loss={best_val_loss:.4f}  metrics={best_metrics}")
    return best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fish Age Prediction – Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to model-specific YAML config",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run a single fold index (0-based). Default: run all folds.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    # ── Load and merge configs ─────────────────────────────────────────────────
    base_cfg = load_config("configs/base_config.yaml")
    model_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, model_cfg)

    if args.seed is not None:
        cfg["data"]["seed"] = args.seed

    seed = cfg["data"].get("seed", 42)
    set_seed(seed)
    configure_determinism(cfg["training"].get("deterministic", True))

    device = get_device()

    # ── Load dataset ──────────────────────────────────────────────────────────
    metadata_csv = cfg["data"]["metadata_csv"]
    df = load_metadata(metadata_csv)

    n_folds = cfg["data"].get("n_folds", 3)
    val_split = cfg["data"].get("val_split", 0.15)
    splits = get_kfold_splits(df, n_folds=n_folds, seed=seed)

    model_name = cfg["model"]["name"]
    experiment_dir = os.path.join(
        cfg["logging"].get("experiment_dir", "experiments"), model_name
    )

    folds_to_run = [args.fold] if args.fold is not None else list(range(n_folds))
    all_fold_metrics = []

    for fold_idx in folds_to_run:
        train_idx, test_idx = splits[fold_idx]
        fold_df = df.iloc[train_idx].reset_index(drop=True)

        # Split fold's train data into train / val
        n_val = max(1, int(len(fold_df) * val_split))
        val_df = fold_df.iloc[:n_val].reset_index(drop=True)
        train_df = fold_df.iloc[n_val:].reset_index(drop=True)

        metrics = train_fold(
            cfg=cfg,
            fold=fold_idx,
            train_df=train_df,
            val_df=val_df,
            experiment_dir=experiment_dir,
            device=device,
        )
        metrics["fold"] = fold_idx
        all_fold_metrics.append(metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training complete – {len(all_fold_metrics)} fold(s) ran.")
    for m in all_fold_metrics:
        print(
            f"  Fold {m['fold']}:  mae={m.get('mae', 'n/a'):.3f}  "
            f"rmse={m.get('rmse', 'n/a'):.3f}  "
            f"±1acc={m.get('pm1_acc', 'n/a'):.3f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
