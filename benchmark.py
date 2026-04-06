"""Benchmark pipeline for fish age prediction.

Trains InceptionV3, EfficientNet-B4, and ConvNeXt hybrid using identical
seeds, folds, and preprocessing. Results are saved under results/ and
experiment folders are created under experiments/.

Usage:
    python benchmark.py [--seed 42] [--folds 3]
"""

import argparse
import os
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
from train import (
    get_device,
    build_optimizer,
    build_scheduler,
    build_criterion,
    build_dataloaders,
    train_one_epoch,
    evaluate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark configuration
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_MODELS = [
    {
        "name": "InceptionV3",
        "config": "configs/inception.yaml",
    },
    {
        "name": "EfficientNet-B4",
        "config": "configs/efficientnet.yaml",
    },
    {
        "name": "ConvNeXt-Hybrid",
        "config": "configs/convnext_hybrid.yaml",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Per-model training wrapper
# ──────────────────────────────────────────────────────────────────────────────

def run_model_benchmark(
    model_entry: dict,
    df: pd.DataFrame,
    splits: list,
    seed: int,
    device: torch.device,
    results_dir: str,
    experiment_base_dir: str,
) -> dict:
    """Run k-fold benchmark for a single model and return aggregated metrics."""
    model_display_name = model_entry["name"]
    config_path = model_entry["config"]

    print(f"\n{'=' * 60}")
    print(f"  Benchmarking: {model_display_name}")
    print(f"  Config:       {config_path}")
    print(f"{'=' * 60}")

    base_cfg = load_config("configs/base_config.yaml")
    model_cfg = load_config(config_path)
    cfg = merge_configs(base_cfg, model_cfg)

    # Force common seed and fold count across all models
    cfg["data"]["seed"] = seed

    task_type = cfg["task"]["type"]
    val_split = cfg["data"].get("val_split", 0.15)
    model_name = cfg["model"]["name"]
    n_folds = len(splits)

    experiment_dir = Path(experiment_base_dir) / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save hyperparameters summary
    _save_hyperparams(cfg, experiment_dir)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n  --- Fold {fold_idx + 1}/{n_folds} ---")

        # Reset seed for each fold for reproducibility
        set_seed(seed + fold_idx)

        fold_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Split fold train into train / val
        n_val = max(1, int(len(fold_df) * val_split))
        val_df = fold_df.iloc[:n_val].reset_index(drop=True)
        train_df = fold_df.iloc[n_val:].reset_index(drop=True)

        fold_metrics = _train_and_evaluate_fold(
            cfg=cfg,
            fold=fold_idx,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            experiment_dir=str(experiment_dir),
            device=device,
        )
        fold_metrics["fold"] = fold_idx
        fold_metrics["model"] = model_display_name
        fold_results.append(fold_metrics)

        # Save per-fold metrics
        fold_results_path = experiment_dir / f"fold_{fold_idx}" / "metrics.csv"
        pd.DataFrame([fold_metrics]).to_csv(fold_results_path, index=False)

    # ── Aggregate across folds ────────────────────────────────────────────────
    metric_keys = ["mae", "rmse", "exact_acc", "pm1_acc", "test_mae", "test_rmse",
                   "test_exact_acc", "test_pm1_acc"]
    aggregated = {"model": model_display_name, "config": config_path}
    for k in metric_keys:
        vals = [r[k] for r in fold_results if k in r]
        if vals:
            aggregated[f"{k}_mean"] = float(np.mean(vals))
            aggregated[f"{k}_std"] = float(np.std(vals))

    # Save all-fold results
    all_folds_df = pd.DataFrame(fold_results)
    all_folds_df.to_csv(experiment_dir / "all_folds.csv", index=False)

    print(f"\n  [Summary] {model_display_name}")
    print(f"  Val  MAE   = {aggregated.get('mae_mean', 0):.3f} ± {aggregated.get('mae_std', 0):.3f}")
    print(f"  Val  RMSE  = {aggregated.get('rmse_mean', 0):.3f} ± {aggregated.get('rmse_std', 0):.3f}")
    print(f"  Val  ±1acc = {aggregated.get('pm1_acc_mean', 0):.3f} ± {aggregated.get('pm1_acc_std', 0):.3f}")
    print(f"  Test MAE   = {aggregated.get('test_mae_mean', 0):.3f} ± {aggregated.get('test_mae_std', 0):.3f}")
    print(f"  Test ±1acc = {aggregated.get('test_pm1_acc_mean', 0):.3f} ± {aggregated.get('test_pm1_acc_std', 0):.3f}")

    return aggregated


def _train_and_evaluate_fold(
    cfg, fold, train_df, val_df, test_df, experiment_dir, device
):
    """Train one fold and evaluate on val + test sets."""
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    task_type = cfg["task"]["type"]
    model_name = cfg["model"]["name"]

    fold_dir = Path(experiment_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Metadata scaler (fit on train only)
    metadata_cols = data_cfg.get("metadata_cols", []) or []
    meta_scaler = None
    if metadata_cols:
        available_cols = [c for c in metadata_cols if c in train_df.columns]
        if available_cols:
            meta_scaler = StandardScaler()
            meta_scaler.fit(train_df[available_cols].fillna(0).values)

    # DataLoaders
    train_loader, val_loader = build_dataloaders(cfg, train_df, val_df, meta_scaler)

    # Test DataLoader
    input_size = cfg["model"]["input_size"]
    image_dir = data_cfg["image_dir"]
    image_col = data_cfg.get("image_col", "filename")
    label_col = data_cfg.get("label_col", "age")
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)

    test_ds = FishAgeDataset(
        image_dir=image_dir,
        metadata_df=test_df,
        image_col=image_col,
        label_col=label_col,
        metadata_cols=metadata_cols,
        transform=get_eval_transforms(input_size),
        task_type=task_type,
        meta_scaler=meta_scaler,
        is_train=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Model / optimiser / loss
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = build_criterion(cfg)
    amp_scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))

    # Logging
    log_dir = cfg["logging"].get("log_dir", "runs")
    tb_writer = setup_tensorboard(log_dir, f"{model_name}_fold{fold}")

    csv_path = str(fold_dir / "training_logs.csv")
    fieldnames = ["epoch", "train_loss", "val_loss", "mae", "rmse",
                  "exact_acc", "pm1_acc", "lr"]
    csv_writer, csv_file = setup_csv_logger(csv_path, fieldnames)

    patience = train_cfg.get("early_stopping_patience", 10)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_val_metrics = {}
    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler, task_type
        )
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, device, task_type
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            for k, v in val_metrics.items():
                tb_writer.add_scalar(f"Metrics/{k}", v, epoch)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = dict(val_metrics)
            epochs_no_improve = 0
            torch.save(model.state_dict(), str(fold_dir / "best_model.pth"))
        else:
            epochs_no_improve += 1

        print(
            f"    Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"mae={val_metrics.get('mae', 0):.3f}  lr={current_lr:.2e}"
        )

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}.")
            break

    csv_file.close()
    if tb_writer:
        tb_writer.close()

    # ── Evaluate on test set with best checkpoint ─────────────────────────────
    best_ckpt = fold_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(str(best_ckpt), map_location=device))

    _, test_metrics, _, _ = evaluate(model, test_loader, criterion, device, task_type)

    combined = {
        **{k: v for k, v in best_val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "val_loss": best_val_loss,
    }
    return combined


def _save_hyperparams(cfg: dict, experiment_dir: Path) -> None:
    """Save a flattened hyperparameter summary to CSV."""
    rows = []
    for section, values in cfg.items():
        if isinstance(values, dict):
            for k, v in values.items():
                rows.append({"section": section, "key": k, "value": str(v)})
        else:
            rows.append({"section": "root", "key": section, "value": str(values)})
    pd.DataFrame(rows).to_csv(experiment_dir / "hyperparams.csv", index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fish Age Prediction – Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds")
    args = parser.parse_args()

    seed = args.seed
    n_folds = args.folds

    print("\n" + "=" * 60)
    print("  Fish Age Prediction Benchmark")
    print(f"  Seed={seed}  Folds={n_folds}")
    print("=" * 60)

    set_seed(seed)
    configure_determinism(True)
    device = get_device()

    # ── Load base config and dataset ──────────────────────────────────────────
    base_cfg = load_config("configs/base_config.yaml")
    metadata_csv = base_cfg["data"]["metadata_csv"]
    df = load_metadata(metadata_csv)

    splits = get_kfold_splits(df, n_folds=n_folds, seed=seed)

    results_dir = base_cfg["logging"].get("results_dir", "results")
    experiment_base_dir = base_cfg["logging"].get("experiment_dir", "experiments")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(experiment_base_dir).mkdir(parents=True, exist_ok=True)

    # ── Run each model ────────────────────────────────────────────────────────
    all_results = []
    start_time = time.time()

    for model_entry in BENCHMARK_MODELS:
        result = run_model_benchmark(
            model_entry=model_entry,
            df=df,
            splits=splits,
            seed=seed,
            device=device,
            results_dir=results_dir,
            experiment_base_dir=experiment_base_dir,
        )
        all_results.append(result)

    elapsed = time.time() - start_time

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(results_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Full results saved to: {results_path}")

    # ── Build summary table ───────────────────────────────────────────────────
    summary_cols = ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std",
                    "pm1_acc_mean", "pm1_acc_std",
                    "test_mae_mean", "test_mae_std",
                    "test_pm1_acc_mean", "test_pm1_acc_std"]
    existing_cols = [c for c in summary_cols if c in results_df.columns]
    summary_df = results_df[existing_cols]
    summary_path = os.path.join(results_dir, "summary_table.csv")
    summary_df.to_csv(summary_path, index=False)

    # ── Print concise summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<20}  {'Val MAE':>8}  {'Val ±1acc':>10}  {'Test MAE':>9}  {'Test ±1acc':>11}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*11}")
    for _, row in summary_df.iterrows():
        print(
            f"  {str(row.get('model','')):<20}  "
            f"{row.get('mae_mean', 0):8.3f}  "
            f"{row.get('pm1_acc_mean', 0):10.3f}  "
            f"{row.get('test_mae_mean', 0):9.3f}  "
            f"{row.get('test_pm1_acc_mean', 0):11.3f}"
        )
    print("=" * 60)
    print(f"\n  Total benchmark time: {elapsed/60:.1f} min")
    print(f"  Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()
