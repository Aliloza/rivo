# Fish Age Prediction Benchmark

A reproducible scientific benchmark for predicting fish age from segmented
otolith (ear-stone) ROI images, with optional metadata fusion.

Three model architectures are compared:

| Model | Type |
|---|---|
| InceptionV3 | Image-only baseline |
| EfficientNet-B4 | Image-only baseline |
| ConvNeXt + Metadata | Hybrid image + metadata fusion |

---

## Project Structure

```
rivo/
├── benchmark.py              # Run full benchmark (all models, all folds)
├── train.py                  # Train a single model
├── requirements.txt
│
├── configs/
│   ├── base_config.yaml      # Shared defaults
│   ├── inception.yaml        # InceptionV3 overrides
│   ├── efficientnet.yaml     # EfficientNet-B4 overrides
│   └── convnext_hybrid.yaml  # ConvNeXt hybrid overrides
│
├── data/
│   ├── dataset.py            # FishAgeDataset (images + metadata)
│   ├── augmentation.py       # Train / eval transforms
│   └── splits.py             # K-fold split utilities
│
├── models/
│   ├── base_model.py         # Abstract base class
│   ├── inception_model.py
│   ├── efficientnet_model.py
│   └── convnext_hybrid_model.py
│
├── utils/
│   ├── config.py             # YAML loading & merging
│   ├── metrics.py            # MAE, RMSE, exact / ±1 accuracy
│   ├── logging.py            # CSV & TensorBoard helpers
│   └── reproducibility.py   # Seed & determinism control
│
├── experiments/              # Per-model fold checkpoints (auto-created)
└── results/                  # Summary CSVs (auto-created)
```

---

## Installation

```bash
git clone https://github.com/Aliloza/rivo.git
cd rivo
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Data Layout

Place your data in the following structure (or update the paths in
`configs/base_config.yaml`):

```
data/
├── images/          # Segmented ROI images (PNG / JPEG)
│   ├── fish_001.jpg
│   ├── fish_002.jpg
│   └── ...
└── metadata.csv     # One row per image
```

### metadata.csv format

| filename | age | length | weight | … |
|---|---|---|---|---|
| fish_001.jpg | 3 | 24.5 | 180.2 | … |

- **filename** – image file name (must match files in `data/images/`)
- **age** – ground-truth age (integer or float)
- Any additional numeric columns can be used as metadata features

---

## Configuration

Edit `configs/base_config.yaml` to set your data paths:

```yaml
data:
  image_dir: "data/images"
  metadata_csv: "data/metadata.csv"
  image_col: "filename"
  label_col: "age"
  metadata_cols: []          # e.g. ["length", "weight"]
```

Model-specific configs (`inception.yaml`, `efficientnet.yaml`,
`convnext_hybrid.yaml`) inherit from the base config and override only
the values they need.

---

## Running the Benchmark

```bash
python benchmark.py
```

Optional flags:

```bash
python benchmark.py --seed 42 --folds 3
```

Outputs:
- `results/results.csv` – per-model, per-fold metrics
- `results/summary_table.csv` – mean ± std across folds
- `experiments/<model>/fold_<k>/best_model.pth` – best checkpoints
- `experiments/<model>/fold_<k>/training_logs.csv` – epoch logs
- `runs/<model>_fold<k>/` – TensorBoard event files

---

## Training a Single Model

```bash
python train.py --config configs/inception.yaml
python train.py --config configs/efficientnet.yaml --seed 0
python train.py --config configs/convnext_hybrid.yaml --fold 1
```

---

## Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error (years) |
| RMSE | Root Mean Squared Error (years) |
| Exact Acc | Fraction where rounded prediction = true age |
| ±1 Acc | Fraction where prediction is within 1 year |

---

## Augmentation Policy

| Split | Augmentations |
|---|---|
| Train | Resize, HFlip, Rotation ±15°, ColorJitter, GaussianBlur, Normalize |
| Val / Test | Resize, Normalize only |

---

## Requirements

- torch ≥ 2.0
- torchvision ≥ 0.15
- timm ≥ 0.9
- numpy, pandas, scikit-learn
- Pillow, PyYAML, tqdm
- tensorboard (optional, for TensorBoard logging)

Install all at once:

```bash
pip install -r requirements.txt
```