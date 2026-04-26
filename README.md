# Engagement Detection with DAiSEE

This repository implements a CPU-friendly, engagement-only pipeline for the DAiSEE dataset using PyTorch and MediaPipe.

## Overview

- **Task:** Binary classification for engagement only
- **Data:** DAiSEE dataset from Kaggle (`olgaparfenova/daisee`)
- **Features:** MediaPipe face mesh extracting EAR, MAR, head-pose proxy, and landmark coordinates
- **Model:** Lightweight GRU with dropout and sigmoid output
- **Mode:** CPU-only training and inference

## Environment

Use the conda environment named `thesis`.

```bash
conda run -n thesis python --version
```

Do not use or depend on `.venv` in this workspace.

## Dependencies

Install the required packages in the `thesis` environment.

```bash
conda run -n thesis python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
conda run -n thesis python -m pip install mediapipe opencv-python pandas numpy kagglehub
```

## Dataset Download

The DAiSEE dataset should be downloaded using `kagglehub` and Kaggle API credentials.

1. Ensure credentials exist at `~/.kaggle/kaggle.json`.
2. Download the dataset into the repository local data path.

```bash
conda run -n thesis python download_daisee.py
```

This script downloads the dataset and mirrors it into `data/raw/daisee/DAiSEE/` by default.

### Notes

- The raw dataset layout should include:
  - `data/raw/daisee/DAiSEE/DataSet/Train`
  - `data/raw/daisee/DAiSEE/DataSet/Validation`
  - `data/raw/daisee/DAiSEE/DataSet/Test`
  - `data/raw/daisee/DAiSEE/Labels/AllLabels.csv`
  - `data/raw/daisee/DAiSEE/GenderClips/`
  - `data/raw/daisee/DAiSEE/README.txt`

## Preprocess Labels

Filter the DAiSEE labels for engagement only and convert the engagement label to binary.

```bash
conda run -n thesis python preprocess_labels.py
```

The output is saved to `data/processed/engagement_only_labels.csv`.

## Feature Extraction

Extract per-frame MediaPipe features and save them as sequence `.npy` files.
Logging is always enabled at all levels by default (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

### Sample mode

Use `--sample` for a fast end-to-end smoke test with only 10 random videos.

```bash
conda run -n thesis python extract_features.py --sample
```

### Full extraction

```bash
conda run -n thesis python extract_features.py
```

### Control extract progress log frequency

```bash
conda run -n thesis python extract_features.py --log-every 1
```

Default save locations:

- features: `data/processed/features/`
- manifest: `data/processed/feature_manifest.csv`

## Training

Train the lightweight GRU model with validation and checkpointing.
The training split is now fixed by official DAiSEE labels:

- train: `TrainLabels.csv`
- validation: `ValidationLabels.csv`
- test: `TestLabels.csv`

No random train/validation split is used anymore.

### Sample training

```bash
conda run -n thesis python train.py --sample
```

### Full training

```bash
conda run -n thesis python train.py
```

### Control training progress log frequency

```bash
conda run -n thesis python train.py --log-every 1
```

This command trains on official train split, monitors official validation split for early stopping,
and reports final metrics on official test split.

Default checkpoint path:

- `checkpoints/engagement_gru.pt`

## File Summary

- `config.py` — hyperparameters and folder paths
- `preprocess_labels.py` — filter Engagement labels and save cleaned CSV
- `extract_features.py` — MediaPipe feature extraction into `.npy` sequences
- `dataset.py` — PyTorch `Dataset` for feature sequence loading
- `model.py` — lightweight GRU classification model
- `train.py` — CPU training loop with sample mode, early stopping, and checkpointing
- `download_daisee.py` — Kaggle dataset download script

## Recommended Workflow

1. `conda run -n thesis python download_daisee.py`
2. `conda run -n thesis python preprocess_labels.py`
3. `conda run -n thesis python extract_features.py --sample`
4. `conda run -n thesis python train.py --sample`

After confirming the pipeline works, rerun extraction and training without `--sample` for full dataset processing.

## Run with tmux (Detached by Default)

There are now 3 tmux scripts:

- `scripts/tmux_pipeline.sh` - run full pipeline (preprocess -> extract -> train)
- `scripts/tmux_extract.sh` - run extract only
- `scripts/tmux_train.sh` - run train only

All scripts support the same tmux controls:

```bash
# start (default command)
./scripts/<script_name>.sh

# or explicit start
./scripts/<script_name>.sh start

# status / attach / logs / stop
./scripts/<script_name>.sh status
./scripts/<script_name>.sh attach
./scripts/<script_name>.sh logs
./scripts/<script_name>.sh stop
```

Detach from tmux without stopping run: press `Ctrl+b`, then `d`.

### 1) Full pipeline in tmux

```bash
./scripts/tmux_pipeline.sh start
```

Sample smoke test:

```bash
./scripts/tmux_pipeline.sh start --sample
```

Useful options:

- `--env NAME` (default: `thesis`)
- `--session NAME` (default: `engagement_pipeline`)
- `--extract-log-every N`
- `--train-log-every N`
- `--run-id ID` (important for output isolation)

### 2) Extract-only in tmux

```bash
./scripts/tmux_extract.sh start
```

Sample extract:

```bash
./scripts/tmux_extract.sh start --sample
```

Useful options:

- `--env NAME` (default: `thesis`)
- `--session NAME` (default: `engagement_extract`)
- `--log-every N`
- `--run-id ID` (important for output isolation)

### 3) Train-only in tmux

```bash
./scripts/tmux_train.sh start
```

Sample train:

```bash
./scripts/tmux_train.sh start --sample
```

Useful options:

- `--env NAME` (default: `thesis`)
- `--session NAME` (default: `engagement_train`)
- `--log-every N`
- `--run-id ID` (important for checkpoint isolation)

### Run parallel tmux jobs without overwriting outputs

When you run extract and pipeline at the same time, always use different `--run-id` values (and usually different `--session` values).

Example:

```bash
./scripts/tmux_extract.sh start --session ex_a --run-id ex_a
./scripts/tmux_pipeline.sh start --session pipe_b --run-id pipe_b
```

Output isolation paths:

- Extract script writes to:
  - `data/processed/runs/extract_<run_id>/features/`
  - `data/processed/runs/extract_<run_id>/feature_manifest.csv`
- Pipeline script writes to:
  - `data/processed/runs/pipeline_<run_id>/features/`
  - `data/processed/runs/pipeline_<run_id>/feature_manifest.csv`
  - `checkpoints/runs/pipeline_<run_id>/engagement_gru.pt`
- Train-only script writes to:
  - `checkpoints/runs/train_<run_id>/engagement_gru.pt`

Log files are stored in `logs/` (`latest.log`, `latest_extract.log`, `latest_train.log` symlinks are updated per script).

`--log-level` options are removed. Logging is always enabled at all levels by default.

## Troubleshooting

- If download fails, ensure `~/.kaggle/kaggle.json` exists and contains valid Kaggle credentials.
- If MediaPipe faces are not detected, verify video file readability and codec support via OpenCV.
- This pipeline is explicitly designed for CPU usage and uses `num_workers=0` for data loading stability.
- If `tmux` is missing, install it first (for Ubuntu/Debian: `sudo apt-get install tmux`).
# engagement-cpu
