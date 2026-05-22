# Project Structure

## Canonical Source Code
- `src/engagement_daisee/common/`: shared config and constants.
- `src/engagement_daisee/rnn/`: MediaPipe + GRU engagement pipeline.
- `src/engagement_daisee/cnn/`: CNN baseline pipeline.
- `src/engagement_daisee/ml/`: classical ML baseline (XGBoost).

## Root Compatibility Wrappers
These files remain in root to keep old commands/scripts working:
- `config.py`
- `preprocess_labels.py`
- `extract_features.py`
- `dataset.py`
- `model.py`
- `train.py`
- `cnn_extract_frames.py`
- `cnn_dataset.py`
- `cnn_model.py`
- `train_cnn.py`
- `train_ml.py`

They forward execution/imports to `src/engagement_daisee/...`.

## Script Options by Module
- `scripts/rnn/`: tmux scripts for RNN/MediaPipe pipeline.
- `scripts/cnn/`: tmux scripts for CNN baseline pipeline.
- `scripts/ml/`: tmux scripts for ML baseline pipeline.
- `scripts/ab/`: tmux scripts for A/B experiments.

Legacy script paths in `scripts/*.sh` are still preserved.
