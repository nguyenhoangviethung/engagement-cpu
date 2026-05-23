# Project Structure

## Canonical Source Tree

- `src/engagement_daisee/common/`
  - shared config/constants (`config.py`)
- `src/engagement_daisee/rnn/`
  - sequence pipeline based on MediaPipe features
  - `models/` now contains each architecture separately:
    - `gru.py`: `EngagementGRU` (BiGRU+Attention) + `BasicGRUClassifier` (GRU cơ bản)
    - `tcn.py`: `EngagementTCN` (1D-CNN/TCN)
    - `transformer.py`: `EngagementTinyTransformer`
    - `builder.py`: unified model factory `build_sequence_model`
  - `model.py`: compatibility facade for old imports
  - `train.py`, `evaluate.py`, `infer.py`, `optimize_inference.py`, `dataset.py`, `extract_features.py`, `preprocess_labels.py`
- `src/engagement_daisee/ml/`
  - TS-Fresh-like feature engineering + tree models (LightGBM/XGBoost)
- `src/engagement_daisee/cnn/`
  - frame-based CNN baseline

## Compatibility Notes

- Existing code importing `engagement_daisee.rnn.model` still works.
- New code should prefer `engagement_daisee.rnn.models.*`.
- Model name aliases in RNN training:
  - `gru`
  - `gru_basic` / `simple_gru`
  - `tcn` / `1dcnn` / `temporal_cnn`
  - `transformer` / `tiny_transformer`
