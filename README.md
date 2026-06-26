---
pretty_name: DAiSEE 4-class Engagement Product
language:
  - vi
tags:
  - daisee
  - engagement-recognition
  - deepforest
  - mediapipe
  - depth-aware
---

# Engagement_DAiSEE

Repo này tập trung vào bài toán **DAiSEE 4-class engagement recognition** trên **processed feature sequences**.

Mục tiêu chính:
- train / val / test trong cùng một run
- lưu đầy đủ metric và latency để so sánh
- có một model product có thể reproduce và deploy lại trên CPU

## Trạng thái hiện tại

- Dataset dùng cho product: `data/processed/final_feature_manifest.csv`
- Label space: 4 lớp engagement gốc của DAiSEE
- Feature chính cho các model nội bộ: `depth_robust_v2`
- Input chính cho model product: processed feature sequence, không phải raw video
- Model product hiện tại: `deep_forest_product_4class`
- Metric product test:
  - Accuracy `76.85%`
  - Balanced Accuracy `85.90%`
  - F1 Macro `78.02%`
  - Model-side latency mean `204.07 ms`
  - E2E latency mean `205.98 ms` trên processed feature sequence sample
  - Raw-video end-to-end mean `5.32 s` trên MediaPipe FaceMesh -> `depth_robust_v2` -> DeepForest

## Artifact quan trọng

- Product summary: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json`
- Product artifact: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib`
- Báo cáo chính: `checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md`
- GUIDE production: `checkpoints/reports/GUIDE.md`

## Cấu trúc model chính

- `src/engagement_daisee/multiclass/`
  - các pipeline 4-class mới
  - `novel_models_4class.py`: DeepForest / ordinal / minirocket
  - `fusion_sweep_xgb.py`: chọn fusion theo sweep
  - `accuracy_targeted_xgb.py`: tune theo mục tiêu accuracy/balanced accuracy
  - `inception_lite_experiment.py`: Inception-lite + XGBoost fusion
  - `late_fusion.py`: late fusion nhiều nhánh
  - `train_all.py`: train/eval toàn bộ model 4-class

- `src/engagement_daisee/rnn/`
  - các model sequence gốc như `gru`, `tcn`, `tiny_transformer`, `hybrid`, `residual_bigru_attn`

- `src/engagement_daisee/ml/`
  - tabular feature engineering và XGBoost/LightGBM baseline

- `src/engagement_daisee/cnn/`
  - CNN baseline trên frame features

## Chạy lại product model

```bash
bash scripts/tmux_retrain_deep_forest_repro_balanced_4class.sh start
```

## Chạy train/all cho 4-class

```bash
bash scripts/tmux_train_all_4class.sh start
```

Các script tmux khác cho 4-class:
- `scripts/tmux_fusion_sweep_xgb_4class.sh`
- `scripts/tmux_accuracy_targeted_xgb_4class.sh`
- `scripts/tmux_inception_lite_ensemble_4class.sh`
- `scripts/tmux_late_fusion_4class.sh`
- `scripts/tmux_novel_models_4class.sh`

## Report và latency

Report đã thống nhất cách ghi:
- `Model-side latency`: latency CPU sau khi đã có processed feature
- `E2E latency`: latency trên processed feature sequence sample
- `raw-video pipeline sample`: chỉ dùng cho benchmark paper Santoni/OpenFace

Nếu cần so sánh với paper, xem:
- `checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md`
- `checkpoints/reports/GUIDE.md`

## Hugging Face

Artifact product đã được đóng gói dạng zip và đẩy lên:
- `Hnug/daisee-processed`
- remote path: `checkpoints/runs/deep_forest_product_4class.zip`

## Dữ liệu raw

Nếu cần re-extract:

```bash
bash scripts/data/download_daisee.sh
```

Raw data sẽ nằm ở:
- `data/raw/daisee/DAiSEE/`
