# Engagement_DAiSEE

Repo này tập trung vào bài toán **DAiSEE 4-class engagement recognition** trên **processed feature sequences**.

Mục tiêu chính:
- lưu đầy đủ metric và latency để so sánh
- có một model product có thể reproduce và deploy lại trên CPU
- tách rõ phần nghiên cứu mô hình và phần ứng dụng triển khai

## Trạng thái hiện tại

- Dataset dùng cho product: `data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv`
- Label space: 4 lớp engagement gốc của DAiSEE
- Input chính cho model product: processed feature sequence, không phải raw video
- Model product hiện tại: `fixed_triple_xgb_fusion`
- Metric product test:
  - Accuracy `76.01%`
  - Balanced Accuracy `79.98%`
  - F1 Macro `77.34%`
  - Model-side latency mean `11.42 ms`
  - E2E latency mean `11.37 ms` trên processed feature sequence sample

## Artifact quan trọng

- Product summary: `checkpoints/runs/product_4class_fixed_triple_xgb/summary.json`
- Product reproduce config: `checkpoints/runs/product_4class_fixed_triple_xgb/reproduction_config.json`
- Product artifact README: `checkpoints/runs/product_4class_fixed_triple_xgb/README.md`
- Báo cáo chính: `checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md`
- GUIDE production: `checkpoints/reports/GUIDE.md`

## Cấu trúc model chính

- `src/engagement_daisee/multiclass/`
  - các pipeline 4-class mới
  - `fusion_fixed_xgb.py`: model product
  - `fusion_sweep_xgb.py`: tìm cấu hình fusion tốt hơn cho quá trình phát triển
  - `accuracy_targeted_xgb.py`: tune theo mục tiêu accuracy/balanced accuracy
  - `inception_lite_experiment.py`: Inception-lite + XGBoost fusion
  - `late_fusion.py`: late fusion nhiều nhánh
  - `novel_models_4class.py`: các hướng model khác như ordinal / minirocket / deep forest
  - `train_all.py`: train/eval toàn bộ model 4-class

- `src/engagement_daisee/rnn/`
  - các model sequence gốc như `gru`, `tcn`, `tiny_transformer`, `hybrid`, `residual_bigru_attn`

- `src/engagement_daisee/ml/`
  - tabular feature engineering và XGBoost/LightGBM baseline

- `src/engagement_daisee/cnn/`
  - CNN baseline trên frame features

## Chạy lại product model

```bash
bash scripts/reproduce_product_4class.sh
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
- remote path: `product_4class_fixed_triple_xgb/product_4class_fixed_triple_xgb.zip`

## Dữ liệu raw

Nếu cần re-extract:

```bash
bash scripts/data/download_daisee.sh
```

Raw data sẽ nằm ở:
- `data/raw/daisee/DAiSEE/`
