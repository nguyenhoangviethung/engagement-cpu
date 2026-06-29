# Engagement DAiSEE - Triple XGB 504-feature pipeline

Repo này đã được rút gọn để chỉ giữ pipeline cần thiết cho product:

```text
raw video -> MediaPipe 504 feature windows -> tsfresh-like tabular features -> Triple XGBoost fusion
```

Product hiện tại:

| Hạng mục | Giá trị |
| :--- | :--- |
| Model | `triple_xgb_depth_robust_target_band_product` |
| Input | `.npy` window shape `(30, 504)` |
| Manifest | `data/processed/final_feature_manifest.csv` |
| Accuracy | `76.85%` |
| Balanced Accuracy | `83.20%` |
| F1 macro | `76.91%` |
| Model-side latency mean | `24.80 ms` |
| Raw-video E2E mean | `4.70 s` |

Mục tiêu reproduce là Triple XGB có accuracy trong khoảng `75-77%` và `balanced_accuracy > 75%`.

## Data và model trên Hugging Face

Repo HF:

```text
Hnug/daisee-processed
```

Các artifact cần thiết:

```text
data/processed/final_feature_manifest.csv
data/processed/feature_manifest.csv
data/processed/engagement_only_labels.csv
data/processed/runs/triple_xgb_504_features/**
checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip
checkpoints/runs/triple_xgb_depth_robust_maxacc_product.zip
```

Tất cả path trong manifest dùng đường dẫn tương đối.

## Chạy lại từ manifest đã có

Không cần extract lại nếu đã có `data/processed/runs/triple_xgb_504_features`.

```bash
bash scripts/reproduce_triple_xgb.sh
```

Kết quả sẽ ghi vào:

```text
checkpoints/runs/triple_xgb_target_band_repro/
checkpoints/reports/triple_xgb_repro_summary.json
```

## Extract lại 504 feature nếu cần

Chỉ chạy khi muốn tạo lại `.npy` từ raw video:

```bash
bash scripts/extract_504_features.sh \
  --labels-csv data/processed/engagement_only_labels.csv \
  --raw-video-dir data/raw/daisee/DAiSEE/DataSet \
  --output-dir data/processed/runs/triple_xgb_504_features
```

Extractor tạo mỗi window có shape `(30, 504)`:

```text
504 = 168 raw landmark-depth features
    + 168 velocity features
    + 168 window-std features
```

## Notebook pipeline

Notebook gọn để kiểm tra từng khối:

```text
notebooks/triple_xgb_pipeline.ipynb
```

Các khối chính:

1. đọc manifest;
2. kiểm tra `.npy` shape `(30, 504)`;
3. build tabular feature cho XGBoost;
4. reproduce Triple XGB;
5. xem metric/HF artifact.

## Báo cáo

```text
checkpoints/reports/GUIDE.md
checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md
```
