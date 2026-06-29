# GUIDE: Triple XGB 504-feature product

Tai lieu nay chi mo ta pipeline product hien tai:

```text
raw video -> MediaPipe 504 feature windows -> tsfresh-like tabular vector -> Triple XGBoost fusion
```

## 1. Product model

| Hang muc | Gia tri |
| :--- | :--- |
| Product | `triple_xgb_depth_robust_target_band_product` |
| Manifest | `data/processed/final_feature_manifest.csv` |
| Input `.npy` | `(30, 504)` |
| Feature folder | `data/processed/runs/triple_xgb_504_features` |
| Feature mode cho XGBoost | `tsfresh` |
| Target reproduce | `75% <= accuracy <= 77%`, `balanced_accuracy > 75%` |

Metric product da chot:

| Metric | Value |
| :--- | ---: |
| Accuracy | 76.85% |
| Balanced Accuracy | 83.20% |
| F1 Macro | 76.91% |
| Model-side latency mean / P95 | 24.80 / 25.52 ms |
| Raw-video E2E mean / P95 | 4,704.27 / 4,780.48 ms |

Product fusion:

```text
final_xgb    = 0.72
boost_xgb    = 0.26
targeted_xgb = 0.02
bias_power   = 0.30
temperature  = 1.00
```

## 2. Data va artifact tren Hugging Face

Repo:

```text
Hnug/daisee-processed
```

Can co cac path sau:

```text
data/processed/final_feature_manifest.csv
data/processed/feature_manifest.csv
data/processed/engagement_only_labels.csv
data/processed/runs/triple_xgb_504_features/**
checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip
checkpoints/runs/triple_xgb_depth_robust_maxacc_product.zip
```

Tat ca path trong manifest la duong dan tuong doi theo repo root.

## 3. Reproduce

Dung manifest da co, khong can extract lai:

```bash
bash scripts/reproduce_triple_xgb.sh
```

Output:

```text
checkpoints/runs/triple_xgb_target_band_repro/
checkpoints/reports/triple_xgb_repro_summary.json
```

Neu can extract lai tu raw video:

```bash
bash scripts/extract_504_features.sh \
  --labels-csv data/processed/engagement_only_labels.csv \
  --raw-video-dir data/raw/daisee/DAiSEE/DataSet \
  --output-dir data/processed/runs/triple_xgb_504_features
```

## 4. Input contract

Moi sample trong manifest tro toi mot file `.npy`:

```text
shape = (30, 504)
dtype = float32
504 = 168 raw + 168 velocity + 168 window_std
```

Service nen tra ve:

```json
{
  "model_name": "triple_xgb_depth_robust_target_band_product",
  "label_space": "daisee_4class",
  "prediction": 3,
  "probabilities": [0.01, 0.04, 0.22, 0.73],
  "input_shape": [30, 504],
  "feature_mode": "tsfresh_on_depth_robust_v2"
}
```

## 5. Latency/SOTA comparison

| Model | Accuracy | Balanced Acc | Model-side mean / P95 | Raw-video E2E mean / P95 |
| :--- | ---: | ---: | :--- | :--- |
| Ours - Triple XGB product target-band | 76.85% | 83.20% | 24.80 / 25.52 ms | 4,704.27 / 4,780.48 ms |
| Santoni/OpenFace+PaperCNN reference | 77.97% | not reported | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms |

Ket luan ngan: product thap hon SOTA reference khoang 1.12 diem accuracy, nhung co balanced accuracy cao va raw-video E2E nhanh hon tren benchmark noi bo.

## 6. Notebook

```text
notebooks/triple_xgb_pipeline.ipynb
```
