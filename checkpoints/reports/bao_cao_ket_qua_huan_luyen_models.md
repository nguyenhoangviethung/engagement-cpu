# Bao cao Triple XGB 504-feature pipeline

Bao cao nay chi giu lai pipeline product hien tai:

```text
raw video -> MediaPipe 504 feature windows -> tsfresh-like tabular features -> Triple XGBoost fusion
```

## 1. Product summary

| Hang muc | Gia tri |
| :--- | :--- |
| Product | `triple_xgb_depth_robust_target_band_product` |
| Manifest | `data/processed/final_feature_manifest.csv` |
| Feature folder | `data/processed/runs/triple_xgb_504_features` |
| Input shape | `(30, 504)` |
| Reproduce target | `75% <= accuracy <= 77%`, `balanced_accuracy > 75%` |
| HF dataset | `Hnug/daisee-processed` |

## 2. Test metrics

| Model | Accuracy | Balanced Acc | F1 Macro | Precision Macro | Recall Macro |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Triple XGB product target-band | **76.85%** | **83.20%** | **76.91%** | 72.57% | 83.20% |

Confusion matrix test:

| True \ Pred | 0 | 1 | 2 | 3 |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 4 | 0 | 0 | 0 |
| 1 | 0 | 36 | 6 | 3 |
| 2 | 1 | 6 | 205 | 57 |
| 3 | 1 | 11 | 43 | 180 |

## 3. Latency

Benchmark raw-video dung video:

```text
data/raw/daisee/DAiSEE/DataSet/Train/310069/3100692005/3100692005.avi
```

Thiet lap: 300 frames, 10 windows, warmup=1, iters=3, CPU threads=2.

| Pipeline | Feature extract mean | Model infer mean | E2E mean | E2E P95 |
| :--- | ---: | ---: | ---: | ---: |
| Triple XGB product target-band | 4,665.83 ms | 38.44 ms | **4,704.27 ms** | **4,780.48 ms** |

## 4. So sanh voi SOTA reference

| Model | Protocol | Accuracy | Balanced Acc | Model-side mean / P95 | Raw-video E2E mean / P95 |
| :--- | :--- | ---: | ---: | :--- | :--- |
| Ours - Triple XGB product target-band | DAiSEE 4-class, 504 depth-aware feature, Triple XGB fusion | 76.85% | 83.20% | 24.80 / 25.52 ms | 4,704.27 / 4,780.48 ms |
| Santoni/OpenFace+PaperCNN reference | DAiSEE 4-class, OpenFace 709D -> SVD/PCA 300D, CNN | 77.97% | not reported | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms |

Ket luan phan bien: product thap hon SOTA reference khoang 1.12 diem accuracy, nhung balanced accuracy cao va raw-video E2E nhanh hon ro tren benchmark noi bo.

## 5. Reproduce

```bash
bash scripts/reproduce_triple_xgb.sh
```

Output:

```text
checkpoints/runs/triple_xgb_target_band_repro/
checkpoints/reports/triple_xgb_repro_summary.json
```

Neu can extract lai 504 feature:

```bash
bash scripts/extract_504_features.sh \
  --labels-csv data/processed/engagement_only_labels.csv \
  --raw-video-dir data/raw/daisee/DAiSEE/DataSet \
  --output-dir data/processed/runs/triple_xgb_504_features
```

## 6. Hugging Face paths

```text
Hnug/daisee-processed/data/processed/final_feature_manifest.csv
Hnug/daisee-processed/data/processed/feature_manifest.csv
Hnug/daisee-processed/data/processed/engagement_only_labels.csv
Hnug/daisee-processed/data/processed/runs/triple_xgb_504_features/**
Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip
```

## 7. Notebook

```text
notebooks/triple_xgb_pipeline.ipynb
```
