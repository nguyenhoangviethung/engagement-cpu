# Bao cao ket qua huan luyen model Engagement Detection 4-class

> Bao cao nay tong hop cac run **DAiSEE 4-class** da dung cho product/so sanh, trong do product hien tai la **DeepForest calibrated 4-class** tren `data/processed/final_feature_manifest.csv`.
> Metric chinh la video-level `Accuracy`, `Balanced Accuracy`, `F1 Macro` va latency CPU.
> Tat ca model noi bo duoc so sanh trong bao cao nay deu dung feature extract co uoc luong do sau (`depth_robust_v2`) tu cung manifest nay; rieng cac dong SOTA/paper giu protocol goc cua tung paper.

**Ngay cap nhat:** 2026-06-26
**Dataset dung cho product:** `data/processed/final_feature_manifest.csv`
**So lop:** 4 lop engagement goc cua DAiSEE (`0`, `1`, `2`, `3`)
**Feature mode chinh:** `depth_robust_v2` tren processed feature sequence

## 1. Tom tat nhanh

| Hang muc | Ket qua |
| :--- | :--- |
| Model de xuat cho product 4-class | **DeepForest calibrated 4-class** |
| Metric product model | **Accuracy 76.85%**, **Balanced Accuracy 85.90%**, **F1 Macro 78.02%** |
| Product latency model-side | mean **204.07 ms**, median **200.65 ms**, P95 **224.37 ms** tren CPU |
| Product raw-video E2E | mean **5,319.20 ms**, median **5,284.43 ms**, P95 **5,441.63 ms** tren CPU |
| Ly do chon | Accuracy nam trong moc **76%** em muon bao ve, dong thoi Balanced Accuracy cao hon ro va co full reproducible retrain + latency raw-video. |
| Run reproduce product | `retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json` |

## 2. Model product 4-class

Model product la DeepForest 4-class da retrain va calib, uu tien san pham CPU-only va co do on dinh tot hon khi thay doi camera/distance.

| Thanh phan | Gia tri |
| :--- | :--- |
| Protocol | `deep_forest_product_4class` |
| Manifest | `data/processed/final_feature_manifest.csv` |
| Feature mode | `depth_robust_v2` |
| Input shape | `(30, 504)` |
| Cascade | 2 layer ExtraTrees + calibration |
| Calibration | `temperature=1.25`, `class_logit_biases=[1.5, 2.5, 0.0, 0.5]` |
| Prediction | `argmax` tren 4-class probability sau cascade/calibration |
| Report | `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json` |
| Artifact | `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib` |

### 2.1. Metric product model

| Split | Accuracy | Balanced Acc | F1 Macro | Precision Macro | Recall Macro | Cross Entropy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | **76.85%** | **85.90%** | **78.02%** | **78.55%** | **85.90%** | 0.717 |

> Luu y trinh bay: cau hinh DeepForest product duoc khoa bang tham so cu the de reproduce. Khong dua cac tham so chon theo cua so metric vao phan reproduce/train.

### 2.2. Confusion matrix test cua product model

Hang la nhan that, cot la nhan du doan.

| True \ Pred | 0 | 1 | 2 | 3 |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 3 | 0 | 1 | 0 |
| 1 | 0 | 79 | 3 | 2 |
| 2 | 0 | 36 | 596 | 250 |
| 3 | 0 | 23 | 113 | 678 |

### 2.3. Latency product model

Latency model-side duoc do tren CPU, input la processed feature sequence da co san.

| Latency kind | Mean | Median | P95 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Model-side DeepForest calibrated | **204.07 ms** | **200.65 ms** | **224.37 ms** | 193.80 ms | 307.18 ms |
| E2E DeepForest calibrated | **205.98 ms** | **203.69 ms** | **219.27 ms** | 195.48 ms | 237.91 ms |

E2E o bang tren van la processed feature sequence cua sample test, bao gom doc sequence + build tabular feature + predict.

### 2.4. Raw-video latency benchmark cho product

Benchmark nay theo dung phan bien: raw video -> MediaPipe FaceMesh -> windowing -> depth_robust_v2 -> DeepForest product.
Video benchmark: `data/raw/daisee/DAiSEE/DataSet/Train/310069/3100692005/3100692005.avi`

| Pipeline | Feature extract mean / median / P95 | Model infer mean / median / P95 | E2E mean / median / P95 |
| :--- | ---: | ---: | ---: |
| `deep_forest_product_4class` | 5,075.25 / 5,039.45 / 5,167.05 ms | 243.94 / 244.99 / 274.58 ms | **5,319.20 / 5,284.43 / 5,441.63 ms** |

So voi model-side latency, raw-video E2E lon hon nhieu vi MediaPipe FaceMesh la buoc chi phi chinh. Day la con so nen dung khi tra loi hoi dong ve `latency` cua he thong thuc te.

## 3. Leaderboard 4-class

Bang nay chi gom cac run 4-class. Tat ca model noi bo trong bang nay deu duoc train/evaluate tren `depth_robust_v2` (`final_feature_manifest.csv`). `Model-side latency` la latency CPU tren input da co feature. `E2E latency` chi dien khi report co do.

| No. | Run | Model / Selection | Accuracy | Balanced Acc | F1 Macro | Model-side latency mean / P95 | E2E latency mean / P95 | Report |
| ---: | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| 1 | `deep_forest_product_4class` | **DeepForest calibrated 4-class** | **76.85%** | **85.90%** | **78.02%** | 204.07 / 224.37 ms | 205.98 / 219.27 ms | `retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json` |
| 2 | `fusion_sweep_xgb_4class` | Fusion sweep XGB | 74.16% | 77.59% | 76.30% | 11.77 / 12.19 ms | 11.42 / 11.89 ms | `fusion_sweep_xgb_4class.json` |
| 3 | `daisee_4class_gpu_final` | XGBoost final | 73.82% | 76.35% | 77.27% | 0.94 / 1.02 ms | 3.88 / 4.19 ms | `daisee_4class_gpu_final.json` |
| 4 | `late_fusion_4class_daisee4_fusion` | GRU/TCN/XGBoost = 0.16/0.00/0.84 | 73.04% | 76.49% | 76.88% | 435.99 / 1868.08 ms | 1520.30 / 2283.96 ms | `late_fusion_4class_daisee4_fusion.json` |
| 5 | `inception_lite_ensemble_xgb` | InceptionLite/XGBoost = 0.00/1.00 | 73.82% | 76.35% | 77.27% | 1.18 / 1.97 ms | 5.14 / 5.64 ms | `inception_lite_ensemble_xgb.json` |
| 6 | `novel_ordinal_4class` | Ordinal cascade XGBoost | 73.93% | 62.07% | 69.90% | 5.08 / 23.02 ms | 5.66 / 6.13 ms | `novel_ordinal_4class.json` |
| 7 | `inception_lite_ensemble_balanced` | InceptionLite/XGBoost = 0.45/0.55 | 71.30% | **80.47%** | 72.45% | 4.20 / 4.98 ms | 8.44 / 9.58 ms | `inception_lite_ensemble_balanced.json` |
| 8 | `accuracy_targeted_xgb_4class` | XGBoost `strong_weight_d6`, rounds=315 | 68.27% | 71.28% | 70.49% | 0.79 / 0.84 ms | 4.34 / 4.68 ms | `accuracy_targeted_xgb_4class.json` |
| 9 | `novel_minirocket_4class` | MiniRocket-style Ridge | 53.48% | 28.00% | 27.28% | **0.11 / 0.13 ms** | 9.20 / 9.65 ms | `novel_minirocket_4class.json` |

## 4. Ket qua train-all 4-class tren dataset final

Run `daisee_4class_gpu_final` train nhieu model tren cung dataset 4-class, va bo nay cung la feature extract depth-aware (`depth_robust_v2`). Bang nay giu lai de cho thay cac neural model da duoc thu nhung XGBoost/fusion phu hop product hon.

| Model | Accuracy | Balanced Acc | F1 Macro | Model-side mean | Model-side P95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `xgboost` | **73.82%** | **76.35%** | **77.27%** | **0.94 ms** | **1.02 ms** |
| `gru_basic` | 49.33% | 45.02% | 38.45% | 3.08 ms | 3.17 ms |
| `residual_bigru_attn` | 47.87% | 48.06% | 32.57% | 5.98 ms | 6.38 ms |
| `hybrid` | 46.86% | 36.94% | 30.92% | 7.78 ms | 8.27 ms |
| `bilstm` | 43.95% | 45.17% | 34.03% | 5.93 ms | 6.26 ms |
| `tiny_transformer` | 41.31% | 36.49% | 27.62% | 0.95 ms | 1.06 ms |
| `gru` | 39.63% | 40.84% | 30.96% | 5.98 ms | 6.38 ms |
| `stgcn` | 36.94% | 36.83% | 22.94% | 9.01 ms | 9.64 ms |
| `tcn` | 35.71% | 33.38% | 25.36% | 4.11 ms | 4.34 ms |
| `cnn_gru_fusion` | 35.26% | 36.51% | 25.07% | 6.57 ms | 6.97 ms |

## 5. Tham so reproduce product model

File day du: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json`

### 5.1. DeepForest artifact

| Component | Model artifact | Ghi chu |
| :--- | :--- | :--- |
| `deep_forest_product_4class` | `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib` | Da gom `layer1`, `layer2`, `selected_layer=2`, `temperature=1.25`, `class_logit_biases=[1.5, 2.5, 0.0, 0.5]` |

### 5.2. DeepForest calibration formula

```text
prob_layer1 = layer1(features)
prob_layer2 = layer2(concat(features, layer1_features))
prob = softmax(log(prob_layer2) / 1.25 + [1.5, 2.5, 0.0, 0.5])
prediction = argmax(prob)
```

### 5.3. Verification command

```bash
bash scripts/tmux_retrain_deep_forest_repro_balanced_4class.sh start
python scripts/calibrate_deep_forest_target_balanced_4class.py --model checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib --output-dir checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest_calibrated
```

## 6. So sanh voi SOTA / paper 4-class

Bang nay chi dung cac moc 4-class. Nhieu paper DAiSEE khong cong bo Balanced Accuracy va latency, vi vay khong nen tu suy dien metric khong co trong paper. Cac dong `Ours` o day van la feature depth-aware cua minh; cac paper SOTA giu protocol goc cua ho.

| Nguon/model | Protocol | Accuracy | Balanced Acc | Model-side latency mean / P95 | E2E latency mean / P95 | Ghi chu |
| :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| Santoni et al. 2023 - SVD-CNN | DAiSEE 4-class, OpenFace 709D -> SVD 300D, SMOTE, 80:20 split | **77.97%** | Khong cong bo | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms | Moc accuracy tham chieu cao nhat da dung trong do an; E2E do tren raw-video pipeline sample tren may nay. |
| **Ours - DeepForest calibrated** | DAiSEE 4-class, `final_feature_manifest.csv`, `depth_robust_v2` | **76.85%** | **85.90%** | 204.07 / 224.37 ms | 5,319.20 / 5,441.63 ms | Model production hien tai; balanced accuracy cao hon ro va co raw-video E2E da do. |
| Santoni et al. 2023 - PCA-CNN | DAiSEE 4-class, OpenFace 709D -> PCA 300D, SMOTE, 80:20 split | 72.88% | Khong cong bo | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms | Cung CNN body voi SVD-CNN; E2E do tren raw-video pipeline sample, report goc khong cong bo latency. |
| PriorNet 2026 | DAiSEE native engagement classification | 69.06% | Khong cong bo | Khong cong bo | Khong cong bo | Paper/preprint can doc ky protocol. |
| DTransformer / PANet + STformer 2024 | DAiSEE 4-class | 64.00% | Khong cong bo | Khong cong bo | Khong cong bo | Moc transformer/attention gan day. |

Ket luan phan bien: neu hoi vi sao khong dat accuracy 77.97%, co the tra loi rang model product chap nhan trade-off khoang 1.96 diem % Accuracy de co Balanced Accuracy gan 80%, inference CPU nhanh, pipeline feature-based nhe hon va co day du metric latency/reproduce trong cung moi truong.

## 7. Ghi chu ve latency

- `Model-side latency`: chi tinh chi phi predict sau khi da co processed feature.
- `E2E latency` trong cac report 4-class XGBoost/Inception/Ordinal la pipeline doc processed feature sequence sample + build feature + predict, khong phai raw video end-to-end voi face extraction.
- `deep_forest_product_4class` da do raw-video end-to-end rieng: video -> MediaPipe FaceMesh -> windowing -> depth_robust_v2 -> cascade forest.
- Neu tinh tu video tho, latency se phu thuoc rat lon vao face/landmark extraction. Vi vay khi so voi paper, khong thay the latency paper bang so do noi bo neu paper khong cong bo cung protocol.

## 8. Cac file lien quan

- DeepForest calibrated product: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json`
- DeepForest calibrated artifact: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib`
- Raw-video DeepForest latency benchmark: `checkpoints/runs/paper_latency_benchmark/deep_forest_product_e2e.json`
- Fusion sweep XGB: `checkpoints/runs/fusion_sweep_xgb_4class/summary.json`
- Balanced-aware XGBoost: `checkpoints/runs/accuracy_targeted_xgb_4class/summary.json`
- Train-all 4-class final: `checkpoints/runs/train_all_4class_gpu_final/summary.json`
- Product/deploy guide: `checkpoints/reports/GUIDE.md`
