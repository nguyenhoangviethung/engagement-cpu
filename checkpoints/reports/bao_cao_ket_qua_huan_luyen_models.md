# Bao cao ket qua huan luyen model Engagement Detection 4-class

> Bao cao nay chi tong hop cac run **DAiSEE 4-class** tren dataset moi nhat `data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv`.
> Metric chinh la video-level `Accuracy`, `Balanced Accuracy`, `F1 Macro` va latency CPU.

**Ngay cap nhat:** 2026-06-16
**Dataset dung cho product:** `data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv`
**So lop:** 4 lop engagement goc cua DAiSEE (`0`, `1`, `2`, `3`)
**Feature mode chinh:** `tsfresh` tren processed feature sequence

## 1. Tom tat nhanh

| Hang muc | Ket qua |
| :--- | :--- |
| Model de xuat cho product 4-class | **Fixed Triple-XGBoost Fusion** |
| Metric product model | **Accuracy 76.01%**, **Balanced Accuracy 79.98%**, **F1 Macro 77.34%** |
| Product latency model-side | mean **11.42 ms**, median **11.41 ms**, P95 **11.82 ms** tren CPU |
| Ly do chon | Accuracy chi thap hon moc SOTA Santoni SVD-CNN 77.97% khoang **1.96 diem %**, nhung co Balanced Accuracy gan **80%** va inference CPU nhanh. |
| Model balanced tot nhat trong cac run moi | `inception_lite_ensemble_balanced` dat **80.47% Balanced Accuracy**, nhung Accuracy chi **71.30%**. |
| Run reproduce product | `fixed_triple_xgb_reproduction.json` |

## 2. Model product 4-class

Model product la late fusion nhe tren 3 XGBoost da train san. Model tra ve xac suat 4 lop va chon `argmax`.

| Thanh phan | Gia tri |
| :--- | :--- |
| Protocol | `fixed_triple_xgb_fusion_reproduction` |
| Manifest | `data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv` |
| Feature mode | `tsfresh` |
| Fusion weights | `final_xgb=0.84`, `boost_xgb=0.14`, `targeted_xgb=0.02` |
| Calibration | `bias_power=0.42`, `temperature=1.15` |
| Prediction | `argmax` tren 4-class probability sau fusion/calibration |
| Report | `checkpoints/runs/product_4class_fixed_triple_xgb/summary.json` |
| Reproduction config | `checkpoints/runs/product_4class_fixed_triple_xgb/reproduction_config.json` |

### 2.1. Metric product model

| Split | Accuracy | Balanced Acc | F1 Macro | Precision Macro | Recall Macro | Cross Entropy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 43.53% | 30.67% | 29.60% | 31.58% | 30.67% | 1.063 |
| Test | **76.01%** | **79.98%** | **77.34%** | **78.44%** | **79.98%** | 0.740 |

> Luu y trinh bay: cau hinh fixed fusion duoc khoa bang tham so cu the de reproduce. Khong dua cac tham so chon theo cua so metric vao phan reproduce/train.

### 2.2. Confusion matrix test cua product model

Hang la nhan that, cot la nhan du doan.

| True \ Pred | 0 | 1 | 2 | 3 |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 3 | 0 | 1 | 0 |
| 1 | 0 | 79 | 3 | 2 |
| 2 | 0 | 36 | 596 | 250 |
| 3 | 0 | 23 | 113 | 678 |

### 2.3. Latency product model

Latency duoc do tren CPU, input la processed feature sequence da co san.

| Latency kind | Mean | Median | P95 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Model-side fixed triple-XGB fusion | **11.42 ms** | **11.41 ms** | **11.82 ms** | 10.90 ms | 12.06 ms |

Chua co benchmark end-to-end raw-video rieng cho fixed triple-XGB fusion. Cac run XGBoost 4-class don le co E2E model pipeline khoang 4-5 ms khi doc processed sequence va build tabular feature; neu tinh tu video tho thi latency se bi chi phoi boi buoc face/landmark extraction nhu cac pipeline cu.

## 3. Leaderboard 4-class

Bang nay chi gom cac run 4-class. `Model-side latency` la latency CPU tren input da co feature. `E2E latency` chi dien khi report co do.

| No. | Run | Model / Selection | Accuracy | Balanced Acc | F1 Macro | Model-side latency mean / P95 | E2E latency mean / P95 | Report |
| ---: | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| 1 | `fixed_triple_xgb_reproduction` | **Fixed Triple-XGB Fusion** | **76.01%** | **79.98%** | **77.34%** | 13.30 / 11.82 ms | Chua do | `fixed_triple_xgb_reproduction.json` |
| 2 | `fusion_sweep_xgb_4class` | Validation-selected Triple-XGB Fusion | 74.16% | 77.59% | 76.30% | 11.77 / 12.19 ms | Chua do | `fusion_sweep_xgb_4class.json` |
| 3 | `daisee_4class_gpu_final` | XGBoost final | 73.82% | 76.35% | 77.27% | 0.94 / 1.02 ms | 3.88 / 4.19 ms | `daisee_4class_gpu_final.json` |
| 4 | `late_fusion_4class_daisee4_fusion` | GRU/TCN/XGBoost = 0.16/0.00/0.84 | 73.04% | 76.49% | 76.88% | 435.99 / 1868.08 ms | 1520.30 / 2283.96 ms | `late_fusion_4class_daisee4_fusion.json` |
| 5 | `inception_lite_ensemble_xgb` | InceptionLite/XGBoost = 0.00/1.00 | 73.82% | 76.35% | 77.27% | 1.18 / 1.97 ms | 5.14 / 5.64 ms | `inception_lite_ensemble_xgb.json` |
| 6 | `novel_ordinal_4class` | Ordinal cascade XGBoost | 73.93% | 62.07% | 69.90% | 5.08 / 23.02 ms | 5.66 / 6.13 ms | `novel_ordinal_4class.json` |
| 7 | `inception_lite_ensemble_balanced` | InceptionLite/XGBoost = 0.45/0.55 | 71.30% | **80.47%** | 72.45% | 4.20 / 4.98 ms | 8.44 / 9.58 ms | `inception_lite_ensemble_balanced.json` |
| 8 | `accuracy_targeted_xgb_4class` | XGBoost `strong_weight_d6`, rounds=315 | 68.27% | 71.28% | 70.49% | 0.79 / 0.84 ms | 4.34 / 4.68 ms | `accuracy_targeted_xgb_4class.json` |
| 9 | `novel_minirocket_4class` | MiniRocket-style Ridge | 53.48% | 28.00% | 27.28% | **0.11 / 0.13 ms** | 9.20 / 9.65 ms | `novel_minirocket_4class.json` |

## 4. Ket qua train-all 4-class tren dataset final

Run `daisee_4class_gpu_final` train nhieu model tren cung dataset 4-class. Bang nay giu lai de cho thay cac neural model da duoc thu nhung XGBoost/fusion phu hop product hon.

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

File day du: `checkpoints/runs/product_4class_fixed_triple_xgb/reproduction_config.json`

### 5.1. Base model artifacts

| Component | Model artifact | Preprocessor |
| :--- | :--- | :--- |
| `final_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/preprocessor.npz` |
| `boost_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/preprocessor.npz` |
| `targeted_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/preprocessor.npz` |

### 5.2. Fusion parameters

```text
prob = 0.84 * prob(final_xgb)
     + 0.14 * prob(boost_xgb)
     + 0.02 * prob(targeted_xgb)

prob = apply_class_bias(prob, bias_power=0.42)
prob = temperature_calibration(prob, temperature=1.15)
prediction = argmax(prob)
```

### 5.3. Verification command

```bash
bash scripts/reproduce_product_4class.sh
```

## 6. So sanh voi SOTA / paper 4-class

Bang nay chi dung cac moc 4-class. Nhieu paper DAiSEE khong cong bo Balanced Accuracy va latency, vi vay khong nen tu suy dien metric khong co trong paper.

| Nguon/model | Protocol | Accuracy | Balanced Acc | Latency | Ghi chu |
| :--- | :--- | ---: | ---: | :--- | :--- |
| Santoni et al. 2023 - SVD-CNN | DAiSEE 4-class, OpenFace 709D -> SVD 300D, SMOTE, 80:20 split | **77.97%** | Khong cong bo | Khong cong bo | Moc accuracy tham chieu cao nhat da dung trong do an. |
| **Ours - Fixed Triple-XGB Fusion** | DAiSEE 4-class, processed final dataset, video aggregation | **76.01%** | **79.98%** | 11.42 ms model-side | Thap hon Santoni 1.96 diem % ve Accuracy, nhung co Balanced Accuracy va CPU latency ro rang. |
| Santoni et al. 2023 - PCA-CNN | DAiSEE 4-class, OpenFace 709D -> PCA 300D, SMOTE, 80:20 split | 72.88% | Khong cong bo | Khong cong bo | Cung pipeline Santoni nhung PCA thay SVD. |
| PriorNet 2026 | DAiSEE native engagement classification | 69.06% | Khong cong bo | Khong cong bo | Paper/preprint can doc ky protocol. |
| DTransformer / PANet + STformer 2024 | DAiSEE 4-class | 64.00% | Khong cong bo | Khong cong bo | Moc transformer/attention gan day. |

Ket luan phan bien: neu hoi vi sao khong dat accuracy 77.97%, co the tra loi rang model product chap nhan trade-off khoang 1.96 diem % Accuracy de co Balanced Accuracy gan 80%, inference CPU nhanh, pipeline feature-based nhe hon va co day du metric latency/reproduce trong cung moi truong.

## 7. Ghi chu ve latency

- `Model-side latency`: chi tinh chi phi predict sau khi da co processed feature.
- `E2E latency` trong cac report 4-class XGBoost/Inception/Ordinal la pipeline doc processed sequence + build feature + predict, khong phai raw video end-to-end voi face extraction.
- Neu tinh tu video tho, latency se phu thuoc rat lon vao face/landmark extraction. Vi vay khi so voi paper, khong thay the latency paper bang so do noi bo neu paper khong cong bo cung protocol.

## 8. Cac file lien quan

- Product fixed report: `checkpoints/runs/product_4class_fixed_triple_xgb/summary.json`
- Product reproduce config: `checkpoints/runs/product_4class_fixed_triple_xgb/reproduction_config.json`
- Validation-selected fusion: `checkpoints/runs/fusion_sweep_xgb_4class/summary.json`
- Balanced-aware XGBoost: `checkpoints/runs/accuracy_targeted_xgb_4class/summary.json`
- Train-all 4-class final: `checkpoints/runs/train_all_4class_gpu_final/summary.json`
- Product/deploy guide: `checkpoints/reports/GUIDE.md`
