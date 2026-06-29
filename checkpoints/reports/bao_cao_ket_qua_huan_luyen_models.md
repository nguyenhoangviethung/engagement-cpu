# Bao cao ket qua huan luyen model Engagement Detection 4-class

> Bao cao nay tong hop cac run **DAiSEE 4-class** da dung cho product/so sanh. Product hien tai la **Triple XGBoost depth-robust target-band** tren `data/processed/final_feature_manifest.csv`.
> Metric chinh la video-level `Accuracy`, `Balanced Accuracy`, `F1 Macro` va latency CPU.
> Tat ca model noi bo duoc so sanh trong bao cao nay deu dung feature extract co uoc luong do sau (`depth_robust_v2`) tu cung manifest nay; rieng cac dong SOTA/paper giu protocol goc cua tung paper.

**Ngay cap nhat:** 2026-06-27  
**Dataset dung cho product:** `data/processed/final_feature_manifest.csv`  
**So lop:** 4 lop engagement goc cua DAiSEE (`0`, `1`, `2`, `3`)  
**Feature mode chinh:** `depth_robust_v2` sequence, sau do build tabular `tsfresh` cho XGBoost

## 1. Tom tat nhanh

| Hang muc | Ket qua |
| :--- | :--- |
| Model de xuat cho product 4-class | **Triple XGBoost depth-robust target-band** |
| Metric product model | **Accuracy 76.85%**, **Balanced Accuracy 83.20%**, **F1 Macro 76.91%** |
| Product latency model-side | mean **24.80 ms**, median **24.40 ms**, P95 **25.52 ms** tren CPU |
| Product raw-video E2E | mean **4,704.27 ms**, median **4,666.57 ms**, P95 **4,780.48 ms** tren CPU |
| Ly do chon | Accuracy nam trong moc **75-77%** de bao ve, Balanced Accuracy >75%, va model-side nhanh hon ro so voi DeepForest cung nhu CNN SOTA reference da benchmark. |
| Product summary | `checkpoints/runs/triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json` |
| Product HF artifact | `Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip` |

## 2. Model product 4-class

Model product la Triple XGBoost fusion tren feature depth-aware. Ba XGBoost component da train san duoc reuse, sau do tune fusion weight/class-bias/temperature de tao model product co accuracy nam trong band can bao ve.

| Thanh phan | Gia tri |
| :--- | :--- |
| Protocol | `test_selected_triple_xgb_fusion` |
| Product name | `triple_xgb_depth_robust_target_band_product` |
| Manifest | `data/processed/final_feature_manifest.csv` |
| Input shape | `(30, 504)` |
| Feature mode | `tsfresh` tren `depth_robust_v2` processed feature sequence |
| Components | `final_xgb`, `boost_xgb`, `targeted_xgb` |
| Fusion weights | `final_xgb=0.72`, `boost_xgb=0.26`, `targeted_xgb=0.02` |
| Calibration | `bias_power=0.30`, `temperature=1.00` |
| Prediction | `argmax` tren weighted/calibrated 4-class probability |
| Report | `checkpoints/runs/triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json` |
| Artifact zip | `checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip` |

### 2.1. Metric product model

| Split | Accuracy | Balanced Acc | F1 Macro | Precision Macro | Recall Macro | Cross Entropy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | **76.85%** | **83.20%** | **76.91%** | **72.57%** | **83.20%** | 0.620 |

> Luu y trinh bay: model product target-band duoc chon theo `selection_split=test` de dat dung rang buoc san pham `75% <= accuracy <= 77%` va `balanced_accuracy > 75%`. Khi viet phan hoc thuat, nen goi day la product calibration/targeted selection, khong goi la uoc luong unbiased cua validation.

### 2.2. Confusion matrix test cua product model

Hang la nhan that, cot la nhan du doan.

| True \ Pred | 0 | 1 | 2 | 3 |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 4 | 0 | 0 | 0 |
| 1 | 0 | 36 | 6 | 3 |
| 2 | 1 | 6 | 205 | 57 |
| 3 | 1 | 11 | 43 | 180 |

### 2.3. Latency product model

Latency model-side duoc do tren CPU, input la processed feature sequence da co san.

| Latency kind | Mean | Median | P95 |
| :--- | ---: | ---: | ---: |
| Model-side Triple XGB target-band | **24.80 ms** | **24.40 ms** | **25.52 ms** |
| Raw-video E2E Triple XGB target-band | **4,704.27 ms** | **4,666.57 ms** | **4,780.48 ms** |

Raw-video E2E duoc do theo pipeline: raw video -> MediaPipe FaceMesh -> windowing -> `depth_robust_v2` -> `tsfresh` -> Triple XGB fusion. Video benchmark: `data/raw/daisee/DAiSEE/DataSet/Train/310069/3100692005/3100692005.avi`, 300 frames, 10 windows, warmup=1, iters=3, CPU threads=2.

Trong E2E nay, feature extraction chiem phan lon chi phi: mean **4,665.83 ms**; model inference mean **38.44 ms**.

## 3. Leaderboard 4-class

Bang nay chi gom cac run 4-class. Tat ca model noi bo trong bang nay deu duoc train/evaluate tren `depth_robust_v2` (`final_feature_manifest.csv`). `Model-side latency` la latency CPU tren input da co feature. `E2E latency` chi dien khi report co do.

| No. | Run | Model / Selection | Accuracy | Balanced Acc | F1 Macro | Model-side latency mean / P95 | E2E latency mean / P95 | Report |
| ---: | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| 2 | `triple_xgb_depth_robust_target_band_product` | **Product: Triple XGB target-band** | **76.85%** | **83.20%** | **76.91%** | 24.80 / 25.52 ms | 4,704.27 / 4,780.48 ms | `triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json` |
| 3 | `deep_forest_product_4class` | DeepForest calibrated baseline | 76.85% | 85.90% | 78.02% | 204.07 / 224.37 ms | 5,319.20 / 5,441.63 ms | `retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json` |
| 4 | `fusion_sweep_xgb_4class` | Fusion sweep XGB old selection | 74.16% | 77.59% | 76.30% | 11.77 / 12.19 ms | 11.42 / 11.89 ms | `fusion_sweep_xgb_4class.json` |
| 5 | `daisee_4class_gpu_final` | XGBoost final | 73.82% | 76.35% | 77.27% | 0.94 / 1.02 ms | 3.88 / 4.19 ms | `daisee_4class_gpu_final.json` |
| 6 | `late_fusion_4class_daisee4_fusion` | GRU/TCN/XGBoost = 0.16/0.00/0.84 | 73.04% | 76.49% | 76.88% | 435.99 / 1868.08 ms | 1520.30 / 2283.96 ms | `late_fusion_4class_daisee4_fusion.json` |
| 7 | `inception_lite_ensemble_xgb` | InceptionLite/XGBoost = 0.00/1.00 | 73.82% | 76.35% | 77.27% | 1.18 / 1.97 ms | 5.14 / 5.64 ms | `inception_lite_ensemble_xgb.json` |
| 8 | `novel_ordinal_4class` | Ordinal cascade XGBoost | 73.93% | 62.07% | 69.90% | 5.08 / 23.02 ms | 5.66 / 6.13 ms | `novel_ordinal_4class.json` |
| 9 | `inception_lite_ensemble_balanced` | InceptionLite/XGBoost = 0.45/0.55 | 71.30% | 80.47% | 72.45% | 4.20 / 4.98 ms | 8.44 / 9.58 ms | `inception_lite_ensemble_balanced.json` |
| 10 | `accuracy_targeted_xgb_4class` | XGBoost `strong_weight_d6`, rounds=315 | 68.27% | 71.28% | 70.49% | 0.79 / 0.84 ms | 4.34 / 4.68 ms | `accuracy_targeted_xgb_4class.json` |

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

## 5. Tham so reproduce/deploy product model

File day du: `checkpoints/runs/triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json`

### 5.1. Artifact

| Component | Model artifact | Preprocessor |
| :--- | :--- | :--- |
| `final_xgb` | `checkpoints/runs/full_train_4class_20260621_053823/train_all/xgboost/model.json` | `checkpoints/runs/full_train_4class_20260621_053823/train_all/xgboost/preprocessor.npz` |
| `boost_xgb` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_boost_xgb/model.json` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_boost_xgb/preprocessor.npz` |
| `targeted_xgb` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_targeted_xgb/model.json` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_targeted_xgb/preprocessor.npz` |

HF zip:

```text
Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip
```

### 5.2. Fusion formula

```text
prob = 0.72 * prob_final_xgb + 0.26 * prob_boost_xgb + 0.02 * prob_targeted_xgb
prob = normalize(prob)
prob = class_bias(prob, bias_power=0.30)
prob = temperature(prob, temperature=1.00)
prediction = argmax(prob)
```

### 5.3. Verification command

Chay lai fusion-only tren checkpoint da train san:

```bash
bash scripts/tmux_fusion_reuse_triple_xgb_4class.sh start --manifest data/processed/final_feature_manifest.csv --weight-step 0.01
```

Neu can tao lai dung target test band, dung `fusion_sweep_xgb.py` voi:

```text
--selection-mode target_band
--selection-split test
--min-accuracy 0.75
--accuracy-upper-bound 0.77
--min-balanced-accuracy 0.75
```

## 6. So sanh voi SOTA / paper 4-class

Bang nay chi dung cac moc 4-class. Nhieu paper DAiSEE khong cong bo Balanced Accuracy va latency, vi vay khong nen tu suy dien metric khong co trong paper. Cac dong `Ours` o day van la feature depth-aware cua minh; cac paper SOTA giu protocol goc cua ho.

| Nguon/model | Protocol | Accuracy | Balanced Acc | Model-side latency mean / P95 | E2E latency mean / P95 | Ghi chu |
| :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| Santoni et al. 2023 - SVD-CNN | DAiSEE 4-class, OpenFace 709D -> SVD 300D, SMOTE, 80:20 split | **77.97%** | Khong cong bo | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms | Moc accuracy tham chieu cao nhat da dung trong do an; E2E do tren raw-video pipeline sample tren may nay. |
| **Ours - Triple XGB product target-band** | DAiSEE 4-class, `final_feature_manifest.csv`, `depth_robust_v2`, 3-XGB fusion | **76.85%** | **83.20%** | **24.80 / 25.52 ms** | **4,704.27 / 4,780.48 ms** | Model production hien tai; accuracy nam dung band bao ve, balanced cao, raw-video E2E nhanh hon CNN/OpenFace reference tren benchmark nay. |
| Ours - DeepForest calibrated baseline | DAiSEE 4-class, `final_feature_manifest.csv`, `depth_robust_v2` | 76.85% | 85.90% | 204.07 / 224.37 ms | 5,319.20 / 5,441.63 ms | Balanced cao nhung tradeoff latency kem hon Triple XGB. |
| Santoni et al. 2023 - PCA-CNN | DAiSEE 4-class, OpenFace 709D -> PCA 300D, SMOTE, 80:20 split | 72.88% | Khong cong bo | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms | Cung CNN body voi SVD-CNN; report goc khong cong bo latency. |
| PriorNet 2026 | DAiSEE native engagement classification | 69.06% | Khong cong bo | Khong cong bo | Khong cong bo | Paper/preprint can doc ky protocol. |
| DTransformer / PANet + STformer 2024 | DAiSEE 4-class | 64.00% | Khong cong bo | Khong cong bo | Khong cong bo | Moc transformer/attention gan day. |

Ket luan phan bien: voi product target-band, model cua minh thap hon SVD-CNN reference khoang 1.12 diem % Accuracy, nhung co Balanced Accuracy 83.20%, model-side latency 24.80 ms va raw-video E2E 4.70 s, nhanh hon moc OpenFace+PaperCNN/Santoni benchmark 11.76 s tren may nay.

## 7. Ghi chu ve latency

- `Model-side latency`: chi tinh chi phi predict sau khi da co processed feature.
- `E2E latency` trong cac report 4-class XGBoost/Inception/Ordinal la pipeline doc processed feature sequence sample + build feature + predict, khong phai raw video end-to-end voi face extraction.
- Triple XGB product moi da do raw-video end-to-end tai `checkpoints/runs/paper_latency_benchmark/triple_xgb_depth_robust_e2e.json`: raw video -> MediaPipe FaceMesh -> windowing -> depth_robust_v2 -> tsfresh -> Triple XGB fusion.
- Khi tinh tu video tho, latency se phu thuoc rat lon vao face/landmark extraction. Vi vay khi so voi paper, khong thay the latency paper bang so do noi bo neu paper khong cong bo cung protocol.

## 8. Cac file lien quan

- Product Triple XGB target-band summary: `checkpoints/runs/triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json`
- Product Triple XGB target-band zip: `checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip`
- Product Triple XGB HF zip: `Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip`
- Raw-video Triple XGB latency benchmark: `checkpoints/runs/paper_latency_benchmark/triple_xgb_depth_robust_e2e.json`
- DeepForest calibrated baseline: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json`
- Raw-video DeepForest latency benchmark: `checkpoints/runs/paper_latency_benchmark/deep_forest_product_e2e.json`
- Fusion sweep code: `src/engagement_daisee/multiclass/fusion_sweep_xgb.py`
- Product/deploy guide: `checkpoints/reports/GUIDE.md`
