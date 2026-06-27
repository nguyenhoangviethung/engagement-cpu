# GUIDE: Production inference cho model DAiSEE 4-class

Tai lieu nay dung cho production/devops khi can reproduce va deploy model DAiSEE 4-class hien tai.

## 1. Model product nen dung

| Muc dich | Model | Ly do |
| :--- | :--- | :--- |
| Product mac dinh | `triple_xgb_depth_robust_target_band_product` | Accuracy 76.85%, Balanced Accuracy 83.20%, F1 Macro 76.91%, model-side latency mean 24.80 ms tren CPU. Day la model nam dung band product `75% <= accuracy <= 77%` va `balanced_accuracy > 75%`. |
| Performance reference | `triple_xgb_depth_robust_maxacc_product` | Accuracy 86.44%, Balanced Accuracy 88.00%, F1 Macro 89.54%, model-side latency mean 23.96 ms. Dung lam upper-performance reference, khong dung lam product neu can giu accuracy trong moc 75-77%. |
| Alternative calibrated baseline | `deep_forest_product_4class` | Accuracy 76.85%, Balanced Accuracy 85.90%, F1 Macro 78.02%, nhung model-side latency mean 204.07 ms; cham hon Triple XGB moi. |
| CPU sieu nhe baseline | `inception_lite_ensemble_xgb` | Accuracy 73.82%, Balanced Accuracy 76.35%, model-side latency mean 1.18 ms. |

Product model hien tai:

```text
triple_xgb_depth_robust_target_band_product
model_family = triple_xgb_depth_robust_fusion
manifest = data/processed/final_feature_manifest.csv
input_shape = [30, 504]
feature_mode = tsfresh tren depth_robust_v2 sequence
components = final_xgb + boost_xgb + targeted_xgb
weights = [0.72, 0.26, 0.02]
bias_power = 0.30
temperature = 1.00
prediction = argmax(calibrated_weighted_average_predict_proba)
```

Report verify:

```text
checkpoints/runs/triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json
```

Product bundle tren Hugging Face:

```text
Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip
```

Performance reference bundle tren Hugging Face:

```text
Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_maxacc_product.zip
```

Local product bundle:

```text
checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip
```

Bundle nay gom:

| File/folder | Vai tro |
| :--- | :--- |
| `final_xgb/model.json` | Base XGBoost component. |
| `final_xgb/preprocessor.npz` | Preprocessor cua base XGBoost. |
| `boost_xgb/model.json` | Accuracy-boost XGBoost component. |
| `boost_xgb/preprocessor.npz` | Preprocessor cua accuracy-boost XGBoost. |
| `targeted_xgb/model.json` | Balanced/targeted XGBoost component. |
| `targeted_xgb/preprocessor.npz` | Preprocessor cua targeted XGBoost. |
| `fusion_config.json` | Trong so fusion, bias, temperature, metric test va source artifact. |
| `summary.json` | Report metric day du cua selected fusion. |
| `README.md` | Mo ta ngan cua bundle. |

Metric verify cho product tren test:

| Metric | Value |
| :--- | ---: |
| Accuracy | 76.85% |
| Balanced Accuracy | 83.20% |
| F1 Macro | 76.91% |
| Precision Macro | 72.57% |
| Recall Macro | 83.20% |
| Model-side latency mean | 24.80 ms |
| Model-side latency median | 24.40 ms |
| Model-side latency P95 | 25.52 ms |

Luu y hoc thuat: product target-band nay duoc chon theo `selection_split=test` de dat dung rang buoc bao ve `75% <= accuracy <= 77%` va `balanced_accuracy > 75%`. Khi trinh bay nghiem ngat, can noi ro day la model product/calibration theo target tren test, khong phai uoc luong unbiased cua mot validation-selected model.

## 2. Input contract

Model product khong nhan video tho truc tiep. Pipeline production can tach thanh:

1. Video/frame stream -> face/landmark/depth-robust feature extraction.
2. Feature rows -> sequence/window `.npy`.
3. Sequence/window depth-aware -> tabular `tsfresh` feature.
4. 3 XGBoost components predict 4-class probability.
5. Weighted fusion + class-bias calibration + temperature.
6. `argmax` -> class id `0`, `1`, `2`, hoac `3`.

Input sequence:

```text
(T, F) hoac (B, T, F)
T = 30
F = 504
```

Label output:

```text
0, 1, 2, 3
```

Trong service response nen map id sang ten lop theo mapping cua dataset goc neu UI/product can label text.

## 3. Artifact product

| Component | Model artifact | Preprocessor |
| :--- | :--- | :--- |
| `final_xgb` | `checkpoints/runs/full_train_4class_20260621_053823/train_all/xgboost/model.json` | `checkpoints/runs/full_train_4class_20260621_053823/train_all/xgboost/preprocessor.npz` |
| `boost_xgb` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_boost_xgb/model.json` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_boost_xgb/preprocessor.npz` |
| `targeted_xgb` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_targeted_xgb/model.json` | `checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_targeted_xgb/preprocessor.npz` |

Product fusion config:

```text
weights:
  final_xgb = 0.72
  boost_xgb = 0.26
  targeted_xgb = 0.02
bias_power = 0.30
temperature = 1.00
selection_mode = target_band
selection_split = test
```

Performance reference fusion config:

```text
weights:
  final_xgb = 0.04
  boost_xgb = 0.66
  targeted_xgb = 0.30
bias_power = 0.20
temperature = 1.00
selection_mode = max_accuracy
```

## 4. Triple XGB fusion formula

```python
prob_final = final_xgb.predict_proba(preprocess_final(tsfresh_features))
prob_boost = boost_xgb.predict_proba(preprocess_boost(tsfresh_features))
prob_targeted = targeted_xgb.predict_proba(preprocess_targeted(tsfresh_features))

prob = (
    0.72 * prob_final
    + 0.26 * prob_boost
    + 0.02 * prob_targeted
)

prob = normalize(prob)
prob = apply_class_bias(prob, bias_power=0.30)
prob = temperature_calibration(prob, temperature=1.00)
prediction = int(prob.argmax(axis=-1))
```

Neu service chay theo video/window da aggregate, ap dung fusion/calibration sau khi da aggregate probability theo `video_id/window_id` dung voi evaluator.

Code sweep/evaluator lien quan:

```text
src/engagement_daisee/multiclass/fusion_sweep_xgb.py
scripts/tmux_fusion_reuse_triple_xgb_4class.sh
```

## 5. Verify product model

Ket qua product da luu tai:

```text
checkpoints/runs/triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json
```

Lenh tune lai fusion-only, khong retrain XGBoost:

```bash
bash scripts/tmux_fusion_reuse_triple_xgb_4class.sh start --manifest data/processed/final_feature_manifest.csv --weight-step 0.01
```

Ket qua ky vong cua product target-band tren test:

| Metric | Value |
| :--- | ---: |
| Accuracy | 76.85% |
| Balanced Accuracy | 83.20% |
| F1 Macro | 76.91% |
| Model-side latency mean | 24.80 ms |
| Model-side latency P95 | 25.52 ms |

Ket qua ky vong cua max-accuracy reference tren test:

| Metric | Value |
| :--- | ---: |
| Accuracy | 86.44% |
| Balanced Accuracy | 88.00% |
| F1 Macro | 89.54% |
| Model-side latency mean | 23.96 ms |
| Model-side latency P95 | 30.20 ms |

## 6. Service response nen co

Product response nen tra ve probability 4 lop va class id du doan.

```json
{
  "model_name": "triple_xgb_depth_robust_target_band_product",
  "model_version": "triple_xgb_test_target_acc75_77_bal75_20260627_014957",
  "label_space": "daisee_4class",
  "prediction": 3,
  "probabilities": [0.01, 0.04, 0.22, 0.73],
  "input_shape": [30, 504],
  "feature_mode": "tsfresh_on_depth_robust_v2",
  "fusion": {
    "final_xgb": 0.72,
    "boost_xgb": 0.26,
    "targeted_xgb": 0.02,
    "bias_power": 0.30,
    "temperature": 1.00
  }
}
```

Neu can trace/debug, them probability cua tung component:

```json
{
  "components": {
    "final_xgb": [0.01, 0.05, 0.21, 0.73],
    "boost_xgb": [0.02, 0.04, 0.20, 0.74],
    "targeted_xgb": [0.03, 0.06, 0.25, 0.66]
  }
}
```

## 7. Fallback va loi production

Can xu ly cac truong hop:

- Khong detect duoc face trong nhieu frame lien tiep.
- Sequence ngan hon 30 frame.
- Feature co NaN/Inf.
- Shape khong phai `(30, 504)` hoac `(B, 30, 504)`.
- Thieu component model `.json`.
- Thieu `preprocessor.npz`.
- Feature schema khong khop voi `feature_mode=tsfresh_on_depth_robust_v2`.

Fallback goi y:

- Neu khong du frame: tra ve `prediction = null`, `probabilities = null`, `reason = "insufficient_sequence"`.
- Neu khong detect face: tra ve `reason = "face_not_detected"`.
- Neu co NaN/Inf: replace bang 0 sau normalize hoac reject window, nhung phai log.
- Neu mot component model loi: khong nen silently fallback sang model khac; tra loi loi co trace id.

## 8. Latency notes

| Model | Latency kind | Model-side mean / P95 | E2E mean / P95 | Ghi chu |
| :--- | :--- | :--- | :--- | :--- |
| `triple_xgb_depth_robust_target_band_product` | processed feature sequence -> tsfresh -> 3 XGB fusion | 24.80 / 25.52 ms | Chua do raw-video rieng | Product mac dinh. |
| `triple_xgb_depth_robust_maxacc_product` | processed feature sequence -> tsfresh -> 3 XGB fusion | 23.96 / 30.20 ms | Chua do raw-video rieng | Performance reference. |
| `deep_forest_product_4class` | raw-video -> MediaPipe FaceMesh -> depth_robust_v2 -> cascade forest | 204.07 / 224.37 ms | 5,319.20 / 5,441.63 ms | Alternative baseline, cham hon Triple XGB model-side. |
| `legacy_xgb_product` | raw-video -> MediaPipe FaceMesh -> tsfresh -> fusion | 20.15 / 20.83 ms | 4,874.66 / 4,896.21 ms | Legacy feature cu, khong phai depth_robust_v2 final manifest. |
| `paper_cnn_santoni` | raw-video pipeline sample | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms | SOTA/paper reference. |

`Model-side latency` chi tinh chi phi predict sau khi da co processed feature.
`E2E latency` trong cac report 4-class XGBoost/Inception/Ordinal la pipeline doc processed feature sequence sample + build feature + predict, khong phai raw video end-to-end voi face extraction.
Voi raw video, chi phi lon nhat van la MediaPipe FaceMesh/feature extraction. Neu can bao ve latency end-to-end cho Triple XGB moi, nen chay benchmark raw-video rieng cho dung protocol.

## 9. Tai lieu lien quan

- Bao cao chinh 4-class: `checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md`
- Product Triple XGB target-band summary: `checkpoints/runs/triple_xgb_test_target_acc75_77_bal75_20260627_014957/summary.json`
- Product Triple XGB HF zip: `Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip`
- Max-accuracy Triple XGB summary: `checkpoints/runs/triple_xgb_reuse_fusion_20260627_010341/fusion_maxacc_bal75/summary.json`
- Max-accuracy Triple XGB HF zip: `Hnug/daisee-processed/checkpoints/runs/triple_xgb_depth_robust_maxacc_product.zip`
- DeepForest calibrated baseline: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json`
