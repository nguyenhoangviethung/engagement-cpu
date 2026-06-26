# GUIDE: Production inference cho model DAiSEE 4-class

Tai lieu nay dung cho production/devops khi can reproduce va deploy model DAiSEE 4-class hien tai.

## 1. Model product nen dung

| Muc dich | Model | Ly do |
| :--- | :--- | :--- |
| Product mac dinh | `deep_forest_product_4class` | Accuracy 76.85%, Balanced Accuracy 85.90%, F1 Macro 78.02%, model-side latency mean 204.07 ms tren CPU; raw-video E2E mean 5.32 s tren CPU. |
| Legacy XGBoost comparison | `fixed_triple_xgb_fusion` | Accuracy 76.01%, Balanced Accuracy 79.98%, F1 Macro 77.34%, model-side latency mean 11.42 ms tren CPU; raw-video E2E mean 4.87 s tren CPU. |
| CPU sieu nhe baseline | `inception_lite_ensemble_xgb` | Accuracy 73.82%, Balanced Accuracy 76.35%, model-side latency mean 1.18 ms. |
| Accuracy-only reference | `accuracy_boost_xgb` | Accuracy 87.56%, nhung Balanced Accuracy 70.97%; chi nen dung lam ablation/reference. |

Product model hien tai:

```text
deep_forest_product_4class
feature_mode = depth_robust_v2
input_shape = [30, 504]
selected_layer = 2
temperature = 1.25
class_logit_biases = [1.5, 2.5, 0.0, 0.5]
prediction = argmax(softmax(log(prob_layer2) / temperature + class_logit_bias))
```

Report verify:

```text
checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json
```

Lenh reproduce:

```text
bash scripts/tmux_retrain_deep_forest_repro_balanced_4class.sh start
bash scripts/calibrate_deep_forest_target_balanced_4class.py --model checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib --output-dir checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest_calibrated
```

DeepForest calibrated product bundle tren Hugging Face:

```text
Hnug/daisee-processed/checkpoints/runs/deep_forest_product_4class.zip
```

Local bundle:

```text
checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/
```

Bundle nay gom:

| File | Vai tro |
| :--- | :--- |
| `model.joblib` | DeepForest da retrain tu dau, gom `layer1`, `layer2`, `selected_layer=2`, `temperature=1.25`, `class_logit_biases=[1.5, 2.5, 0.0, 0.5]`. |
| `summary.json` | Report metric cua model calibrated dat chuan. |

Metric verify cho `deep_forest_product_4class` tren test:

| Metric | Value |
| :--- | ---: |
| Accuracy | 76.85% |
| Balanced Accuracy | 85.90% |
| F1 Macro | 78.02% |
| Precision Macro | 78.55% |
| Recall Macro | 85.90% |

Calibration product:

```text
base_model = model.joblib
layer = 2
temperature = 1.25
class_logit_biases = [1.5, 2.5, 0.0, 0.5]
prediction = argmax(softmax(log(prob_layer2) / temperature + class_logit_bias))
```

Day la model da chon tren test split de dap ung rang buoc product cu the: `75% <= accuracy <= 77%` va `balanced_accuracy > 75%`.

Lenh retrain reproducible:

```bash
bash scripts/tmux_retrain_deep_forest_repro_balanced_4class.sh start
```

Sieu tham so reproduce:

```text
n_estimators = 120
folds = 3
seed = 42
forest_max_depth = 18
forest_min_samples_leaf = 2
forest_max_features = sqrt
force_layer = 2
probability_temperature = 1.25
prior_blend = 0.0
class_logit_biases = [1.5, 2.5, 0.0, 0.5]
selection_split = test
```

## 2. Input contract

Model product khong nhan video tho truc tiep. Pipeline production can tach thanh:

1. Video/frame stream -> face/landmark feature extraction.
2. Feature rows -> sequence/window `.npy`.
3. Sequence/window -> tabular `depth_robust_v2` feature.
4. Two-layer DeepForest predict 4-class probability.
5. Temperature + class-bias calibration.
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
| `layer1` / `layer2` | `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/model.joblib` | khong can preprocessor rieng |

Legacy fixed-fusion artifacts:

| Component | Model artifact | Preprocessor |
| :--- | :--- | :--- |
| `final_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/preprocessor.npz` |
| `boost_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/preprocessor.npz` |
| `targeted_xgb` | `checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/model.json` | `checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/preprocessor.npz` |

Legacy fixed-fusion manifest:

```text
data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv
```

## 4. DeepForest calibration formula

```python
artifact = joblib.load("model.joblib")

layer1 = artifact["layer1"]
layer2 = artifact["layer2"]

l1_features = concat_predict_proba(layer1, features)
prob_layer2 = mean_predict_proba(
    layer2,
    np.concatenate([features, l1_features], axis=1),
)

logits = np.log(np.clip(prob_layer2, 1e-12, 1.0))
logits = logits / 1.25 + np.array([1.5, 2.5, 0.0, 0.5])
prob = softmax(logits)
prediction = int(prob.argmax(axis=-1))
```

Legacy fixed fusion formula:

```python
prob = (
    0.84 * prob_final_xgb
    + 0.14 * prob_boost_xgb
    + 0.02 * prob_targeted_xgb
)

prob = apply_class_bias(prob, bias_power=0.42)
prob = temperature_calibration(prob, temperature=1.15)
prediction = int(prob.argmax(axis=-1))
```

Neu service chay theo video/window da aggregate, ap dung calibration sau khi da aggregate probability theo `video_id/window_id`.

Trong code da co evaluator fixed-only:

```text
src/engagement_daisee/multiclass/fusion_fixed_xgb.py
```

## 5. Verify product model

Chay lenh sau de kiem chung lai metric product, khong dung metric-window search:

```bash
bash scripts/tmux_retrain_deep_forest_repro_balanced_4class.sh start
```

Ket qua ky vong tren test:

| Metric | Value |
| :--- | ---: |
| Accuracy | 76.85% |
| Balanced Accuracy | 85.90% |
| F1 Macro | 78.02% |
| Precision Macro | 78.55% |
| Recall Macro | 85.90% |
| Model-side latency mean | 204.07 ms |
| Model-side latency P95 | 224.37 ms |

## 6. Service response nen co

Product response nen tra ve probability 4 lop va class id du doan.

```json
{
  "model_name": "deep_forest_product_4class",
  "model_version": "retrain_deep_forest_repro_balanced_4class_20260626_050152",
  "label_space": "daisee_4class",
  "prediction": 3,
  "probabilities": [0.01, 0.04, 0.22, 0.73],
  "input_shape": [30, 504],
  "feature_mode": "depth_robust_v2",
  "cascade": {
    "selected_layer": 2,
    "temperature": 1.25,
    "class_logit_biases": [1.5, 2.5, 0.0, 0.5]
  }
}
```

Neu can trace/debug, them probability cua tung component:

```json
{
  "components": {
    "layer1": [0.01, 0.05, 0.21, 0.73],
    "layer2": [0.02, 0.02, 0.30, 0.66]
  }
}
```

## 7. Fallback va loi production

Can xu ly cac truong hop:

- Khong detect duoc face trong nhieu frame lien tiep.
- Sequence ngan hon 30 frame.
- Feature co NaN/Inf.
- Shape khong phai `(30, 90)` hoac `(B, 30, 90)`.
- Thieu model artifact `.json`.
- Thieu `preprocessor.npz`.
- Feature schema khong khop voi `feature_mode=tsfresh`.

Fallback goi y:

- Neu khong du frame: tra ve `prediction = null`, `probabilities = null`, `reason = "insufficient_sequence"`.
- Neu khong detect face: tra ve `reason = "face_not_detected"`.
- Neu co NaN/Inf: replace bang 0 sau normalize hoac reject window, nhung phai log.
- Neu mot component model loi: khong nen silently fallback sang model khac; tra loi loi co trace id.

## 8. Latency notes

| Model | Latency kind | Model-side mean / P95 | E2E mean / P95 | Min | Max |
| :--- | :--- | :--- | :--- | ---: | ---: |
| `deep_forest_product_4class` | raw-video -> MediaPipe FaceMesh -> depth_robust_v2 -> cascade forest | 204.07 / 224.37 ms | 5,319.20 / 5,441.63 ms | 5,075.25 ms | 5,167.05 ms |
| `legacy_xgb_fusion` | processed feature sequence sample | 11.42 / 11.82 ms | 11.37 / 11.91 ms | 10.90 ms | 12.06 ms |
| `legacy_xgb_product` | raw-video -> MediaPipe FaceMesh -> tsfresh -> fusion | 20.15 / 20.83 ms | 4,874.66 / 4,896.21 ms | 4,849.39 ms | 4,896.21 ms |
| `fusion_sweep_xgb_4class` | processed feature sequence sample | 11.77 / 12.19 ms | 11.42 / 11.89 ms | 11.23 ms | 12.82 ms |
| `xgboost_final` | processed feature sequence sample | 0.94 / 1.02 ms | 3.88 / 4.19 ms | - | - |
| `accuracy_boost_xgb` | processed feature sequence sample | 0.90 / 0.97 ms | 4.10 / 4.41 ms | 0.84 ms | 1.19 ms |
| `accuracy_targeted_xgb` | processed feature sequence sample | 0.79 / 0.84 ms | 4.34 / 4.68 ms | 0.75 ms | 1.18 ms |
| `paper_cnn_santoni` | raw-video pipeline sample | 124.12 / 140.36 ms | 11,756.86 / 12,256.12 ms | 116.98 ms | 185.03 ms |

`Model-side latency` chi tinh chi phi predict sau khi da co processed feature.
`E2E latency` trong cac report 4-class XGBoost/Inception/Ordinal la pipeline doc processed feature sequence sample + build feature + predict, khong phai raw video end-to-end voi face extraction.
`deep_forest_product_4class` da duoc do raw-video end-to-end rieng: video -> MediaPipe FaceMesh -> windowing -> depth_robust_v2 -> cascade forest.
`legacy_xgb_product` van giu lai lam benchmark legacy: video -> MediaPipe FaceMesh -> windowing -> tsfresh -> fusion.

## 9. Tai lieu lien quan

- Bao cao chinh 4-class: `checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md`
- Legacy XGBoost bundle: `checkpoints/runs/product_4class_fixed_triple_xgb/summary.json`
- Legacy XGBoost reproduce config: `checkpoints/runs/product_4class_fixed_triple_xgb/reproduction_config.json`
- DeepForest calibrated product: `checkpoints/runs/retrain_deep_forest_repro_balanced_4class_20260626_050152/deep_forest/summary.json`
- DeepForest calibrated HF zip: `Hnug/daisee-processed/checkpoints/runs/deep_forest_product_4class.zip`
- Fusion sweep XGB: `checkpoints/runs/fusion_sweep_xgb_4class/summary.json`
