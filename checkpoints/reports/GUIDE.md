# GUIDE: Production inference cho tat ca model

Tai lieu nay dung cho production/devops khi can lay artifact hien co va chay infer. Tat ca duong dan ben duoi la duong dan trong repo sau khi da doi ten run folder cho ro nghia.

## 1. Chon model

| Muc dich | Model nen dung | Ly do |
|---|---|---|
| Accuracy/F1 tot nhat | `late_fusion_gru_tcn_xgb` | Ensemble GRU + TCN + XGBoost, Accuracy 75.95%, F1 macro 73.86% |
| Single model co balanced accuracy cao nhat | `gru` | Test Balanced Accuracy 77.35% |
| Single model moi tot nhat | `residual_bigru_attn` | Balanced Accuracy 73.76%, Accuracy 72.03% |
| CPU don gian, model-side latency thap | `xgboost` | Model-side latency khoang 0.48 ms |
| Neural latency nhe hon | `cnn_gru_fusion` | Model-side latency khoang 8.96 ms |

## 2. Input contract

Model production trong bao cao khong nhan video tho truc tiep. Pipeline can tach thanh:

1. Video/frame stream -> MediaPipe FaceMesh features
2. Features -> window/sequence `.npy`
3. Sequence `.npy` -> model probability
4. Apply threshold -> prediction

Shape dau vao chuan cho RNN/TCN/hybrid:

```text
(T, F) hoac (B, T, F)
T = 30
F = 90
```

XGBoost cung nhan sequence `.npy`, sau do `ml.infer` tu chuyen sequence sang tabular feature bang `feature-mode tsfresh`.

CNN baseline la ngoai le: no nhan anh/frame RGB truc tiep, khong phai sequence.

Luu y manifest tren disk dung cot `partition` cho train/validation/test. Cac CLI van giu tham so `--split` de tuong thich voi code train/evaluate hien co.

## 3. Artifact va threshold

| Model | Artifact | Threshold | Ghi chu |
|---|---|---:|---|
| `gru` | `checkpoints/runs/final_rnn_temporal_models_20260529/rnn_gru/engagement_gru.pt` | 0.63 | Best single-model balanced accuracy |
| `tcn` | `checkpoints/runs/final_rnn_temporal_models_20260529/rnn_tcn/engagement_tcn.pt` | 0.61 | Nen dung trong late fusion |
| `gru_basic` | `checkpoints/runs/final_rnn_temporal_models_20260529/rnn_gru_basic/engagement_gru_basic.pt` | 0.61 | Baseline RNN nhe |
| `tiny_transformer` | `checkpoints/runs/final_rnn_temporal_models_20260529/rnn_tiny_transformer/engagement_tiny_transformer.pt` | 0.55 | Baseline transformer |
| `xgboost` | `checkpoints/runs/final_xgboost_tsfresh_20260529/engagement_xgb.json` | 0.49 | Can them summary/preprocess |
| `cnn_gru_fusion` | `checkpoints/runs/evolved_rnn_hybrid_models_20260608/rnn_cnn_gru_fusion/engagement_cnn_gru_fusion.pt` | 0.64 | Hybrid 1D-CNN + GRU |
| `residual_bigru_attn` | `checkpoints/runs/evolved_rnn_hybrid_models_20260608/rnn_residual_bigru_attn/engagement_residual_bigru_attn.pt` | 0.55 | Residual BiGRU + attention |
| `late_fusion_gru_tcn_xgb` | `checkpoints/reports/late_fusion_gru_tcn_xgb_report.json` | 0.54 | Fuse prob: 0.30 GRU + 0.30 TCN + 0.40 XGBoost |
| `cnn_efficientnet_b0` | `checkpoints/runs/train_all_phase34_20260523_030236/cnn/engagement_cnn.pt` | 0.19 | Frame-based baseline, khong khuyen nghi lam model chinh |

XGBoost can them:

```text
checkpoints/runs/final_xgboost_tsfresh_20260529/engagement_xgb.summary.json
checkpoints/runs/final_xgboost_tsfresh_20260529/engagement_xgb.preprocess.npz
```

## 4. Infer RNN/TCN/hybrid

Checkpoint `.pt` la training checkpoint. Khi deploy CPU, nen convert sang TorchScript truoc.

Vi du convert GRU:

```bash
PYTHONPATH=src python3 -m engagement_daisee.rnn.optimize_inference \
  --checkpoint checkpoints/runs/final_rnn_temporal_models_20260529/rnn_gru/engagement_gru.pt \
  --output-dir checkpoints/deploy/gru_final \
  --cpu-threads 2 \
  --benchmark-iters 200
```

Lenh nay tao:

```text
checkpoints/deploy/gru_final/engagement_fp32.ts
checkpoints/deploy/gru_final/engagement_int8_dynamic.ts
checkpoints/deploy/gru_final/inference_meta.json
```

Infer mot sequence:

```bash
PYTHONPATH=src python3 -m engagement_daisee.rnn.infer \
  --artifact checkpoints/deploy/gru_final/engagement_int8_dynamic.ts \
  --meta checkpoints/deploy/gru_final/inference_meta.json \
  --sequence /path/to/window.npy \
  --cpu-threads 2
```

Dung cung quy trinh tren cho:

```text
tcn
gru_basic
tiny_transformer
cnn_gru_fusion
residual_bigru_attn
```

Chi can doi `--checkpoint` va `--output-dir`.

Neu can evaluate truc tiep checkpoint `.pt` tren manifest:

```bash
PYTHONPATH=src python3 -m engagement_daisee.rnn.evaluate \
  --manifest data/processed/runs/daisee_binary_final_dataset/feature_manifest.csv \
  --checkpoint checkpoints/runs/final_rnn_temporal_models_20260529/rnn_gru/engagement_gru.pt \
  --split test \
  --threshold 0.63 \
  --aggregation video
```

## 5. Infer XGBoost

```bash
PYTHONPATH=src python3 -m engagement_daisee.ml.infer \
  --model checkpoints/runs/final_xgboost_tsfresh_20260529/engagement_xgb.json \
  --summary-json checkpoints/runs/final_xgboost_tsfresh_20260529/engagement_xgb.summary.json \
  --sequence /path/to/window.npy \
  --feature-mode tsfresh
```

Output co dang:

```json
{
  "model": ".../engagement_xgb.json",
  "sequence": ".../window.npy",
  "backend": "xgboost",
  "feature_mode": "tsfresh",
  "threshold": 0.49,
  "probability": 0.0,
  "prediction": 0
}
```

## 6. Infer late-fusion ensemble

Late-fusion hien chua co CLI single-window rieng. Production can tinh 3 xac suat rieng roi fuse:

```python
p_final = (0.30 * p_gru) + (0.30 * p_tcn) + (0.40 * p_xgb)
prediction = int(p_final >= 0.54)
```

Artifact can doc de lay weights/metrics:

```text
checkpoints/reports/late_fusion_gru_tcn_xgb_report.json
```

Thanh phan ensemble:

```text
GRU:     checkpoints/runs/final_rnn_temporal_models_20260529/rnn_gru/engagement_gru.pt
TCN:     checkpoints/runs/final_rnn_temporal_models_20260529/rnn_tcn/engagement_tcn.pt
XGBoost: checkpoints/runs/final_xgboost_tsfresh_20260529/engagement_xgb.json
```

Khuyen nghi deploy:

1. Convert GRU va TCN sang TorchScript bang `rnn.optimize_inference`.
2. Chay GRU infer lay `p_gru`.
3. Chay TCN infer lay `p_tcn`.
4. Chay XGBoost infer lay `p_xgb`.
5. Tinh `p_final` theo cong thuc tren.
6. Tra ve threshold `0.54` va weights trong response de trace.

## 7. Infer CNN baseline

CNN baseline nhan anh RGB, phu hop lam baseline frame-level hon la model chinh.

```bash
PYTHONPATH=src python3 -m engagement_daisee.cnn.infer \
  --checkpoint checkpoints/runs/train_all_phase34_20260523_030236/cnn/engagement_cnn.pt \
  --image /path/to/frame.jpg \
  --threshold 0.19
```

Output co `probability`, `prediction`, `model_name`, `image_size`.

## 8. Service response nen co

Moi response production nen log toi thieu:

```json
{
  "model_name": "late_fusion_gru_tcn_xgb",
  "model_version": "20260608",
  "threshold": 0.54,
  "probability": 0.0,
  "prediction": 0,
  "window_start_ms": 0,
  "window_end_ms": 0,
  "input_shape": [30, 90]
}
```

Voi late fusion, nen them:

```json
{
  "components": {
    "gru": 0.0,
    "tcn": 0.0,
    "xgboost": 0.0
  },
  "weights": {
    "gru": 0.30,
    "tcn": 0.30,
    "xgboost": 0.40
  }
}
```

## 9. Fallback va loi production

Can xu ly cac truong hop:

- Khong detect duoc face trong nhieu frame lien tiep.
- Sequence ngan hon 30 frame.
- Feature co NaN/Inf.
- Shape khong phai `(30, 90)` hoac `(B, 30, 90)`.
- File `.ts` khong di kem `inference_meta.json`.
- XGBoost thieu `summary.json` hoac `preprocess.npz`.

Fallback goi y:

- Neu khong du frame: tra ve `prediction = null`, `probability = null`, `reason = "insufficient_sequence"`.
- Neu khong detect face: tra ve `reason = "face_not_detected"`.
- Neu co NaN/Inf: replace bang 0 sau normalize hoac reject window, nhung phai log.

## 10. Tai lieu lien quan

- Report chinh: `checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md`
- Latency model-side: `checkpoints/reports/latency_benchmark_models.csv`
- Latency raw-video-to-prediction: `checkpoints/reports/end_to_end_latency_benchmark.csv`
- Late fusion metrics: `checkpoints/reports/late_fusion_gru_tcn_xgb_report.json`
