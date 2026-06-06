# Huong dan infer cho production pipeline

Tai lieu nay mo ta cach dung cac artifact hien co de suy luan trong moi truong production. Muc tieu la deploy nhanh, uu tien CPU, va tan dung cac model da duoc day len Hugging Face duoi dang zip.

## 1. Chon pipeline nao cho production

Co 2 huong infer phu hop nhat voi repo hien tai:

1. `RNN GRU`:
   - Chat luong tot nhat trong bao cao.
   - Nen dung cho production neu uu tien do chinh xac.
   - Nen convert checkpoint `.pt` thanh TorchScript `.ts` truoc khi deploy CPU.

2. `XGBoost final`:
   - Nhe hon, de deploy hon.
   - Hop khi uu tien don gian va latency phan model.
   - Van can sequence `.npy` dau vao, sau do tu trich feature tabular trong luc infer.

Khuyen nghi mac dinh:

- Neu can model manh nhat: dung `train_all_train_all.zip` -> `rnn_gru`.
- Neu can model nhe va don gian: dung `trainml_train_all_ml.zip` -> `engagement_xgb.json`.

## 2. Dieu can nho nhat ve dau vao

Entry point infer hien tai trong repo khong nhan video tho truc tiep.

Ca 2 nhanh `rnn` va `ml` deu nhan dau vao la file `sequence .npy`:

- `rnn`: [src/engagement_daisee/rnn/infer.py](../../src/engagement_daisee/rnn/infer.py)
- `ml`: [src/engagement_daisee/ml/infer.py](../../src/engagement_daisee/ml/infer.py)

Dieu nay co nghia la production pipeline can co 2 tang:

1. `Preprocess`: video/frame stream -> feature extraction -> windowing/sequence `.npy`
2. `Model infer`: `.npy` -> xac suat + nhan

Neu muon pipeline online tu webcam/video:

- MediaPipe feature extraction: [src/engagement_daisee/mediapipe/extract_features.py](../../src/engagement_daisee/mediapipe/extract_features.py)
- RNN-style sequence extraction: [src/engagement_daisee/rnn/extract_features.py](../../src/engagement_daisee/rnn/extract_features.py)

## 3. Artifact can lay tu Hugging Face

Tren HF, checkpoint dang duoc luu chu yeu duoi dang zip tung run, vi script upload dung `--zip-runs`:

- [scripts/data/hf_push_data.sh](../../scripts/data/hf_push_data.sh:309)

No zip toan bo noi dung cua moi run folder roi moi upload. Nghia la model weights, summary json, eval json, preprocess artifact... nam ben trong file zip cua run.

Nhung goi can quan tam:

- `checkpoints/runs/train_all_train_all.zip`
  - chua cac model `rnn_gru`, `rnn_tcn`, `rnn_tiny_transformer`, `gru_basic`
- `checkpoints/runs/trainml_train_all_ml.zip`
  - chua model XGBoost final
- `checkpoints/runs/train_all_phase34_20260523_030236.zip`
  - chua CNN tot nhat trong nhom da bao cao

Ngoai ra, cac file tong hop da co san tren HF:

- `checkpoints/reports/selected_model_performance.csv`
- `checkpoints/reports/run_metrics_manifest709.csv`
- `checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md`

## 4. Pipeline de xuat cho production

### 4.1 Pipeline A: RNN GRU production

Day la pipeline nen uu tien neu ban can model manh nhat.

Buoc 1: lay va giai nen run zip

- Lay `checkpoints/runs/train_all_train_all.zip` tu HF
- Giai nen vao mot thu muc tam, vi du:

```bash
mkdir -p /tmp/engagement_prod
unzip train_all_train_all.zip -d /tmp/engagement_prod/train_all_train_all
```

Sau khi giai nen, checkpoint can dung la:

```text
/tmp/engagement_prod/train_all_train_all/rnn_gru/engagement_gru.pt
```

Buoc 2: convert checkpoint sang artifact deploy CPU

Script convert:

- [src/engagement_daisee/rnn/optimize_inference.py](../../src/engagement_daisee/rnn/optimize_inference.py)

Script nay tao ra:

- `engagement_fp32.ts`
- `engagement_int8_dynamic.ts`
- `inference_meta.json`

Vi du:

```bash
PYTHONPATH=src python -m engagement_daisee.rnn.optimize_inference \
  --checkpoint /tmp/engagement_prod/train_all_train_all/rnn_gru/engagement_gru.pt \
  --output-dir checkpoints/deploy/gru_final \
  --cpu-threads 2 \
  --benchmark-iters 200
```

Artifact deploy sau cung:

```text
checkpoints/deploy/gru_final/engagement_int8_dynamic.ts
checkpoints/deploy/gru_final/inference_meta.json
```

Buoc 3: chay infer

```bash
PYTHONPATH=src python -m engagement_daisee.rnn.infer \
  --artifact checkpoints/deploy/gru_final/engagement_int8_dynamic.ts \
  --meta checkpoints/deploy/gru_final/inference_meta.json \
  --sequence /path/to/sequence.npy \
  --cpu-threads 2
```

Output la JSON gom:

- `threshold`
- `probabilities`
- `predictions`

### 4.2 Pipeline B: XGBoost production

Day la pipeline nhe hon, de rollout hon neu ban da co sequence `.npy`.

Buoc 1: lay va giai nen run zip

- Lay `checkpoints/runs/trainml_train_all_ml.zip` tu HF
- Giai nen ra thu muc tam

Artifact can dung:

```text
/tmp/engagement_prod/trainml_train_all_ml/engagement_xgb.json
/tmp/engagement_prod/trainml_train_all_ml/engagement_xgb.summary.json
/tmp/engagement_prod/trainml_train_all_ml/engagement_xgb.preprocess.npz
```

Buoc 2: chay infer

```bash
PYTHONPATH=src python -m engagement_daisee.ml.infer \
  --model /tmp/engagement_prod/trainml_train_all_ml/engagement_xgb.json \
  --summary-json /tmp/engagement_prod/trainml_train_all_ml/engagement_xgb.summary.json \
  --sequence /path/to/sequence.npy \
  --feature-mode tsfresh
```

Luu y:

- `ml.infer` se tu bien doi sequence thanh feature tabular thong qua `_sequence_to_tabular_features`.
- `summary_json` duoc dung de lay `selected_threshold`.

## 5. Chay bang tmux scripts co san

Neu muon chay infer bang script bao log/tmux thay vi goi module Python truc tiep:

### RNN

- [scripts/rnn/tmux_infer.sh](../../scripts/rnn/tmux_infer.sh)

Vi du:

```bash
./scripts/rnn/tmux_infer.sh start \
  --artifact checkpoints/deploy/gru_final/engagement_int8_dynamic.ts \
  --meta checkpoints/deploy/gru_final/inference_meta.json \
  --sequence /path/to/sequence.npy \
  --cpu-threads 2
```

### ML

- [scripts/ml/tmux_infer.sh](../../scripts/ml/tmux_infer.sh)

Vi du:

```bash
./scripts/ml/tmux_infer.sh start \
  --model /tmp/engagement_prod/trainml_train_all_ml/engagement_xgb.json \
  --summary-json /tmp/engagement_prod/trainml_train_all_ml/engagement_xgb.summary.json \
  --sequence /path/to/sequence.npy \
  --feature-mode tsfresh
```

## 6. Production pipeline toi thieu

Neu trien khai thanh service, pipeline toi thieu nen la:

1. Nhan video stream hoac frame stream
2. Extract feature bang MediaPipe
3. Gom frame thanh mot sequence co do dai co dinh
4. Luu/tao tensor hoac `.npy` trung gian
5. Goi model infer
6. Ap threshold tu metadata
7. Tra ve:
   - `probability`
   - `prediction`
   - timestamp/window id

Kieu shape hien tai:

- RNN infer ky vong `(T, F)` hoac `(B, T, F)`
- ML infer ky vong sequence `.npy`, sau do noi bo chuyen thanh vector tabular

## 7. Khuyen nghi deploy thuc te

- Chon `RNN GRU` neu uu tien chat luong tong the.
- Chon `XGBoost` neu uu tien pipeline gon va model-side infer nhanh.
- Dung TorchScript `int8_dynamic` cho CPU-serving cua RNN.
- Giu `inference_meta.json` di kem artifact, vi no chua threshold can dung luc infer.
- Khong nen dung checkpoint `.pt` truc tiep trong production neu ban da co the convert sang `.ts`.

## 8. Ngoai le: focus monitor webcam

Repo co mot ung dung webcam realtime:

- [src/engagement_daisee/app/focus_monitor.py](../../src/engagement_daisee/app/focus_monitor.py)
- [scripts/app/run_focus_monitor_webcam.sh](../../scripts/app/run_focus_monitor_webcam.sh)

Nhanh nay la heuristic multi-signal tren MediaPipe, khong phai chinh model DAiSEE final trong bao cao. No phu hop cho demo hoac prototype webcam, nhung khong nen xem la duong infer chuan cua `gru final` hoac `xgboost final`.

## 9. Checklist truoc khi go live

- Da co artifact tu HF zip va da giai nen
- Da co sequence `.npy` dung shape
- Da xac nhan threshold trong `inference_meta.json` hoac `engagement_xgb.summary.json`
- Da benchmark tren may deploy that su
- Da log lai `probability`, `prediction`, `model_version`, `threshold`
- Da co fallback khi khong detect duoc mat hoac sequence rong
