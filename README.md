# Engagement_DAiSEE

## 1) Mục tiêu chạy thực tế

Bạn muốn:
- train + eval trong **cùng 1 tmux session**
- tự lưu JSON để so sánh run
- có 1 script tổng chạy hết model qua đêm

Repo hiện đã hỗ trợ đúng theo luồng đó.

## 2) Dữ liệu đang dùng

Bạn đang dùng processed data tại:
- `data/processed/runs/pipeline_2/feature_manifest.csv`
- `data/processed/runs/pipeline_2/features/`

Với `rnn` và `ml`, có thể train/eval trực tiếp từ đây, không cần re-process.

## 3) Cấu trúc model (đã tách rõ)

- `src/engagement_daisee/rnn/models/gru.py`
  - `gru` (BiGRU + attention, model chính)
  - `gru_basic` / `simple_gru` (GRU cơ bản, giữ lại để baseline)
- `src/engagement_daisee/rnn/models/tcn.py`
  - `tcn` / `1dcnn` / `temporal_cnn`
- `src/engagement_daisee/rnn/models/transformer.py`
  - `transformer` / `tiny_transformer`
- `src/engagement_daisee/rnn/models/builder.py`
  - factory build model

## 4) Script train+eval theo từng module (cùng tmux)

### 4.1 RNN train+eval

```bash
./scripts/tmux_train_eval.sh rnn start \
  --session rnn_te_p2 \
  --manifest data/processed/runs/pipeline_2/feature_manifest.csv \
  --run-id p2_tcn_01 \
  --model tcn \
  --device cuda \
  --amp
```

### 4.2 ML train+eval

```bash
./scripts/tmux_train_eval.sh ml start \
  --session ml_te_p2 \
  --manifest data/processed/runs/pipeline_2/feature_manifest.csv \
  --run-id p2_ml_01 \
  --backend lightgbm \
  --feature-mode tsfresh \
  --cpu-workers 2
```

### 4.3 CNN train+eval

```bash
./scripts/tmux_train_eval.sh cnn start \
  --session cnn_te_01 \
  --manifest data/processed/cnn_frame_manifest.csv \
  --run-id cnn_01 \
  --model mobilenet_v3_small
```

## 5) File JSON được lưu để so sánh

Mỗi lần `train+eval` sẽ lưu trong thư mục run:
- `checkpoints/runs/<module>_train_eval_<run_id>/train_eval_summary.json`
- đồng thời append vào history:
  - `checkpoints/reports/rnn_train_eval_history.jsonl`
  - `checkpoints/reports/ml_train_eval_history.jsonl`
  - `checkpoints/reports/cnn_train_eval_history.jsonl`

## 6) Script tổng chạy qua đêm (1 tmux)

Script:
- `scripts/tmux_train_all.sh`

Mặc định chạy tuần tự:
- RNN: `gru gru_basic tcn tiny_transformer`
- ML: bật
- CNN: tắt mặc định (bật bằng `--with-cnn`)

Ví dụ chạy qua đêm:

```bash
./scripts/tmux_train_all.sh start \
  --session nightly_all \
  --run-id-prefix nightly_p2 \
  --rnn-manifest data/processed/runs/pipeline_2/feature_manifest.csv \
  --ml-manifest data/processed/runs/pipeline_2/feature_manifest.csv \
  --device cuda
```

Nếu muốn kèm CNN:

```bash
./scripts/tmux_train_all.sh start \
  --session nightly_all_cnn \
  --run-id-prefix nightly_full \
  --with-cnn \
  --cnn-manifest data/processed/cnn_frame_manifest.csv
```

Kết quả tổng hợp:
- `checkpoints/runs/train_all_<prefix>_<timestamp>/train_all_summary.json`
- history:
  - `checkpoints/reports/train_all_history.jsonl`

## 7) Quản lý tmux

Ví dụ với script tổng:

```bash
./scripts/tmux_train_all.sh status --session nightly_all
./scripts/tmux_train_all.sh logs --session nightly_all
./scripts/tmux_train_all.sh attach --session nightly_all
./scripts/tmux_train_all.sh stop --session nightly_all
```

Tương tự cho `tmux_train_eval.sh rnn|ml|cnn`.

## 8) Smoke test script trước khi chạy thật

```bash
./scripts/tmux_train_eval.sh rnn start --example --session ex_rnn_te
./scripts/tmux_train_all.sh start --example --session ex_all
```

## 9) Tải lại raw dataset khi cần re-extract

```bash
./scripts/data/download_daisee.sh
```

Dataset raw sẽ vào đúng vị trí:
- `data/raw/daisee/DAiSEE/`
