# Engagement DAiSEE - Triple XGB 504-feature pipeline

Repo này đã được rút gọn để chỉ giữ pipeline cần thiết cho product:

```text
raw video -> MediaPipe 504 feature windows -> tsfresh-like tabular features -> Triple XGBoost fusion
```

Product hiện tại:

| Hạng mục | Giá trị |
| :--- | :--- |
| Model | `triple_xgb_depth_robust_target_band_product` |
| Input | `.npy` window shape `(30, 504)` |
| Manifest | `data/processed/final_feature_manifest.csv` |
| Accuracy | `76.85%` |
| Balanced Accuracy | `83.20%` |
| F1 macro | `76.91%` |
| Model-side latency mean | `24.80 ms` |
| Raw-video E2E mean | `4.70 s` |

Mục tiêu reproduce là Triple XGB có accuracy trong khoảng `75-77%` và `balanced_accuracy > 75%`.

## Data và model trên Hugging Face

Repo HF:

```text
Hnug/daisee-processed
```

Trang web:

```text
https://huggingface.co/datasets/Hnug/daisee-processed
```

Các artifact cần thiết để tái hiện pipeline 504-feature -> Triple XGB:

```text
data/raw/daisee_raw_videos.zip
data/processed/final_feature_manifest.csv
data/processed/feature_manifest.csv
data/processed/engagement_only_labels.csv
data/processed/runs/triple_xgb_504_features.zip
checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip
checkpoints/runs/triple_xgb_depth_robust_maxacc_product.zip
```

Đường dẫn trực tiếp:

| Artifact | URL |
| :--- | :--- |
| Raw video zip | `https://huggingface.co/datasets/Hnug/daisee-processed/resolve/main/data/raw/daisee_raw_videos.zip` |
| Feature 504 zip | `https://huggingface.co/datasets/Hnug/daisee-processed/resolve/main/data/processed/runs/triple_xgb_504_features.zip` |
| Final manifest | `https://huggingface.co/datasets/Hnug/daisee-processed/resolve/main/data/processed/final_feature_manifest.csv` |
| Feature manifest mirror | `https://huggingface.co/datasets/Hnug/daisee-processed/resolve/main/data/processed/feature_manifest.csv` |
| Engagement labels | `https://huggingface.co/datasets/Hnug/daisee-processed/resolve/main/data/processed/engagement_only_labels.csv` |
| Product model | `https://huggingface.co/datasets/Hnug/daisee-processed/resolve/main/checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip` |
| Max-accuracy model | `https://huggingface.co/datasets/Hnug/daisee-processed/resolve/main/checkpoints/runs/triple_xgb_depth_robust_maxacc_product.zip` |

Ý nghĩa nhanh:

| Artifact | Vai trò |
| :--- | :--- |
| `data/raw/daisee_raw_videos.zip` | Raw DAiSEE video source, dùng khi cần extract lại feature từ đầu |
| `data/processed/final_feature_manifest.csv` | Manifest chính cho training/evaluation product |
| `data/processed/feature_manifest.csv` | Bản mirror/compat của manifest chính |
| `data/processed/engagement_only_labels.csv` | Label 4-class đã chuẩn hóa từ DAiSEE engagement |
| `data/processed/runs/triple_xgb_504_features.zip` | Feature `.npy` đã extract sẵn, dùng để train lại nhanh mà không cần raw video |
| `checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip` | Product model, target accuracy 75-77%, balanced accuracy >75% |
| `checkpoints/runs/triple_xgb_depth_robust_maxacc_product.zip` | Model so sánh, ưu tiên accuracy cao |

Tất cả path trong manifest dùng đường dẫn tương đối. Sau khi unzip đúng vị trí repo, các dòng manifest sẽ trỏ tới:

```text
data/raw/daisee/DAiSEE/DataSet/...
data/processed/runs/triple_xgb_504_features/.../*.npy
```

## Kéo data/model từ Hugging Face

Cài công cụ HF nếu máy mới chưa có:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -U huggingface_hub
```

Nếu repo private/gated, đăng nhập trước:

```bash
huggingface-cli login
```

Tải các file nhỏ:

```bash
huggingface-cli download Hnug/daisee-processed \
  data/processed/final_feature_manifest.csv \
  data/processed/feature_manifest.csv \
  data/processed/engagement_only_labels.csv \
  --repo-type dataset \
  --local-dir .
```

Tải feature `.npy` đã zip sẵn:

```bash
huggingface-cli download Hnug/daisee-processed \
  data/processed/runs/triple_xgb_504_features.zip \
  --repo-type dataset \
  --local-dir .
```

Giải nén feature về đúng vị trí manifest đang trỏ:

```bash
mkdir -p data/processed/runs/triple_xgb_504_features
unzip -q data/processed/runs/triple_xgb_504_features.zip \
  -d data/processed/runs/triple_xgb_504_features/
```

Tải raw video nếu cần extract lại từ đầu:

```bash
huggingface-cli download Hnug/daisee-processed \
  data/raw/daisee_raw_videos.zip \
  --repo-type dataset \
  --local-dir .
```

Giải nén raw video về đúng vị trí manifest/extractor đang dùng:

```bash
mkdir -p data/raw/daisee/DAiSEE
unzip -q data/raw/daisee_raw_videos.zip -d data/raw/daisee/DAiSEE/
```

Tải product model:

```bash
huggingface-cli download Hnug/daisee-processed \
  checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip \
  --repo-type dataset \
  --local-dir .
```

Giải nén product model:

```bash
mkdir -p checkpoints/runs/triple_xgb_depth_robust_target_band_product
unzip -q checkpoints/runs/triple_xgb_depth_robust_target_band_product.zip \
  -d checkpoints/runs/triple_xgb_depth_robust_target_band_product/
```

Kiểm tra nhanh manifest và feature sau khi tải:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/final_feature_manifest.csv")
row = df.iloc[0]
arr = np.load(row["feature_path"], mmap_mode="r")
print("rows:", len(df))
print("sample video exists:", Path(row["video_path"]).exists(), row["video_path"])
print("sample feature exists:", Path(row["feature_path"]).exists(), row["feature_path"])
print("sample feature shape:", arr.shape, arr.dtype)
PY
```

Kỳ vọng:

```text
rows: 70467
sample video exists: True
sample feature exists: True
sample feature shape: (30, 504) float32
```

Nếu chỉ muốn train/reproduce mà không extract lại, raw video không bắt buộc; chỉ cần manifest, label và `triple_xgb_504_features.zip`.

## Reproducibility checklist

Mức tái hiện thực tế:

| Mục tiêu | Mức đảm bảo |
| :--- | :--- |
| Kéo lại đúng raw video, manifest, label, feature `.npy`, product model | Cao, vì artifact đã nằm trên HF theo đường dẫn cố định |
| Chạy inference bằng product model đã upload | Cao, nếu cài đúng dependency trong `requirements.txt` |
| Re-train lại Triple XGB từ manifest để đạt khoảng `75-77%` accuracy và `balanced_accuracy >75%` | Cao, nhưng có thể lệch nhỏ theo version XGBoost/CPU/threading |
| Re-extract feature từ raw video | Chậm hơn và phụ thuộc MediaPipe/OpenCV; nên dùng feature zip nếu mục tiêu là reproduce model |

Khuyến nghị để máy khác ít lệch nhất:

```bash
python3 --version   # nên dùng Python 3.10-3.12
pip install -r requirements.txt
```

Sau khi tải và unzip, kiểm tra nhanh:

```bash
bash scripts/reproduce_triple_xgb.sh
```

Expected product/reference:

```text
model: triple_xgb_depth_robust_target_band_product
accuracy: ~76.85%
balanced_accuracy: ~83.20%
f1_macro: ~76.91%
```

Nếu cần kết quả gần như cố định tuyệt đối, dùng trực tiếp zip model product đã upload thay vì train lại từ đầu.

## Pipeline tái hiện

### Chạy lại từ manifest đã có

Không cần extract lại nếu đã có `data/processed/runs/triple_xgb_504_features`.

```bash
bash scripts/reproduce_triple_xgb.sh
```

Kết quả sẽ ghi vào:

```text
checkpoints/runs/triple_xgb_target_band_repro/
checkpoints/reports/triple_xgb_repro_summary.json
```

### Extract lại 504 feature nếu cần

Chỉ chạy khi muốn tạo lại `.npy` từ raw video:

```bash
bash scripts/extract_504_features.sh \
  --labels-csv data/processed/engagement_only_labels.csv \
  --raw-video-dir data/raw/daisee/DAiSEE/DataSet \
  --output-dir data/processed/runs/triple_xgb_504_features
```

Extractor tạo mỗi window có shape `(30, 504)`:

```text
504 = 168 raw landmark-depth features
    + 168 velocity features
    + 168 window-std features
```

### Notebook pipeline

Notebook gọn để kiểm tra từng khối:

```text
notebooks/triple_xgb_pipeline.ipynb
```

Các khối chính:

1. đọc manifest;
2. kiểm tra `.npy` shape `(30, 504)`;
3. build tabular feature cho XGBoost;
4. reproduce Triple XGB;
5. xem metric/HF artifact.

## Báo cáo

```text
checkpoints/reports/GUIDE.md
checkpoints/reports/bao_cao_ket_qua_huan_luyen_models.md
```
