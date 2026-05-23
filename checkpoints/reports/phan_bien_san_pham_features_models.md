# Ghi chú phản biện sản phẩm: Feature Extraction và Kiến trúc Model

## 1) Tổng quan pipeline

Hệ thống hiện có 3 nhánh mô hình chính chạy song song theo pipeline:
- `RNN family` (GRU, GRU basic, TCN, Tiny Transformer) dùng chuỗi đặc trưng theo thời gian từ landmark khuôn mặt.
- `ML tree baseline` (XGBoost/LightGBM) dùng feature engineering trên chuỗi (`basic` hoặc `tsfresh-like`).
- `CNN frame-based` (MobileNetV3/EfficientNet/TinyCNN) dùng frame ảnh sampling từ video.

Nguồn code chính:
- `src/engagement_daisee/rnn/extract_features.py`
- `src/engagement_daisee/rnn/models/*.py`
- `src/engagement_daisee/ml/train.py`
- `src/engagement_daisee/cnn/extract_frames.py`
- `src/engagement_daisee/cnn/model.py`

## 2) Extract feature cho nhánh RNN/ML (chuỗi thời gian)

### 2.1 Landmark và đặc trưng mức frame

Extractor dùng MediaPipe FaceMesh / FaceLandmarker để lấy landmark theo từng frame.

Các feature gốc mỗi frame gồm:
- Eye Aspect Ratio trái/phải (`EAR left`, `EAR right`).
- Mouth Aspect Ratio (`MAR`).
- Head/face geometry proxy:
  - Với `base`: thêm `pitch, yaw, roll`.
  - Với `enhanced`: thêm đầy đủ 8 thông số geometry (`pitch, yaw, roll, face_width, face_height, mouth_width, eye_mouth_distance, nose_chin_distance`) + các vector tương đối mắt-miệng-mũi.
- Tọa độ landmark 3D (`x,y,z`) theo danh sách landmark chọn sẵn.

### 2.2 Feature set và số chiều

Từ code `_frame_feature_dim`:
- `base`: `3 + 3 + 8*3 = 30` chiều/frame.
- `enhanced`: `3 + 8 + 6 + 25*3 = 92` chiều/frame.

### 2.3 Window hóa chuỗi và temporal enrichment

Video được chia chunk theo `SEQUENCE_LENGTH=30` frame.
Mỗi chunk sau đó được enrich thành:
- `padded_chunk` (feature gốc),
- `velocity` (sai phân bậc 1 theo thời gian),
- `window_std` lặp theo thời gian.

=> Feature cuối mỗi timestep = `3 * frame_dim`.
- `base` => `90` chiều/timestep.
- `enhanced` => `276` chiều/timestep.

Manifest lưu các trường quan trọng: `feature_path, video_id, label, split, feature_set, feature_dim`.

### 2.4 Điểm phản biện mạnh cho phần feature

Điểm mạnh:
- Chi phí thấp, chạy CPU/GPU nhẹ hơn nhiều so với video backbone nặng.
- Có tín hiệu động học (velocity, std) chứ không chỉ landmark tĩnh.
- Dễ giải thích: EAR/MAR/head pose gắn trực tiếp với hành vi chú ý/mất tập trung.

Điểm yếu/rủi ro:
- Phụ thuộc chất lượng detect mặt; frame không detect được bị zero-feature.
- Landmark hand-crafted có thể bỏ sót ngữ cảnh ngoài mặt (tay, tư thế thân, môi trường).
- Nếu domain shift (camera góc khác, ánh sáng xấu), robustness có thể giảm.

## 3) Kiến trúc model nhánh RNN

Factory model: `src/engagement_daisee/rnn/models/builder.py`.

### 3.1 `gru` (mặc định chính)

`EngagementGRU` gồm:
- Tiền xử lý: `LayerNorm -> Linear(input->hidden) -> LayerNorm -> ReLU -> Dropout`.
- `BiGRU` nhiều lớp (`num_layers`, mặc định 2).
- `TemporalAttention` (learned attention pooling theo thời gian).
- Head: `Linear(2h->h) -> LayerNorm -> ReLU -> Dropout -> Linear(h->1)`.

Ưu điểm phản biện:
- Cân bằng tốt giữa hiệu năng và chi phí.
- Attention giúp diễn giải frame/time-step quan trọng.

### 3.2 `gru_basic`

`BasicGRUClassifier`:
- GRU 1 chiều (không bidirectional), lấy hidden cuối cùng, dropout + linear.

Vai trò:
- Baseline rõ ràng để chứng minh lợi ích của attention + biGRU.

### 3.3 `tcn`

`EngagementTCN`:
- Input projection `LayerNorm + Linear + ReLU`.
- Stack `TemporalConvBlock` residual với dilation tăng theo lũy thừa 2 (`1,2,4,...`).
- Mỗi block: `Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> Dropout + residual`.
- Head dùng `avg pooling + max pooling` concat rồi MLP ra logit.

Ưu điểm:
- Song song hóa tốt hơn RNN, bắt phụ thuộc dài bằng dilation.

### 3.4 `tiny_transformer`

`EngagementTinyTransformer`:
- Input projection + learned positional embedding.
- `TransformerEncoder` (GELU, norm-first, multi-head attention).
- Attention pooling tuyến tính (`attn_pool`) rồi classifier `LayerNorm + Dropout + Linear`.

Ưu điểm:
- Mạnh về modeling long-range dependency.

Rủi ro:
- Nhạy với dữ liệu/regularization; dễ overfit hơn model đơn giản nếu data không đủ đa dạng.

## 4) Nhánh ML tree baseline

File: `src/engagement_daisee/ml/train.py`.

### 4.1 Feature engineering

Hai mode:
- `basic`: mean/std/min/max + first/last/delta + seq_len.
- `tsfresh-like`: thêm slope, energy, IQR, quantile, skewness, kurtosis, abs sum change, second diff, peak rate, zero-crossing, autocorr lag1, low-frequency ratio FFT.

### 4.2 Pipeline train

- Standardize theo train-set (`mean/std`).
- Optional giảm chiều: `none / pca / svd`.
- Optional oversample train (`random`).
- Backend `xgboost` hoặc `lightgbm`.
- Chọn `best_iteration` và `threshold` theo objective validation (`accuracy/balanced_accuracy/f1_pos/f2_pos`).

Thông điệp phản biện:
- Đây là baseline mạnh, dễ deploy CPU, giúp chứng minh hiệu quả thật sự của deep model so với mô hình bảng.

## 5) Nhánh CNN frame-based

### 5.1 Extract frame

File: `src/engagement_daisee/cnn/extract_frames.py`.

- Uniform sample `frames_per_video` (mặc định 8) từ toàn video.
- Resize vuông (`frame_size`, mặc định 112 khi extract; train có thể resize lại).
- Lưu manifest theo `split/video/frame_order`.

### 5.2 Model

File: `src/engagement_daisee/cnn/model.py`.

Hỗ trợ:
- `tinycnn` tự thiết kế nhẹ.
- `mobilenet_v3_small` (phổ biến cho edge).
- `efficientnet_b0`.

Có thể `pretrained` và `freeze_backbone`.

## 6) Bằng chứng thực nghiệm gần nhất (để phản biện)

Nguồn: `checkpoints/runs/train_all_phase12_20260523_011612/train_all_summary.json`.

Mốc thời gian run: ngày `2026-05-23` (UTC).

Một số kết quả test-level theo `video aggregation`:
- `ML (XGBoost + tsfresh-like)`: Accuracy `0.6132`, Balanced Accuracy `0.5908`, F1_pos `0.6995`, Recall_pos `0.6486`.
- `RNN TCN`: Accuracy `0.6418`, Balanced Accuracy `0.5428`, F1_pos `0.7556`, Recall_pos `0.7981`.
- `RNN Tiny Transformer`: Accuracy `0.5465`, Balanced Accuracy `0.5540`, F1_pos `0.6207`.
- `CNN MobileNetV3`: Accuracy `0.6867`, nhưng Balanced Accuracy chỉ `0.5050` do Recall_neg rất thấp (`0.0366`) và Recall_pos rất cao (`0.9733`) => thiên lệch dự đoán dương.

Luận điểm phản biện gợi ý:
- Nếu ưu tiên không bỏ sót mẫu positive: `TCN/CNN` có recall dương cao.
- Nếu cần cân bằng hai lớp tốt hơn: `ML tsfresh-like` ổn định hơn trên balanced accuracy.
- Vì vậy hệ thống nên được trình bày theo hướng `multi-model tradeoff`, không chỉ 1 chỉ số accuracy.

## 7) Hạn chế hiện tại và hướng cải tiến đề xuất

Hạn chế:
- Nhãn nhị phân có thể gây mất thông tin mức độ engagement chi tiết.
- CNN frame sampling có thể bỏ lỡ đoạn quan trọng vì chỉ lấy 8 frame/video.
- RNN landmark chưa khai thác thông tin ngoài vùng mặt.

Hướng cải tiến:
- Late fusion `RNN landmark + CNN frame` để lấy cả tín hiệu hình học và appearance.
- Calibrate threshold theo mục tiêu nghiệp vụ (ưu tiên recall hoặc precision).
- Thêm kiểm thử domain shift (thiết bị khác, ánh sáng khác).
- Nếu cần giải thích: thêm attribution theo timestep (attention map) và feature importance cho XGBoost.

---

## Phụ lục: các file chính để đối chiếu nhanh

- `src/engagement_daisee/rnn/extract_features.py`
- `src/engagement_daisee/rnn/models/gru.py`
- `src/engagement_daisee/rnn/models/tcn.py`
- `src/engagement_daisee/rnn/models/transformer.py`
- `src/engagement_daisee/rnn/models/attention.py`
- `src/engagement_daisee/rnn/models/builder.py`
- `src/engagement_daisee/ml/train.py`
- `src/engagement_daisee/cnn/extract_frames.py`
- `src/engagement_daisee/cnn/model.py`
- `checkpoints/runs/train_all_phase12_20260523_011612/train_all_summary.json`
