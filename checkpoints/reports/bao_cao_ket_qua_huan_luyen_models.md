# Báo cáo kết quả huấn luyện các model Engagement Detection

> Báo cáo tổng hợp từ các checkpoint/log trong repo. Các chỉ số được trình bày theo test set, ưu tiên `Balanced Accuracy` vì bài toán có lệch lớp.

**Ngày lập báo cáo:** 2026-06-08

## 1. Tóm tắt nhanh

| Hạng mục | Kết quả |
| :--- | :--- |
| Model cân bằng tốt nhất | gru (train_all_rnn_gru) - **Balanced Accuracy 77.35%**, Accuracy 71.97% |
| Model có Accuracy/F1 cao nhất | late-fusion ensemble GRU + TCN + XGBoost - **Accuracy 75.95%**, **F1 Macro 73.86%**, Balanced Accuracy 76.48% |
| End-to-end latency đã đo | **9.53 s/video 300 frames** nhanh nhất trong benchmark mới, gồm MediaPipe FaceMesh extraction + windowing + model predict |
| Model-side latency nhanh nhất | xgboost_final - **0.48 ms/input**, chỉ tính model predict khi feature đã có sẵn |
| Nhận xét chính | Late-fusion ensemble là lựa chọn trình bày mạnh nhất theo Accuracy/F1 và gần mốc SOTA tham khảo; GRU vẫn là model đơn mạnh nhất về Balanced Accuracy; TCN ổn định; XGBoost nhẹ và hữu ích trong ensemble. |
| Run nên dùng để báo cáo/chọn checkpoint | Nếu cần model đơn: `final_rnn_temporal_models_20260529` với `rnn_gru`; nếu cần kết quả Accuracy/F1 tốt nhất: late-fusion ensemble trong `late_fusion_gru_tcn_xgb_report.json` |

## 2. Leaderboard các mốc model đã chọn

| Rank | Milestone | Model | Balanced Acc | Accuracy | F1 Macro | Threshold | Checkpoint |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 1 | final | gru | **77.35%** | 71.97% | 71.22% | 0.63 | [engagement_gru.pt](../runs/final_rnn_temporal_models_20260529/rnn_gru/engagement_gru.pt) |
| 2 | late_fusion_20260608 | GRU + TCN + XGBoost ensemble | 76.48% | **75.95%** | **73.86%** | 0.54 | [late_fusion_gru_tcn_xgb_report.json](late_fusion_gru_tcn_xgb_report.json) |
| 3 | evolved_20260608 | residual_bigru_attn | 73.76% | 72.03% | 70.30% | 0.55 | [engagement_residual_bigru_attn.pt](../runs/evolved_rnn_hybrid_models_20260608/rnn_residual_bigru_attn/engagement_residual_bigru_attn.pt) |
| 4 | final | tcn | 72.52% | 69.67% | 68.30% | 0.61 | [engagement_tcn.pt](../runs/final_rnn_temporal_models_20260529/rnn_tcn/engagement_tcn.pt) |
| 5 | final | gru_basic | 70.66% | 64.18% | 63.77% | 0.61 | [engagement_gru_basic.pt](../runs/final_rnn_temporal_models_20260529/rnn_gru_basic/engagement_gru_basic.pt) |
| 6 | evolved_20260608 | cnn_gru_fusion | 70.33% | 67.49% | 66.14% | 0.64 | [engagement_cnn_gru_fusion.pt](../runs/evolved_rnn_hybrid_models_20260608/rnn_cnn_gru_fusion/engagement_cnn_gru_fusion.pt) |
| 7 | final | xgboost | 69.34% | 70.52% | - | 0.49 | [engagement_xgb.json](../runs/final_xgboost_tsfresh_20260529/engagement_xgb.json) |
| 8 | final | tiny_transformer | 64.74% | 67.54% | 63.75% | 0.55 | [engagement_tiny_transformer.pt](../runs/final_rnn_temporal_models_20260529/rnn_tiny_transformer/engagement_tiny_transformer.pt) |
| 9 | phase12 | xgboost | 59.08% | 61.32% | - | 0.49 | [engagement_xgb.json](../runs/trainml_phase12_ml/engagement_xgb.json) |
| 10 | openface_pca160 | xgboost | 58.92% | 61.29% | - | 0.47 | [engagement_xgb.json](../runs/trainml_openface709_safe_train_20260525_ml_pca160_tsfresh/engagement_xgb.json) |
| 11 | phase34_nocnn | xgboost | 58.37% | 62.89% | - | 0.49 | [engagement_xgb.json](../runs/trainml_phase34_nocnn_train_ml/engagement_xgb.json) |
| 12 | phase34_nocnn | tiny_transformer | 58.21% | 61.83% | 57.50% | 0.21 | [engagement_tiny_transformer.pt](../runs/train_all_phase34_nocnn_train_20260523_163113/rnn_tiny_transformer/engagement_tiny_transformer.pt) |
| 13 | phase34_nocnn | tcn | 57.95% | 51.51% | 51.34% | 0.46 | [engagement_tcn.pt](../runs/train_all_phase34_nocnn_train_20260523_163113/rnn_tcn/engagement_tcn.pt) |
| 14 | phase34_nocnn | gru | 57.29% | 56.28% | 54.53% | 0.29 | [engagement_gru.pt](../runs/train_all_phase34_nocnn_train_20260523_163113/rnn_gru/engagement_gru.pt) |
| 15 | phase12 | gru_basic | 55.99% | 63.79% | 56.12% | 0.17 | [engagement_gru_basic.pt](../runs/train_all_phase12_20260523_011612/rnn_gru_basic/engagement_gru_basic.pt) |
| 16 | baseline_bilstm | bilstm | 55.87% | 50.67% | 50.33% | 0.39 | [engagement_bilstm.pt](../runs/rnn_train_eval_rnn_bilstm_copur/engagement_bilstm.pt) |
| 17 | phase34_nocnn | gru_basic | 55.83% | 65.70% | 55.92% | 0.14 | [engagement_gru_basic.pt](../runs/train_all_phase34_nocnn_train_20260523_163113/rnn_gru_basic/engagement_gru_basic.pt) |
| 18 | phase12 | tiny_transformer | 55.40% | 54.65% | 52.85% | 0.31 | [engagement_tiny_transformer.pt](../runs/train_all_phase12_20260523_011612/rnn_tiny_transformer/engagement_tiny_transformer.pt) |
| 19 | phase12 | gru | 55.01% | 50.06% | 49.69% | 0.32 | [engagement_gru.pt](../runs/train_all_phase12_20260523_011612/rnn_gru/engagement_gru.pt) |
| 20 | phase12 | tcn | 54.28% | 64.18% | 54.26% | 0.15 | [engagement_tcn.pt](../runs/train_all_phase12_20260523_011612/rnn_tcn/engagement_tcn.pt) |
| 21 | cnn_best | efficientnet_b0 | 53.09% | 70.91% | 47.80% | 0.19 | [engagement_cnn.pt](../runs/train_all_phase34_20260523_030236/cnn/engagement_cnn.pt) |
| 22 | phase12 | mobilenet_v3_small | 50.50% | 68.67% | 43.92% | 0.11 | [engagement_cnn.pt](../runs/train_all_phase12_20260523_011612/cnn/engagement_cnn.pt) |

## 3. Kết quả final run

Bảng này lấy từ milestone `final` trong `selected_model_performance.csv`, phù hợp để đưa vào phần kết quả chính của báo cáo.

| Model | Nhóm | Val Balanced Acc | Test Balanced Acc | Test Accuracy | Test F1 Macro | E2E latency/video | Model-side latency | Test Loss | Threshold | Checkpoint |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| gru | RNN | 61.00% | **77.35%** | 71.97% | 71.22% | 9.55 s | 23.95 ms | **0.341** | 0.63 | [engagement_gru.pt](../runs/final_rnn_temporal_models_20260529/rnn_gru/engagement_gru.pt) |
| GRU + TCN + XGBoost | Ensemble | **65.30%** | 76.48% | **75.95%** | **73.86%** | 9.78 s | 16.05 ms | 0.522 | 0.54 | [late_fusion_gru_tcn_xgb_report.json](late_fusion_gru_tcn_xgb_report.json) |
| tcn | RNN | 60.53% | 72.52% | 69.67% | 68.30% | 9.54 s | 16.24 ms | 0.361 | 0.61 | [engagement_tcn.pt](../runs/final_rnn_temporal_models_20260529/rnn_tcn/engagement_tcn.pt) |
| gru_basic | RNN | 59.41% | 70.66% | 64.18% | 63.77% | - | - | 0.401 | 0.61 | [engagement_gru_basic.pt](../runs/final_rnn_temporal_models_20260529/rnn_gru_basic/engagement_gru_basic.pt) |
| residual_bigru_attn | Evolved RNN | 61.15% | 73.76% | 72.03% | 70.30% | 9.58 s | 9.24 ms | 0.835 | 0.55 | [engagement_residual_bigru_attn.pt](../runs/evolved_rnn_hybrid_models_20260608/rnn_residual_bigru_attn/engagement_residual_bigru_attn.pt) |
| cnn_gru_fusion | Evolved RNN | 59.78% | 70.33% | 67.49% | 66.14% | 9.57 s | 8.96 ms | 0.708 | 0.64 | [engagement_cnn_gru_fusion.pt](../runs/evolved_rnn_hybrid_models_20260608/rnn_cnn_gru_fusion/engagement_cnn_gru_fusion.pt) |
| xgboost | ML | 61.36% | 69.34% | 70.52% | - | **9.53 s** | **0.48 ms** | - | 0.49 | [engagement_xgb.json](../runs/final_xgboost_tsfresh_20260529/engagement_xgb.json) |
| tiny_transformer | RNN | 58.51% | 64.74% | 67.54% | 63.75% | - | - | 0.417 | 0.55 | [engagement_tiny_transformer.pt](../runs/final_rnn_temporal_models_20260529/rnn_tiny_transformer/engagement_tiny_transformer.pt) |

Late-fusion dùng cùng feature pipeline MediaPipe như các model final đã đo. Trọng số fusion được chọn trên validation: `GRU=0.30`, `TCN=0.30`, `XGBoost=0.40`; test giữ làm tập đánh giá cuối.

## 4. Latency benchmark

Latency được đo bằng script `scripts/data/benchmark_end_to_end_latency.py` trên CPU, `2` threads, từ video thô đến dự đoán cuối: đọc frame video, chạy MediaPipe FaceMesh, tạo feature/window, rồi gọi model predict. Mẫu đo là video DAiSEE test `5000441001.avi`, xử lý `300` frames và tạo `10` windows. Đây là con số dùng để so sánh latency trong báo cáo vì cùng tính toàn bộ pipeline.

| Pipeline | Feature backend | Frames | Windows | E2E mean | E2E median | E2E P95 | Throughput |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost_final | MediaPipe FaceMesh | 300 | 10 | **9.53 s** | **9.57 s** | **10.31 s** | **0.105 video/s** |
| rnn_tcn_final_torchscript | MediaPipe FaceMesh | 300 | 10 | 9.54 s | 9.57 s | 10.32 s | 0.105 video/s |
| rnn_gru_final_int8 | MediaPipe FaceMesh | 300 | 10 | 9.55 s | 9.58 s | 10.34 s | 0.105 video/s |
| rnn_cnn_gru_fusion_torchscript | MediaPipe FaceMesh | 300 | 10 | 9.57 s | 9.60 s | 10.37 s | 0.104 video/s |
| rnn_residual_bigru_attn_torchscript | MediaPipe FaceMesh | 300 | 10 | 9.58 s | 9.61 s | 10.36 s | 0.104 video/s |
| late_fusion_gru_tcn_xgb | MediaPipe FaceMesh | 300 | 10 | 9.78 s | 9.69 s | 11.17 s | 0.102 video/s |

**Kết luận latency:** các pipeline feature-based gần như bằng nhau vì cùng dùng MediaPipe FaceMesh để extract feature trước khi dự đoán. XGBoost có E2E thấp nhất trong lần đo mới, nhưng chênh lệch chỉ khoảng vài chục mili-giây so với RNN đơn; late-fusion chậm hơn do chạy ba predictor. `Model-side latency` vẫn hữu ích để hiểu chi phí riêng của model sau khi feature đã có sẵn, nhưng không thay thế cho E2E latency khi so sánh với paper hoặc pipeline khác.

### Model-side latency tham khảo

Bảng dưới chỉ tính latency model-side trên input đã có sẵn, đo bằng `scripts/data/benchmark_model_latency.py` trên CPU, `2` threads, `30` warmup iterations và `300` benchmark iterations. Không dùng bảng này để so sánh “táo với táo” nếu pipeline phải extract feature từ video.

| Model | Nhóm | Biến thể | Input | Mean latency | Median | P95 | Nhận xét |
| :--- | :---: | :--- | :--- | ---: | ---: | ---: | :--- |
| xgboost_final | ML | xgboost | tabular:1x2161 | **0.48 ms** | **0.48 ms** | **0.53 ms** | Predict nhẹ nhất khi feature đã có sẵn. |
| rnn_cnn_gru_fusion | RNN | int8_dynamic | sequence:1x30x90 | 8.96 ms | 8.90 ms | 9.48 ms | RNN mới nhanh nhất theo mean latency. |
| rnn_residual_bigru_attn | RNN | int8_dynamic | sequence:1x30x90 | 9.24 ms | 9.14 ms | 9.76 ms | Model đơn mới tốt nhất về Accuracy, latency vẫn nhẹ. |
| rnn_cnn_gru_fusion | RNN | torchscript_fp32 | sequence:1x30x90 | 10.78 ms | 10.74 ms | 11.40 ms | Bản TorchScript của CNN-GRU fusion. |
| rnn_cnn_gru_fusion | RNN | eager_fp32 | sequence:1x30x90 | 11.16 ms | 11.07 ms | 12.08 ms | Eager FP32 của CNN-GRU fusion. |
| rnn_residual_bigru_attn | RNN | torchscript_fp32 | sequence:1x30x90 | 14.85 ms | 11.06 ms | 25.08 ms | TorchScript nhanh hơn eager cho residual BiGRU attention. |
| late_fusion_gru_tcn_xgb | Ensemble | gru_int8+tcn_torchscript+xgboost | sequence:1x30x90+tabular:1x2161 | 16.05 ms | 15.62 ms | 18.87 ms | Ensemble chạy ba predictor rồi fuse xác suất. |
| rnn_tcn_final | RNN | int8_dynamic | sequence:1x30x90 | 16.24 ms | 12.99 ms | 31.95 ms | TCN int8 nhanh nhất trong các biến thể TCN ở lần đo mới. |
| rnn_tcn_final | RNN | eager_fp32 | sequence:1x30x90 | 19.67 ms | 16.40 ms | 33.65 ms | Eager TCN. |
| rnn_tcn_final | RNN | torchscript_fp32 | sequence:1x30x90 | 23.27 ms | 14.55 ms | 100.04 ms | TorchScript TCN có outlier P95 cao trong lần đo mới. |
| rnn_gru_final | RNN | eager_fp32 | sequence:1x30x90 | 23.95 ms | 23.81 ms | 29.03 ms | GRU final bản eager ổn định nhất theo mean ở lần đo mới. |
| rnn_gru_final | RNN | torchscript_fp32 | sequence:1x30x90 | 27.43 ms | 23.08 ms | 27.83 ms | GRU TorchScript. |
| rnn_gru_final | RNN | int8_dynamic | sequence:1x30x90 | 30.88 ms | 20.14 ms | 38.50 ms | GRU int8 có outlier mean trong lần đo mới. |
| rnn_residual_bigru_attn | RNN | eager_fp32 | sequence:1x30x90 | 33.21 ms | 21.03 ms | 84.24 ms | Eager residual BiGRU attention có outlier lớn. |
| cnn_efficientnet_b0 | CNN | eager_fp32 | image:1x3x224x224 | 54.12 ms | 49.89 ms | 80.01 ms | Chưa gồm decode/extract frame. |
| santoni_pca_svd_cnn_arch | Paper CNN | eager_fp32_untrained_arch | image:1x1x300x300 | 120.06 ms | 119.01 ms | 127.19 ms | Kiến trúc paper tái tạo; chưa gồm OpenFace/PCA/SVD/SMOTE/decode video. |

File kết quả benchmark:
- `checkpoints/reports/end_to_end_latency_benchmark.json`
- `checkpoints/reports/end_to_end_latency_benchmark.csv`
- `checkpoints/reports/latency_benchmark_models.json`
- `checkpoints/reports/latency_benchmark_models.csv`

## 5. Map kết quả với latency

Bảng dưới ghép trực tiếp kết quả test với latency end-to-end. Với các model của project, latency gồm MediaPipe feature extraction + windowing + model predict trên video 300 frames. Với paper/SOTA, paper không công bố latency end-to-end nên không đưa số latency thay thế.

| Model/nguồn | Protocol | Accuracy | Balanced Acc | E2E latency/video | Model-side latency | Điểm đọc nhanh |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **RNN GRU final** | Ours binary, video aggregation | 71.97% | **77.35%** | 9.55 s | 23.95 ms | Model đơn cân bằng tốt nhất; latency E2E ngang các pipeline MediaPipe khác. |
| **Late-fusion GRU + TCN + XGBoost** | Ours binary, video aggregation | **75.95%** | 76.48% | 9.78 s | 16.05 ms | Accuracy/F1 tốt nhất trong các kết quả đã hoàn tất; dùng ba model thành phần sau cùng bước MediaPipe feature extraction. |
| **Residual BiGRU Attention** | Ours binary, video aggregation | 72.03% | 73.76% | 9.58 s | 9.24 ms | Model đơn mới tốt nhất; model-side khá nhẹ sau int8 dynamic. |
| **CNN-GRU Fusion** | Ours binary, video aggregation | 67.49% | 70.33% | 9.57 s | 8.96 ms | RNN mới nhanh nhất theo model-side trong nhóm neural. |
| **RNN TCN final** | Ours binary, video aggregation | 69.67% | 72.52% | 9.54 s | 16.24 ms | Ổn định, nhưng chất lượng thấp hơn GRU và residual BiGRU attention. |
| **XGBoost final** | Ours binary, video aggregation | 70.52% | 69.34% | **9.53 s** | **0.48 ms** | Predict nhẹ nhất, E2E nhanh nhất trong lần đo mới nhưng vẫn bị MediaPipe chi phối. |
| CNN EfficientNet-B0 | Ours binary, frame-based | 70.91% | 53.09% | Chưa đo E2E | 54.12 ms | Chỉ có model-side latency; chưa có phép đo từ video thô đến dự đoán. |
| Santoni SVD-CNN | Paper DAiSEE 4-class | **77.97%** | - | Không công bố | 120.06 ms* | Accuracy paper cao nhất trong bảng; latency chính thức không có. |
| Santoni PCA-CNN | Paper DAiSEE 4-class | 72.88% | - | Không công bố | 120.06 ms* | Cùng kiến trúc CNN với SVD-CNN; khác bước giảm chiều PCA/SVD. |
| PriorNet 2026 | Paper DAiSEE native protocol | 69.06% | - | Không công bố | Không công bố | Không có checkpoint trong repo để đo cùng máy. |
| DTransformer / PANet + STformer 2024 | Paper DAiSEE 4-class | 64.00% | - | Không công bố | Không công bố | Không có checkpoint trong repo để đo cùng máy. |

`*` Latency Santoni là số đo lại bằng class `PaperCNN` tái tạo trong `src/engagement_daisee/ml/reproduce_paper_cnn.py`, input giả lập `1x1x300x300`, CPU 2 threads. Paper Santoni không công bố latency E2E; con số này chỉ đại diện cho forward của kiến trúc tái tạo trên máy hiện tại.

## 6. CNN benchmark

CNN có Accuracy khá cao nhưng Balanced Accuracy thấp hơn do recall negative thấp. Vì vậy CNN nên được xem là baseline appearance hoặc ứng viên fusion, chưa nên dùng độc lập làm model chính.

| Milestone | Run ID | Model | Val Balanced Acc | Test Balanced Acc | Test Accuracy | F1 Macro | E2E latency/video | Model-side latency | Threshold | Checkpoint |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| cnn_best | phase34_cnn | efficientnet_b0 | **64.11%** | **53.09%** | **70.91%** | **47.80%** | Chưa đo | **54.12 ms** | 0.19 | [engagement_cnn.pt](../runs/train_all_phase34_20260523_030236/cnn/engagement_cnn.pt) |
| phase12 | phase12_cnn | mobilenet_v3_small | 57.35% | 50.50% | 68.67% | 43.92% | - | - | 0.11 | [engagement_cnn.pt](../runs/train_all_phase12_20260523_011612/cnn/engagement_cnn.pt) |

## 7. So sánh với SOTA/paper benchmark

Các kết quả SOTA/paper bên dưới chủ yếu báo cáo `4-class accuracy` cho nhãn engagement của DAiSEE, trong khi kết quả của project này đang là bài toán `binary engagement` và có thêm `Balanced Accuracy`. Vì vậy bảng này nên đọc như bối cảnh benchmark, không phải so sánh tuyệt đối cùng protocol.

| Nguồn/model | Protocol báo cáo | Accuracy | Balanced Acc | Ghi chú |
| :--- | :--- | ---: | ---: | :--- |
| **Ours - Late-fusion GRU + TCN + XGBoost** | Binary engagement, video aggregation, test split nội bộ | **75.95%** | 76.48% | Ensemble/hybrid score tốt nhất theo Accuracy/F1; rất phù hợp để trình bày như kết quả chính khi nhấn mạnh hiệu quả tổng thể. |
| **Ours - RNN GRU final** | Binary engagement, video aggregation, test split nội bộ | 71.97% | **77.35%** | Model đơn tốt nhất theo Balanced Accuracy; tối ưu cân bằng hai lớp. |
| **Ours - RNN TCN final** | Binary engagement, video aggregation, test split nội bộ | 69.67% | 72.52% | Ứng viên thứ hai, mạnh về temporal modeling. |
| **Ours - XGBoost final** | Binary engagement, video aggregation, test split nội bộ | 70.52% | 69.34% | Baseline CPU nhẹ, dễ deploy. |
| Santoni et al. 2023 - SVD-CNN | DAiSEE 4-class, OpenFace 709D -> SVD 300D, SMOTE, 80:20 split | **77.97%** | - | Paper báo cáo đây là kết quả cao nhất của SVD-CNN; không thấy latency công bố. |
| Santoni et al. 2023 - PCA-CNN | DAiSEE 4-class, OpenFace 709D -> PCA 300D, SMOTE, 80:20 split | 72.88% | - | Cùng pipeline với SVD-CNN nhưng dùng PCA; paper không công bố latency E2E. |
| PriorNet 2026 | DAiSEE native engagement classification, face-video prior-guided model | 69.06% | - | Preprint; chính tác giả cảnh báo DAiSEE literature không đồng nhất protocol. |
| EfficientNetB7 + LSTM, Selim et al. 2022 | DAiSEE 4-class | 67.48% | - | Một trong các baseline hybrid CNN-temporal mạnh được trích trong Santoni/PriorNet. |
| EfficientNetB7 + Bi-LSTM, Selim et al. 2022 | DAiSEE 4-class | 66.39% | - | Hybrid EfficientNetB7 và Bi-LSTM. |
| EfficientNetB7 + TCN, Selim et al. 2022 | DAiSEE 4-class | 64.67% | - | Hybrid EfficientNetB7 và TCN. |
| DTransformer / PANet + STformer, Su et al. 2024 | DAiSEE 4-class test set | 64.00% | - | Transformer/attention architecture cho learner engagement. |
| ResNet + TCN, Abedi et al. | DAiSEE 4-class | 63.90% | - | Mốc SOTA cũ trong nhiều bảng tổng hợp. |
| DAiSEE original LRCN baseline | DAiSEE baseline benchmarking | 57.90% | - | Mốc baseline ban đầu thường được dùng để đối chiếu. |

**Nhận xét so sánh:** nếu chỉ nhìn Accuracy, `rnn_gru final` của project đạt 71.97%, cao hơn một số benchmark 4-class hiện đại như PriorNet 69.06% và EfficientNetB7+LSTM 67.48%, nhưng thấp hơn mốc Santoni SVD-CNN 77.97%. Tuy nhiên do khác nhị phân/4-class, khác split và khác cách cân bằng dữ liệu, kết luận đúng hơn là: model của project đã đạt mức cạnh tranh trong bối cảnh DAiSEE, còn để tuyên bố vượt SOTA cần chạy lại đúng protocol 4-class hoặc tái huấn luyện SOTA trên cùng binary split.

Nguồn tham khảo chính: [Santoni et al. 2023](https://thesai.org/Downloads/Volume14No3/Paper_71-Convolutional_Neural_Network_Model.pdf), [PriorNet arXiv 2026](https://arxiv.org/abs/2605.03615), [DTransformer 2024](https://doi.org/10.1016/j.aej.2024.06.074).

## 8. Cơ sở khoa học cho feature extraction

Các feature trong project không phải được lấy ngẫu nhiên; chúng tương ứng với các nhóm tín hiệu thị giác đã được dùng nhiều trong drowsiness, fatigue, attention và facial behavior analysis. Một phần nguồn là y khoa/clinical-adjacent, đặc biệt các nghiên cứu về buồn ngủ, obstructive sleep apnea và driver fatigue; một phần là computer vision/HCI.

| Nhóm feature trong project | Ý nghĩa hành vi/sinh lý | Nguồn có thể trích dẫn | Cách dùng trong project |
| :--- | :--- | :--- | :--- |
| `EAR`, trạng thái mắt, blink/eye closure | Mắt nhắm lâu, chớp mắt và eyelid closure liên quan đến buồn ngủ/mất tỉnh táo. | PERCLOS review trên PubMed Central; nghiên cứu OSA dùng tín hiệu visual + EEG; Soukupová & Čech 2016 về EAR từ facial landmarks. | Dùng landmark mắt để tạo `EAR left/right`, velocity và window statistics. |
| `PERCLOS`/eye-closure proxy | PERCLOS là chỉ số lâu đời để đo tỷ lệ thời gian mắt gần/hoàn toàn nhắm, thường dùng trong fatigue/drowsiness monitoring. | Review “PERCLOS-based technologies for detecting drowsiness” và các nghiên cứu driver fatigue/OSA. | Có thể diễn giải các feature EAR theo hướng eye-closure proxy, dù pipeline hiện dùng EAR trực tiếp hơn là PERCLOS chuẩn. |
| `MAR`, mouth opening/yawn proxy | Miệng mở rộng/yawning là dấu hiệu mệt mỏi hoặc giảm tỉnh táo trong nhiều hệ driver monitoring. | Các paper driver fatigue dùng Mouth Aspect Ratio và yawn analysis, ví dụ Scientific Reports/MDPI fatigue driving studies. | Dùng `MAR`, mouth width và mouth/eye geometry làm tín hiệu biểu cảm-mệt mỏi. |
| Head pose: `pitch`, `yaw`, `roll` | Cúi đầu, quay đầu, lệch hướng nhìn có thể phản ánh mất chú ý hoặc thay đổi visual focus. | Nghiên cứu attention/VFOA dùng head pose + gaze; OpenFace 2.0 cung cấp head pose/gaze/facial behavior features. | Dùng pitch/yaw/roll và geometry proxy trong feature set `base/enhanced`. |
| Facial landmarks 2D/3D và geometry | Landmark khuôn mặt là nền tảng để suy ra mắt, miệng, pose, tỷ lệ khuôn mặt và dynamics theo thời gian. | OpenFace/OpenFace 2.0 papers; MediaPipe/landmark literature; các nghiên cứu blink/yawn dùng landmark. | Dùng landmark 3D đã chọn, face width/height, eye-mouth distance, nose-chin distance. |
| Temporal features: velocity, window std | Trạng thái engagement/fatigue không chỉ là ảnh tĩnh; biến đổi theo thời gian quan trọng cho blink, yawn, head nod và expression dynamics. | Các nghiên cứu drowsiness/fatigue thường dùng chuỗi frame/video và thống kê cửa sổ thời gian. | Mỗi chunk được enrich bằng feature gốc + sai phân bậc 1 + thống kê cửa sổ. |

Nguồn feature extraction nên trích dẫn:
- PERCLOS review: https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/
- Visual signals + EEG in obstructive sleep apnea drowsiness detection: https://pmc.ncbi.nlm.nih.gov/articles/PMC11055081/
- Soukupová & Čech, “Eye Blink Detection Using Facial Landmarks”: https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf
- OpenFace 2.0 facial behavior toolkit: https://par.nsf.gov/servlets/purl/10099458
- Driver fatigue with MAR/head posture/facial state: https://www.nature.com/articles/s41598-026-44994-4
- Fatigue driving recognition with eye and mouth aspect ratios: https://www.mdpi.com/2079-9292/11/24/4103
- Attention/VFOA using head pose and gaze: https://www.sciencedirect.com/science/article/pii/S0167865514003067

**Cách viết trong báo cáo:** “Các đặc trưng hình học khuôn mặt như EAR, MAR, head pose và landmark dynamics được lựa chọn dựa trên các nghiên cứu drowsiness/fatigue và visual attention. EAR/PERCLOS đại diện cho trạng thái mắt và buồn ngủ; MAR đại diện cho yawning/mouth opening; head pose/gaze liên quan đến visual focus of attention; các đặc trưng temporal giúp nắm bắt blink/yawn/head movement theo chuỗi thời gian.”

## 9. PCA sweep - model tốt nhất theo từng kích thước feature

Các run PCA ngày `2026-06-01` dùng bộ feature OpenFace/manifest709 sau khi giảm chiều PCA. Bảng dưới chọn model có `Test Balanced Acc` cao nhất trong mỗi kích thước PCA.

| PCA dim | Best model | Nhóm | Val Balanced Acc | Test Balanced Acc | Test Accuracy | Test F1 Macro | Threshold | Checkpoint |
| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 128 | tiny_transformer | RNN | **58.20%** | 51.46% | **49.82%** | 48.54% | 0.50 | [engagement_tiny_transformer.pt](../runs/train_all_pca128_20260601_095609/rnn_tiny_transformer/engagement_tiny_transformer.pt) |
| 160 | gru | RNN | 57.71% | 51.94% | 42.46% | 42.35% | 0.68 | [engagement_gru.pt](../runs/train_all_pca160_20260601_101009/rnn_gru/engagement_gru.pt) |
| 192 | tcn | RNN | 56.75% | 52.17% | 47.51% | 47.15% | 0.69 | [engagement_tcn.pt](../runs/train_all_pca192_20260601_102540/rnn_tcn/engagement_tcn.pt) |
| 224 | tiny_transformer | RNN | 57.49% | 54.75% | 49.85% | **49.47%** | 0.88 | [engagement_tiny_transformer.pt](../runs/train_all_pca224_20260601_104210/rnn_tiny_transformer/engagement_tiny_transformer.pt) |
| 256 | gru_basic | RNN | 57.70% | 54.58% | 44.90% | 44.82% | 0.72 | [engagement_gru_basic.pt](../runs/train_all_pca256_20260601_110040/rnn_gru_basic/engagement_gru_basic.pt) |
| 300 | gru | RNN | 57.11% | 54.26% | 46.95% | 46.92% | 0.82 | [engagement_gru.pt](../runs/train_all_pca300_20260601_112041/rnn_gru/engagement_gru.pt) |
| 384 | tiny_transformer | RNN | 55.20% | **56.98%** | 47.49% | 47.46% | 0.82 | [engagement_tiny_transformer.pt](../runs/train_all_pca384_20260601_114342/rnn_tiny_transformer/engagement_tiny_transformer.pt) |

## 10. PCA sweep riêng cho XGBoost

XGBoost PCA có validation khá ổn định quanh 62-63% Balanced Accuracy, nhưng test Balanced Accuracy thấp hơn. PCA-192 đạt test Balanced Accuracy cao nhất trong nhóm XGBoost PCA; PCA-384 có F1 positive cao nhất nhưng recall negative giảm mạnh.

| PCA dim | Rows | Threshold | Val Balanced Acc | Test Balanced Acc | Test Accuracy | F1 Pos | Recall Neg | Recall Pos | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 128 | 34,284 | 0.47 | 62.40% | 52.94% | 55.34% | 64.76% | 46.75% | 59.13% | [engagement_xgb.json](../runs/trainml_pca128_ml/engagement_xgb.json) |
| 160 | 34,284 | 0.49 | 62.71% | 53.66% | 54.25% | 62.60% | **52.15%** | 55.17% | [engagement_xgb.json](../runs/trainml_pca160_ml/engagement_xgb.json) |
| 192 | 34,284 | 0.47 | **63.03%** | **53.70%** | 56.71% | 66.34% | 45.92% | 61.47% | [engagement_xgb.json](../runs/trainml_pca192_ml/engagement_xgb.json) |
| 224 | 34,284 | 0.49 | 62.49% | 53.08% | 54.06% | 62.69% | 50.55% | 55.61% | [engagement_xgb.json](../runs/trainml_pca224_ml/engagement_xgb.json) |
| 256 | 34,284 | 0.45 | 62.99% | 52.72% | 57.61% | 68.14% | 40.11% | 65.33% | [engagement_xgb.json](../runs/trainml_pca256_ml/engagement_xgb.json) |
| 300 | 34,284 | 0.43 | 62.64% | 51.76% | 57.93% | 69.06% | 35.85% | 67.67% | [engagement_xgb.json](../runs/trainml_pca300_ml/engagement_xgb.json) |
| 384 | 34,284 | 0.43 | 62.40% | 51.38% | **59.11%** | **70.76%** | 31.46% | **71.30%** | [engagement_xgb.json](../runs/trainml_pca384_ml/engagement_xgb.json) |

## 11. Chi tiết tất cả model RNN trong PCA sweep

| PCA dim | Model | Nhóm | Val Balanced Acc | Test Balanced Acc | Test Accuracy | Test F1 Macro | Test Loss |
| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| 128 | gru | RNN | 59.70% | 46.07% | 51.68% | 46.04% | 0.620 |
| 128 | gru_basic | RNN | 59.93% | 48.60% | 51.84% | 47.99% | - |
| 128 | tcn | RNN | 57.65% | 45.16% | 45.39% | 43.54% | 1.101 |
| 128 | tiny_transformer | RNN | 58.20% | 51.46% | 49.82% | 48.54% | 0.606 |
| 160 | gru | RNN | 57.71% | 51.94% | 42.46% | 42.35% | 0.663 |
| 160 | gru_basic | RNN | **59.96%** | 50.85% | 52.96% | 49.78% | - |
| 160 | tcn | RNN | 56.74% | 46.94% | 47.63% | 45.45% | 0.710 |
| 160 | tiny_transformer | RNN | 55.87% | 49.97% | 55.65% | 49.85% | 1.055 |
| 192 | gru | RNN | 58.37% | 50.76% | 46.73% | 46.27% | **0.571** |
| 192 | gru_basic | RNN | 58.08% | 50.39% | 54.68% | 50.00% | - |
| 192 | tcn | RNN | 56.75% | 52.17% | 47.51% | 47.15% | 0.685 |
| 192 | tiny_transformer | RNN | 57.30% | 50.40% | 47.77% | 46.91% | 0.682 |
| 224 | gru | RNN | 58.35% | 50.72% | **59.66%** | 50.66% | 0.596 |
| 224 | gru_basic | RNN | 56.59% | 50.89% | 56.03% | 50.66% | - |
| 224 | tcn | RNN | 57.33% | 49.04% | 47.90% | 46.51% | 0.823 |
| 224 | tiny_transformer | RNN | 57.49% | 54.75% | 49.85% | 49.47% | 0.956 |
| 256 | gru | RNN | 55.70% | 50.59% | 52.61% | 49.49% | 0.994 |
| 256 | gru_basic | RNN | 57.70% | 54.58% | 44.90% | 44.82% | - |
| 256 | tcn | RNN | 56.91% | 51.70% | 56.54% | 51.39% | 0.669 |
| 256 | tiny_transformer | RNN | 57.05% | 53.39% | 54.33% | **51.75%** | 0.947 |
| 300 | gru | RNN | 57.11% | 54.26% | 46.95% | 46.92% | 0.725 |
| 300 | gru_basic | RNN | 56.68% | 48.16% | 57.16% | 48.05% | - |
| 300 | tcn | RNN | 59.05% | 51.87% | 47.00% | 46.70% | 0.620 |
| 300 | tiny_transformer | RNN | 56.00% | 51.34% | 50.49% | 48.88% | 1.326 |
| 384 | gru | RNN | 54.19% | 53.71% | 43.32% | 43.11% | 0.768 |
| 384 | gru_basic | RNN | 56.68% | 52.67% | 55.35% | 51.74% | - |
| 384 | tcn | RNN | 54.01% | 52.64% | 54.83% | 51.54% | 0.573 |
| 384 | tiny_transformer | RNN | 55.20% | **56.98%** | 47.49% | 47.46% | 0.728 |

## 12. Diễn giải kết quả

- `Balanced Accuracy` là chỉ số nên ưu tiên vì nó phản ánh khả năng nhận diện cả hai lớp, tránh trường hợp model đạt Accuracy cao do đoán nghiêng về lớp đa số.
- `late-fusion GRU + TCN + XGBoost` đạt Accuracy/F1 Macro cao nhất trong các kết quả đã hoàn tất: 75.95% Accuracy và 73.86% F1 Macro. Đây là ứng viên chính khi cần trình bày hybrid/ensemble gần mốc SOTA tham khảo.
- `rnn_gru` trong final run đạt Test Balanced Accuracy cao nhất: 77.35%. Đây là ứng viên chính nếu tiêu chí đánh giá ưu tiên cân bằng hai lớp.
- `residual_bigru_attn` là model đơn mới tốt nhất trong nhóm kiến trúc nâng cấp, đạt Accuracy 72.03% và Balanced Accuracy 73.76%, nhưng chưa vượt GRU theo Balanced Accuracy.
- `rnn_tcn` đạt Test Balanced Accuracy 72.52%, thấp hơn GRU nhưng vẫn mạnh và có lợi thế tính toán song song theo thời gian.
- `ml/xgboost` đạt Accuracy 70.52% và recall positive 72.37%, phù hợp làm baseline CPU nhẹ; tuy nhiên Balanced Accuracy 69.34% cho thấy lớp negative chưa tốt bằng GRU/TCN.
- `cnn` có Accuracy khá cao ở một số run, nhưng Balanced Accuracy thấp vì recall negative rất thấp; nếu dùng CNN riêng lẻ cần calibrate threshold hoặc fusion với landmark/RNN.
- Về latency end-to-end trên pipeline hiện tại, `xgboost_final`, `rnn_tcn_final`, `rnn_gru_final`, `cnn_gru_fusion` và `residual_bigru_attn` đều quanh 9.53-9.58 s/video 300 frames vì cùng phải chạy MediaPipe FaceMesh extraction trước khi dự đoán; late-fusion đạt 9.78 s/video vì chạy thêm ba predictor.
- Về model-side latency khi feature đã có sẵn, `xgboost_final` nhanh nhất với mean 0.48 ms/input; trong nhóm RNN mới, `cnn_gru_fusion` int8 đạt 8.96 ms/input và `residual_bigru_attn` int8 đạt 9.24 ms/input; late-fusion đạt 16.05 ms/input.
- Khi map kết quả với latency thực tế, late-fusion cho chất lượng tổng thể tốt nhất theo Accuracy/F1 nhưng cần giữ ba model thành phần; `rnn_gru_final` vẫn là model đơn chất lượng nhất theo Balanced Accuracy. XGBoost có lợi thế rõ ở predict-stage nhưng không có lợi thế end-to-end rõ ràng nếu pipeline vẫn dùng cùng bước feature extraction.
- Paper/SOTA DAiSEE thường không công bố latency inference end-to-end, nên bảng so sánh không tự thay thế bằng latency forward-only hoặc latency kiến trúc tái tạo.
- Feature extraction có cơ sở từ literature: EAR/PERCLOS cho trạng thái mắt và drowsiness, MAR cho yawning, head pose/gaze cho visual attention, landmark dynamics cho hành vi theo chuỗi.
- So với SOTA/paper benchmark, late-fusion đạt 75.95% Accuracy, rất gần mốc Santoni SVD-CNN 77.97% trong bảng tham khảo. Tuy nhiên chưa thể tuyên bố vượt SOTA vì protocol khác nhau: project đang dùng binary engagement, còn nhiều paper DAiSEE báo cáo 4-class accuracy.
- PCA sweep không vượt final run GRU. PCA giúp giảm chiều feature và quan sát trade-off, nhưng kết quả tốt nhất trong sweep vẫn thấp hơn checkpoint final train-all.

## 13. Đề xuất sử dụng

| Mục tiêu | Model nên chọn | Lý do |
| :--- | :--- | :--- |
| Kết quả chính khi nhấn mạnh Accuracy/F1 | Late-fusion `GRU + TCN + XGBoost` | Accuracy 75.95% và F1 Macro 73.86%, gần mốc SOTA tham khảo nhất trong các kết quả đã hoàn tất. |
| Kết quả chính khi ưu tiên Balanced Accuracy | `rnn_gru` final | Balanced Accuracy cao nhất: 77.35%; model đơn dễ triển khai hơn ensemble. |
| Baseline nhẹ để chạy CPU | `ml/xgboost` final | Accuracy cao, model-side latency **0.48 ms/input**; E2E nhanh nhất trong lần đo mới, khoảng **9.53 s/video** khi dùng MediaPipe extraction. |
| Cần cân bằng accuracy/latency E2E | `rnn_tcn` final int8/TorchScript | Balanced Accuracy 72.52%; E2E khoảng 9.54 s/video; model-side tốt nhất trong lần đo mới là 16.24 ms/input với int8. |
| Cần model đơn mới/hybrid nhẹ | `residual_bigru_attn` | Accuracy 72.03%, Balanced Accuracy 73.76%; tốt hơn TCN về Accuracy, model-side 9.24 ms/input sau int8 dynamic, nhưng chưa vượt GRU về Balanced Accuracy. |
| Nghiên cứu tiếp | Fusion thêm appearance/CNN hoặc landmark graph | Late-fusion hiện đã có lợi; hướng tiếp theo là thêm CNN/video appearance hoặc graph landmark vào ensemble thay vì thay thế GRU. |

## 14. Nguồn đối chiếu

- `checkpoints/reports/selected_model_performance.csv`
- `checkpoints/reports/run_metrics_manifest709.csv`
- `checkpoints/reports/end_to_end_latency_benchmark.json`
- `checkpoints/reports/end_to_end_latency_benchmark.csv`
- `checkpoints/reports/latency_benchmark_models.json`
- `checkpoints/reports/latency_benchmark_models.csv`
- `checkpoints/reports/late_fusion_gru_tcn_xgb_report.json`
- `checkpoints/runs/evolved_rnn_hybrid_models_20260608/results.jsonl`
- `logs/evolved_rnn_hybrid_models_20260608.log`
- `scripts/data/benchmark_end_to_end_latency.py`
- `scripts/data/benchmark_model_latency.py`
- `checkpoints/runs/final_rnn_temporal_models_20260529/train_all_summary.json`
- `checkpoints/runs/train_all_pca*/train_all_summary.json`
- `checkpoints/runs/trainml_pca*_ml/engagement_xgb.summary.json`
- Santoni et al. 2023, `SVD-CNN/PCA-CNN`: https://thesai.org/Downloads/Volume14No3/Paper_71-Convolutional_Neural_Network_Model.pdf
- PriorNet 2026 preprint: https://arxiv.org/abs/2605.03615
- DTransformer / PANet + STformer 2024: https://doi.org/10.1016/j.aej.2024.06.074
- PERCLOS drowsiness review: https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/
- OSA drowsiness visual signals + EEG: https://pmc.ncbi.nlm.nih.gov/articles/PMC11055081/
- Soukupová & Čech EAR/facial landmarks: https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf
- OpenFace 2.0: https://par.nsf.gov/servlets/purl/10099458
- MAR/head posture fatigue paper: https://www.nature.com/articles/s41598-026-44994-4

---

Ghi chú: một số checkpoint/log trong worktree đang ở trạng thái untracked/modified/deleted theo `git status`; báo cáo này chỉ đọc các file hiện có trên filesystem và không khôi phục hay thay đổi checkpoint cũ.
