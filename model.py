import torch
from torch import nn
import torch.nn.functional as F

from config import FEATURE_DIM

class TemporalAttention(nn.Module):
    """
    Cơ chế Attention giúp mô hình tự động tìm ra các khung hình (frames) 
    quan trọng nhất trong chuỗi 30 frames thay vì chỉ lấy frame cuối cùng.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, rnn_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # rnn_output shape: (Batch, Seq_len, Hidden_size)
        
        # Tính điểm chú ý cho từng frame
        attn_weights = self.attention(rnn_output) # (Batch, Seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1) 
        
        # Nhân điểm chú ý với output của GRU và cộng dồn lại (Weighted Sum)
        context = torch.sum(attn_weights * rnn_output, dim=1) # (Batch, Hidden_size)
        return context, attn_weights


class EngagementGRU(nn.Module):
    def __init__(self, input_size: int = FEATURE_DIM, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # 1. Feature Extractor: Xử lý vector đặc trưng ban đầu
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Bi-GRU: Học thông tin chuỗi thời gian theo 2 chiều
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, # Rất quan trọng!
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Attention Layer (Nhân đôi hidden_size vì GRU chạy 2 chiều)
        self.attention = TemporalAttention(hidden_size * 2)
        
        # 4. Classifier: Đưa ra dự đoán cuối cùng
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 1. Biến đổi đặc trưng ban đầu
        x = self.feature_extractor(inputs)
        
        # 2. Chạy qua mạng GRU
        gru_out, _ = self.gru(x)
        
        # 3. Áp dụng Attention để nén 30 frames thành 1 vector ngữ cảnh duy nhất
        context, attn_weights = self.attention(gru_out)
        
        # 4. Phân loại
        logits = self.classifier(context)
        
        # Dùng view(-1) để tránh lỗi Broadcast Loss
        return logits.view(-1)