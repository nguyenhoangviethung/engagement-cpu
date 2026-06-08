import torch
import torch.nn.functional as F
from torch import nn

from engagement_daisee.common.config import FEATURE_DIM
from engagement_daisee.rnn.models.attention import TemporalAttention


class EngagementCNNGRUFusion(nn.Module):
    """Parallel 1D-CNN and BiGRU fusion for short and sustained temporal cues."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        kernel_size: int = 5,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        self.gru_branch = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_attention = TemporalAttention(hidden_size * 2)
        fusion_size = hidden_size * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_encoder(inputs)

        cnn = self.cnn_branch(x.transpose(1, 2))
        cnn_avg = torch.mean(cnn, dim=-1)
        cnn_max = torch.amax(cnn, dim=-1)

        gru_out, _ = self.gru_branch(x)
        gru_context, _ = self.gru_attention(gru_out)

        fused = torch.cat([cnn_avg, cnn_max, gru_context], dim=-1)
        logits = self.classifier(fused)
        return logits.view(-1)


class EngagementResidualBiGRUAttention(nn.Module):
    """Residual BiGRU with lightweight self-attention and gated pooling."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()
        if (hidden_size * 2) % num_heads != 0:
            raise ValueError(f"hidden_size*2={hidden_size * 2} must be divisible by num_heads={num_heads}")

        self.input_encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru_input = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(hidden_size * 2)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size * 2)
        self.gate = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 6),
            nn.Linear(hidden_size * 6, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_encoder(inputs)
        gru_in = self.gru_input(encoded)
        gru_out, _ = self.gru(gru_in)
        x = self.gru_norm(gru_out + encoded)
        attended, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.attn_norm(x + attended)

        weights = F.softmax(self.gate(x), dim=1)
        weighted = torch.sum(weights * x, dim=1)
        pooled = torch.cat([weighted, torch.mean(x, dim=1), torch.amax(x, dim=1)], dim=-1)
        logits = self.classifier(pooled)
        return logits.view(-1)
