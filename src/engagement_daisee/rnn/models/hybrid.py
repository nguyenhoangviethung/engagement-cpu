import torch
import torch.nn.functional as F
from torch import nn

from engagement_daisee.common.config import FEATURE_DIM


class MultiScaleTemporalConv(nn.Module):
    """Parallel temporal filters for short/medium motion cues."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        branches = []
        for kernel_size in (3, 5, 7):
            branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=hidden_size,
                    ),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_size * len(branches), hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fuse(x)
        return x.transpose(1, 2)


class GatedAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.value(inputs)) * torch.sigmoid(self.gate(inputs))
        weights = F.softmax(self.score(gated), dim=1)
        pooled = torch.sum(weights * inputs, dim=1)
        return pooled, weights


class EngagementHybridAttention(nn.Module):
    """Multi-scale TCN + BiGRU + self-attention model for DAiSEE features."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        max_seq_len: int = 64,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_frontend = MultiScaleTemporalConv(hidden_size=hidden_size, dropout=dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        sequence_size = hidden_size * 2
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, sequence_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=sequence_size,
            nhead=num_heads,
            dim_feedforward=sequence_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pooling = GatedAttentionPooling(sequence_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(sequence_size * 3),
            nn.Linear(sequence_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        nn.init.trunc_normal_(self.positional_embedding, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = inputs.shape[1]
        if seq_len > self.positional_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max positional length {self.positional_embedding.shape[1]}"
            )

        x = self.feature_encoder(inputs)
        x = x + self.temporal_frontend(x)
        x, _ = self.gru(x)
        x = x + self.positional_embedding[:, :seq_len, :]
        x = self.self_attention(x)
        attentive, _ = self.pooling(x)
        pooled = torch.cat([attentive, torch.mean(x, dim=1), torch.amax(x, dim=1)], dim=-1)
        logits = self.classifier(pooled)
        return logits.view(-1)
