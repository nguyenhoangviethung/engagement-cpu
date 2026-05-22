import torch
from torch import nn

from engagement_daisee.common.config import FEATURE_DIM
from engagement_daisee.rnn.models.attention import TemporalAttention


class EngagementGRU(nn.Module):
    """Bidirectional GRU + attention (existing default architecture)."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(inputs)
        gru_out, _ = self.gru(x)
        context, _ = self.attention(gru_out)
        logits = self.classifier(context)
        return logits.view(-1)


class BasicGRUClassifier(nn.Module):
    """Lightweight baseline GRU kept explicitly for comparison."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(inputs)
        last_hidden = output[:, -1, :]
        logits = self.classifier(self.dropout(last_hidden))
        return logits.view(-1)
