import torch
from torch import nn


class EngagementBiLSTM(nn.Module):
    """Bidirectional LSTM head inspired by Copur's OpenFace engagement model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(features)
        return self.head(outputs[:, -1, :]).squeeze(-1)
