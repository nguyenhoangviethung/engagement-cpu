import torch
import torch.nn.functional as F
from torch import nn

from engagement_daisee.common.config import FEATURE_DIM


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        return F.relu(x + residual, inplace=True)


class EngagementTCN(nn.Module):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        dropout: float = 0.3,
        kernel_size: int = 3,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        blocks = []
        dilations = [2**i for i in range(num_blocks)]
        for dilation in dilations:
            blocks.append(TemporalConvBlock(hidden_size, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(inputs)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        pooled_avg = torch.mean(x, dim=-1)
        pooled_max = torch.amax(x, dim=-1)
        pooled = torch.cat([pooled_avg, pooled_max], dim=-1)
        logits = self.head(pooled)
        return logits.view(-1)
