import torch
import torch.nn.functional as F
from torch import nn
import math

from engagement_daisee.common.config import FEATURE_DIM


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.graph_proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time, nodes]
        residual = self.residual(x)
        x = torch.einsum("nctv,vw->nctw", x, adjacency)
        x = self.graph_proj(x)
        x = self.temporal(x)
        x = self.bn(x)
        x = self.dropout(x)
        return F.relu(x + residual, inplace=True)


class EngagementSTGCN(nn.Module):
    """Adaptive ST-GCN for privacy-preserving landmark-like feature sequences."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        dropout: float = 0.3,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.num_nodes = math.ceil(input_size / 3)
        self.graph_input_size = self.num_nodes * 3

        self.input_adapter = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.graph_input_size),
        )
        self.input_norm = nn.BatchNorm1d(self.graph_input_size)
        self.adjacency_logits = nn.Parameter(torch.eye(self.num_nodes, dtype=torch.float32))

        channels = [3] + [hidden_size] * max(1, num_blocks)
        blocks = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.append(STGCNBlock(in_channels=in_channels, out_channels=out_channels, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = inputs.shape
        x = inputs.reshape(batch_size * seq_len, feat_dim)
        x = self.input_adapter(x)
        x = self.input_norm(x)
        x = x.reshape(batch_size, seq_len, self.num_nodes, 3)
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, channels=3, time, nodes]

        adjacency = F.softmax(self.adjacency_logits, dim=-1)
        for block in self.blocks:
            x = block(x, adjacency)

        pooled_avg = x.mean(dim=(2, 3))
        pooled_max = x.amax(dim=(2, 3))
        pooled = torch.cat([pooled_avg, pooled_max], dim=-1)
        logits = self.head(pooled)
        return logits.view(-1)
