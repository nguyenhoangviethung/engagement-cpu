import math

import torch
from torch import nn
import torch.nn.functional as F

from engagement_daisee.common.config import FEATURE_DIM


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.Tanh(),
            nn.Linear(max(1, hidden_size // 2), 1),
        )

    def forward(self, rnn_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_weights = self.attention(rnn_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * rnn_output, dim=1)
        return context, attn_weights


class EngagementGRU(nn.Module):
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
        dilations = [2 ** i for i in range(num_blocks)]
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


class EngagementTinyTransformer(nn.Module):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        dropout: float = 0.3,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 64,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
        )
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
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

        x = self.input_proj(inputs)
        x = x + self.positional_embedding[:, :seq_len, :]
        x = self.encoder(x)

        attn = F.softmax(self.attn_pool(x), dim=1)
        context = torch.sum(attn * x, dim=1)
        logits = self.classifier(context)
        return logits.view(-1)


def build_sequence_model(
    model_name: str,
    input_size: int = FEATURE_DIM,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_heads: int = 4,
    kernel_size: int = 3,
    tcn_blocks: int = 3,
    max_seq_len: int = 64,
) -> nn.Module:
    name = model_name.strip().lower()

    if name == "gru":
        return EngagementGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if name == "tcn":
        return EngagementTCN(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            kernel_size=kernel_size,
            num_blocks=tcn_blocks,
        )

    if name == "transformer":
        return EngagementTinyTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

    raise ValueError(f"Unsupported model_name: {model_name}. Expected one of: gru, tcn, transformer")
