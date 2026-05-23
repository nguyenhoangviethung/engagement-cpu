import torch
import torch.nn.functional as F
from torch import nn

from engagement_daisee.common.config import FEATURE_DIM


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
