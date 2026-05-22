import torch
import torch.nn.functional as F
from torch import nn


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
