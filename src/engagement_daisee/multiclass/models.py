from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from engagement_daisee.common.config import FEATURE_DIM
from engagement_daisee.rnn.models.evolved import (
    EngagementCNNGRUFusion as BinaryCNNGRUFusion,
    EngagementResidualBiGRUAttention as BinaryResidualBiGRUAttention,
)
from engagement_daisee.rnn.models.gru import BasicGRUClassifier as BinaryBasicGRUClassifier
from engagement_daisee.rnn.models.gru import EngagementGRU as BinaryEngagementGRU
from engagement_daisee.rnn.models.hybrid import EngagementHybridAttention as BinaryHybridAttention
from engagement_daisee.rnn.models.lstm import EngagementBiLSTM as BinaryBiLSTM
from engagement_daisee.rnn.models.stgcn import EngagementSTGCN as BinarySTGCN
from engagement_daisee.rnn.models.tcn import EngagementTCN as BinaryTCN
from engagement_daisee.rnn.models.transformer import EngagementTinyTransformer as BinaryTinyTransformer


NUM_CLASSES = 4


def _make_last_linear(module: nn.Module, num_classes: int) -> nn.Module:
    if isinstance(module, nn.Linear):
        return nn.Linear(module.in_features, num_classes)
    raise TypeError(f"Expected nn.Linear, got {type(module)!r}")


class MulticlassEngagementGRU(BinaryEngagementGRU):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.classifier[-1] = _make_last_linear(self.classifier[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(inputs)
        gru_out, _ = self.gru(x)
        context, _ = self.attention(gru_out)
        return self.classifier(context)


class MulticlassBasicGRUClassifier(BinaryBasicGRUClassifier):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.classifier = _make_last_linear(self.classifier, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(inputs)
        last_hidden = output[:, -1, :]
        return self.classifier(self.dropout(last_hidden))


class MulticlassBiLSTM(BinaryBiLSTM):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.head[-1] = _make_last_linear(self.head[-1], num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(features)
        return self.head(outputs[:, -1, :])


class MulticlassTCN(BinaryTCN):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        dropout: float = 0.3,
        kernel_size: int = 3,
        num_blocks: int = 3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
        )
        self.num_classes = num_classes
        self.head[-1] = _make_last_linear(self.head[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(inputs)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        pooled_avg = torch.mean(x, dim=-1)
        pooled_max = torch.amax(x, dim=-1)
        pooled = torch.cat([pooled_avg, pooled_max], dim=-1)
        return self.head(pooled)


class MulticlassSTGCN(BinarySTGCN):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        dropout: float = 0.3,
        num_blocks: int = 3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_blocks=num_blocks,
        )
        self.num_classes = num_classes
        self.head[-1] = _make_last_linear(self.head[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = inputs.shape
        x = inputs.reshape(batch_size * seq_len, feat_dim)
        x = self.input_adapter(x)
        x = self.input_norm(x)
        x = x.reshape(batch_size, seq_len, self.num_nodes, 3)
        x = x.permute(0, 3, 1, 2).contiguous()

        adjacency = F.softmax(self.adjacency_logits, dim=-1)
        for block in self.blocks:
            x = block(x, adjacency)

        pooled_avg = x.mean(dim=(2, 3))
        pooled_max = x.amax(dim=(2, 3))
        pooled = torch.cat([pooled_avg, pooled_max], dim=-1)
        return self.head(pooled)


class MulticlassHybridAttention(BinaryHybridAttention):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        max_seq_len: int = 64,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )
        self.num_classes = num_classes
        self.classifier[-1] = _make_last_linear(self.classifier[-1], num_classes)

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
        return self.classifier(pooled)


class MulticlassCNNGRUFusion(BinaryCNNGRUFusion):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        kernel_size: int = 5,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.num_classes = num_classes
        self.classifier[-1] = _make_last_linear(self.classifier[-1], num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_encoder(inputs)

        cnn = self.cnn_branch(x.transpose(1, 2))
        cnn_avg = torch.mean(cnn, dim=-1)
        cnn_max = torch.amax(cnn, dim=-1)

        gru_out, _ = self.gru_branch(x)
        gru_context, _ = self.gru_attention(gru_out)

        fused = torch.cat([cnn_avg, cnn_max, gru_context], dim=-1)
        return self.classifier(fused)


class MulticlassResidualBiGRUAttention(BinaryResidualBiGRUAttention):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_heads: int = 4,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
        )
        self.num_classes = num_classes
        self.classifier[-1] = _make_last_linear(self.classifier[-1], num_classes)

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
        return self.classifier(pooled)


class MulticlassTinyTransformer(BinaryTinyTransformer):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        dropout: float = 0.3,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 64,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )
        self.num_classes = num_classes
        self.classifier[-1] = _make_last_linear(self.classifier[-1], num_classes)

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
        return self.classifier(context)


MULTICLASS_MODELS: dict[str, type[nn.Module]] = {
    "gru": MulticlassEngagementGRU,
    "gru_basic": MulticlassBasicGRUClassifier,
    "simple_gru": MulticlassBasicGRUClassifier,
    "bilstm": MulticlassBiLSTM,
    "bi_lstm": MulticlassBiLSTM,
    "copur_bilstm": MulticlassBiLSTM,
    "tcn": MulticlassTCN,
    "1dcnn": MulticlassTCN,
    "temporal_cnn": MulticlassTCN,
    "stgcn": MulticlassSTGCN,
    "st-gcn": MulticlassSTGCN,
    "graph_tcn": MulticlassSTGCN,
    "hybrid": MulticlassHybridAttention,
    "hybrid_attn": MulticlassHybridAttention,
    "tcn_gru_attn": MulticlassHybridAttention,
    "multiscale_gru_attn": MulticlassHybridAttention,
    "cnn_gru_fusion": MulticlassCNNGRUFusion,
    "spatiotemporal_hybrid": MulticlassCNNGRUFusion,
    "cnn_bigru_fusion": MulticlassCNNGRUFusion,
    "residual_bigru_attn": MulticlassResidualBiGRUAttention,
    "res_bigru_attn": MulticlassResidualBiGRUAttention,
    "bigru_self_attn": MulticlassResidualBiGRUAttention,
    "transformer": MulticlassTinyTransformer,
    "tiny_transformer": MulticlassTinyTransformer,
}


def build_multiclass_model(
    model_name: str,
    input_size: int = FEATURE_DIM,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_heads: int = 4,
    kernel_size: int = 3,
    tcn_blocks: int = 3,
    max_seq_len: int = 64,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    name = model_name.strip().lower()
    if name not in MULTICLASS_MODELS:
        raise ValueError(
            "Unsupported model_name: "
            f"{model_name}. Expected one of: {', '.join(sorted(MULTICLASS_MODELS))}"
        )

    model_cls = MULTICLASS_MODELS[name]
    if model_cls is MulticlassEngagementGRU:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            num_classes=num_classes,
        )
    if model_cls is MulticlassBasicGRUClassifier:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            num_classes=num_classes,
        )
    if model_cls is MulticlassBiLSTM:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            num_classes=num_classes,
        )
    if model_cls is MulticlassTCN:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            kernel_size=kernel_size,
            num_blocks=tcn_blocks,
            num_classes=num_classes,
        )
    if model_cls is MulticlassSTGCN:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_blocks=tcn_blocks,
            num_classes=num_classes,
        )
    if model_cls is MulticlassHybridAttention:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
        )
    if model_cls is MulticlassCNNGRUFusion:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            kernel_size=kernel_size,
            num_classes=num_classes,
        )
    if model_cls is MulticlassResidualBiGRUAttention:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            num_heads=num_heads,
            num_classes=num_classes,
        )
    if model_cls is MulticlassTinyTransformer:
        return model_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
        )
    raise ValueError(f"Unhandled multiclass model class for {model_name}: {model_cls!r}")
