from engagement_daisee.common.config import FEATURE_DIM
from engagement_daisee.rnn.models.evolved import EngagementCNNGRUFusion, EngagementResidualBiGRUAttention
from engagement_daisee.rnn.models.gru import BasicGRUClassifier, EngagementGRU
from engagement_daisee.rnn.models.hybrid import EngagementHybridAttention
from engagement_daisee.rnn.models.lstm import EngagementBiLSTM
from engagement_daisee.rnn.models.stgcn import EngagementSTGCN
from engagement_daisee.rnn.models.tcn import EngagementTCN
from engagement_daisee.rnn.models.transformer import EngagementTinyTransformer


def build_sequence_model(
    model_name: str,
    input_size: int = FEATURE_DIM,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_heads: int = 4,
    kernel_size: int = 3,
    tcn_blocks: int = 3,
    tcn_kernel_size: int | None = None,
    max_seq_len: int = 64,
):
    name = model_name.strip().lower()
    if tcn_kernel_size is not None:
        kernel_size = int(tcn_kernel_size)

    if name == "gru":
        return EngagementGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if name in {"gru_basic", "simple_gru"}:
        return BasicGRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
        )

    if name in {"bilstm", "bi_lstm", "copur_bilstm"}:
        return EngagementBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
        )

    if name in {"tcn", "1dcnn", "temporal_cnn"}:
        return EngagementTCN(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            kernel_size=kernel_size,
            num_blocks=tcn_blocks,
        )

    if name in {"stgcn", "st-gcn", "graph_tcn"}:
        return EngagementSTGCN(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_blocks=tcn_blocks,
        )

    if name in {"hybrid", "hybrid_attn", "tcn_gru_attn", "multiscale_gru_attn"}:
        return EngagementHybridAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

    if name in {"cnn_gru_fusion", "spatiotemporal_hybrid", "cnn_bigru_fusion"}:
        return EngagementCNNGRUFusion(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            kernel_size=kernel_size,
        )

    if name in {"residual_bigru_attn", "res_bigru_attn", "bigru_self_attn"}:
        return EngagementResidualBiGRUAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            dropout=dropout,
            num_heads=num_heads,
        )

    if name in {"transformer", "tiny_transformer"}:
        return EngagementTinyTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

    raise ValueError(
        "Unsupported model_name: "
        f"{model_name}. Expected one of: gru, gru_basic|simple_gru, "
        "bilstm|bi_lstm|copur_bilstm, tcn|1dcnn|temporal_cnn, "
        "stgcn|st-gcn|graph_tcn, hybrid_attn|tcn_gru_attn, "
        "cnn_gru_fusion|residual_bigru_attn, "
        "transformer|tiny_transformer"
    )
