"""Compatibility facade for sequence models.

Canonical implementations now live under engagement_daisee.rnn.models.
This module is kept to avoid breaking older imports.
"""

from engagement_daisee.rnn.models import (
    BasicGRUClassifier,
    EngagementCNNGRUFusion,
    EngagementBiLSTM,
    EngagementGRU,
    EngagementResidualBiGRUAttention,
    EngagementTCN,
    EngagementTinyTransformer,
    TemporalAttention,
    TemporalConvBlock,
    build_sequence_model,
)

__all__ = [
    "TemporalAttention",
    "TemporalConvBlock",
    "EngagementGRU",
    "BasicGRUClassifier",
    "EngagementBiLSTM",
    "EngagementTCN",
    "EngagementTinyTransformer",
    "EngagementCNNGRUFusion",
    "EngagementResidualBiGRUAttention",
    "build_sequence_model",
]
