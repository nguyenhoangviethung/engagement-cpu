"""RNN package with lazy model imports.

Feature extraction does not require PyTorch. Keeping these imports lazy allows
the extraction CLI to run in lightweight or recovery environments.
"""

from importlib import import_module

__all__ = [
    "EngagementGRU",
    "BasicGRUClassifier",
    "EngagementBiLSTM",
    "EngagementTCN",
    "EngagementTinyTransformer",
    "build_sequence_model",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    models = import_module("engagement_daisee.rnn.models")
    return getattr(models, name)
