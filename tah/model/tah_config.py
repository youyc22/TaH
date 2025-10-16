from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class TaHConfig:
    """Configuration for TaH model components."""
    # Overidable configs
    embedding_key: str = "model.embed_tokens"
    max_iter: int = None
    iter_decider: str = None
    input_updater: str = None
    output_updater: str = None
    train_loss: str = None
    eval_loss: str = None
    # Optional: use a different iter_decider for evaluation/inference
    eval_iter_decider: str = None
    adapter: str = None
    iter_label_generator: str = None
    iter_attention_mode: str = "duo"  # Attention visibility mode: "duo", "root", or "same_iter"

    # Non-overidable configs
    iter_decider_kwargs: Dict[str, Any] = field(default_factory=dict)
    input_updater_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_updater_kwargs: Dict[str, Any] = field(default_factory=dict)
    train_loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_iter_decider_kwargs: Dict[str, Any] = field(default_factory=dict)
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    iter_label_generator_kwargs: Dict[str, Any] = field(default_factory=dict)
