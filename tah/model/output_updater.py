from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tah.model.recurrent_transformer import TaHForCausalLM

from tah.model.registry import register_output_updater, get_output_updater_class, capture_init_args


class OutputUpdater(nn.Module, ABC):
    """
    Base class for updating output logits between iterations.
    
    This class is designed to efficiently handle tensors of arbitrary shape (..., vocab_size),
    where the leading dimensions can be any combination of batch, sequence, or other
    dimensions. All operations preserve the leading dimensions and only operate on
    the last dimension for vocabulary operations.
    """

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor] = None,
        iter_depth: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Return updated logits for accumulation.
        
        This method efficiently handles tensors of arbitrary shape, preserving all
        leading dimensions while operating only on the vocabulary dimension.
        
        Args:
            logits: The current iteration logits, shape (..., vocab_size)
            prev_logits: The previous accumulated logits, shape (..., vocab_size) or None for first iteration
            iter_depth: Current iteration depth (0-indexed)
            **kwargs: Additional arguments
        
        Returns:
            The updated accumulated logits, shape (..., vocab_size)
            
        Note:
            All leading dimensions (...) are preserved exactly. The implementation
            is optimized for efficient processing regardless of the number or size
            of leading dimensions (e.g., batch size, sequence length, etc.).
        """


@register_output_updater
@capture_init_args
class NoneUpdater(OutputUpdater):
    """
    No-op output updater that returns current logits without accumulation.
    This is the default behavior to maintain backward compatibility.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor] = None,
        iter_depth: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """Simply return current logits without any accumulation."""
        return logits


@register_output_updater
@capture_init_args
class AdditiveLogitsUpdater(OutputUpdater):
    """
    Additive output updater that accumulates logits across iterations.
    
    On the first iteration (prev_logits is None), returns current logits.
    On subsequent iterations, returns prev_logits + current logits.
    This allows the model to learn residual corrections to the output.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor] = None,
        iter_depth: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Accumulate logits additively.
        
        Args:
            logits: Current iteration logits (..., vocab_size)
            prev_logits: Previous accumulated logits (..., vocab_size) or None
            iter_depth: Current iteration depth (0-indexed)
            
        Returns:
            Accumulated logits (..., vocab_size)
        """
        if prev_logits is None:
            # First iteration: return current logits as-is
            return logits
        else:
            # Subsequent iterations: add to accumulated logits
            return prev_logits + logits
    

