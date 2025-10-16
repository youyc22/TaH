import torch
import torch.nn as nn
from typing import Optional

from tah.model.registry import (
    register_iter_label_generator,
    get_iter_label_generator_class,
)


class IterLabelGenerator(nn.Module):
    """Base class for generating per-token iter-count labels.

    Contract:
    - prepare(batch_size, seq_len, device, dtype): allocate internal buffers
    - intra_iter_labels(...): return labels for current active tokens, and update internal full labels
    - finalize(): return full (B, S) labels accumulated across iterations
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.full_labels = None

    def prepare(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.full_labels = torch.full(
            (batch_size, seq_len), fill_value=0, device=device, dtype=torch.long
        )

    @staticmethod
    def _assign_active(current_iter_mask: torch.BoolTensor, src: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        """Scatter active `src` back to dense `dest` (no padding handling beyond mask)."""
        B, S = current_iter_mask.shape
        active_counts = current_iter_mask.sum(1)
        for b in range(B):
            n = int(active_counts[b].item())
            if n:
                dest[b, current_iter_mask[b]] = src[b, :n]
        return dest

    def intra_iter_labels(
        self,
        active_logits: torch.Tensor,
        active_labels_shifted: Optional[torch.Tensor],
        iter_depth: int,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        prompt_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> Optional[torch.LongTensor]:
        raise NotImplementedError

    def finalize(self) -> Optional[torch.LongTensor]:
        return self.full_labels


@register_iter_label_generator
class FixedIterLabelGenerator(IterLabelGenerator):
    """Pass-through labels coming from dataset via `iter_count_labels`.

    The model will supply the active slice, this generator just maps prompt to ignore.
    """

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def intra_iter_labels(
        self,
        active_iter_count_labels: torch.LongTensor,
        current_iter_mask: torch.BoolTensor,
        **kwargs,
    ) -> Optional[torch.LongTensor]:
        # Ensure long dtype for labels
        active_iter_count_labels = active_iter_count_labels.to(dtype=torch.long)

        # Update full labels with the latest observed labels for active positions
        if self.full_labels is not None and (self.full_labels.shape == (current_iter_mask.shape[0], current_iter_mask.shape[1])):
            proposal = torch.zeros_like(active_iter_count_labels)
            valid = (active_iter_count_labels != self.ignore_index)
            proposal[valid] = active_iter_count_labels[valid]
            current = self.full_labels.clone()
            tmp = torch.zeros_like(self.full_labels)
            tmp = self._assign_active(current_iter_mask, proposal, tmp)
            self.full_labels = torch.maximum(current, tmp)

        return active_iter_count_labels


@register_iter_label_generator
class DynamicMismatchIterLabelGenerator(IterLabelGenerator):
    """Generate per-iteration pseudo count labels based on mismatch.

    Rule at depth d (1-indexed):
    - If mismatch → label = d + 1
    - Else        → label = d
    So that (label > d) is the desired continue target.
    """

    def __init__(self, max_iter: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter

    @staticmethod
    def _compute_mismatch_continue(logits: torch.Tensor, labels_shifted: torch.Tensor, ignore_index: int) -> torch.BoolTensor:
        # Handle causal LM shift: logits[i] predicts labels[i+1]
        # We need to compare logits[:-1] with labels[1:] for proper alignment
        if logits.dim() >= 2 and logits.shape[-2] > 1 and labels_shifted.shape[-1] > 1:
            shifted_logits = logits[..., :-1, :]
            shifted_labels = labels_shifted[..., :-1]
            predicted = torch.argmax(shifted_logits, dim=-1)
            mismatch = (predicted != shifted_labels)
            cont = torch.cat([mismatch, torch.zeros_like(mismatch[..., :1])], dim=-1)
        else:
            predicted = torch.argmax(logits, dim=-1)
            mismatch = (predicted != labels_shifted)
            cont = mismatch
        # Exclude ignore positions
        valid = (labels_shifted != ignore_index)
        return (cont & valid)

    def intra_iter_labels(
        self,
        active_logits: torch.Tensor,
        active_labels_shifted: Optional[torch.Tensor],
        iter_depth: int,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        prompt_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> Optional[torch.LongTensor]:
        if active_labels_shifted is None or active_logits is None:
            return None

        # Compute mismatch-based continue mask on the active slice
        continue_mask = self._compute_mismatch_continue(active_logits, active_labels_shifted, ignore_index)

        # Only supervise valid active tokens
        valid_active = (active_valid_mask == 1)

        # Build count labels for active slice
        depth_tensor = torch.full_like(active_logits[..., 0], fill_value=iter_depth, dtype=torch.long)
        labels_active = torch.where(continue_mask, depth_tensor + 1, depth_tensor)
        # Ensure labels do not exceed max_iter
        labels_active = torch.clamp(labels_active, max=self.max_iter)

        # Also ignore positions that are not valid active tokens
        labels_active = labels_active.masked_fill(~valid_active.bool(), ignore_index)
        labels_active = labels_active.to(dtype=torch.long)

        # Accumulate full labels: take max across depths to ensure monotonicity
        if self.full_labels is not None and (self.full_labels.shape == (current_iter_mask.shape[0], current_iter_mask.shape[1])):
            proposal = labels_active.clone()
            proposal = proposal.masked_fill(proposal == ignore_index, 0)
            current = self.full_labels.clone()
            tmp = torch.zeros_like(self.full_labels)
            tmp = self._assign_active(current_iter_mask, proposal, tmp)
            self.full_labels = torch.maximum(current, tmp)

        return labels_active




@register_iter_label_generator
class MaxIterLabelGenerator(IterLabelGenerator):
    """Always assign `max_iter` as the label for active tokens.

    Invalid or non-active positions are masked with `ignore_index`.
    """

    def __init__(self, max_iter: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter

    def intra_iter_labels(
        self,
        active_logits: Optional[torch.Tensor],
        active_labels_shifted: Optional[torch.Tensor],
        iter_depth: int,
        current_iter_mask: torch.BoolTensor,
        active_valid_mask: torch.LongTensor,
        prompt_mask: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> Optional[torch.LongTensor]:
        # Determine the active slice shape and device
        base_tensor: Optional[torch.Tensor] = None
        if active_logits is not None:
            base_tensor = active_logits[..., 0]
        elif active_labels_shifted is not None:
            base_tensor = active_labels_shifted
        else:
            base_tensor = active_valid_mask

        labels_active = torch.full(
            base_tensor.shape,
            fill_value=self.max_iter,
            device=base_tensor.device,
            dtype=torch.long,
        )

        # Only supervise valid active tokens
        valid_active = (active_valid_mask == 1)
        labels_active = labels_active.masked_fill(~valid_active.bool(), ignore_index)

        # Accumulate full labels for the dense (B, S) view
        if self.full_labels is not None and (self.full_labels.shape == (current_iter_mask.shape[0], current_iter_mask.shape[1])):
            proposal = labels_active.clone()
            proposal = proposal.masked_fill(proposal == ignore_index, 0)
            current = self.full_labels.clone()
            tmp = torch.zeros_like(self.full_labels)
            tmp = self._assign_active(current_iter_mask, proposal, tmp)
            self.full_labels = torch.maximum(current, tmp)

        return labels_active
