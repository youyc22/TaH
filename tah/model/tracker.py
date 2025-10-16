import torch
from typing import Any, Dict, List
import pandas as pd

class TaHTracker:
    """Utility to track TaH model internal states.

    Currently it records top-k logits for each call of
    :func:`TaHForCausalLM._process_sparse_iteration` which corresponds to one
    iteration in the recurrent loop. The tracker can be attached to an
    ``TaHForCausalLM`` instance without modifying its code.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.records: List[Dict[str, Any]] = []
        self._orig_fn = None
        self._model = None
        self._call_idx = 0
        # Keep original output_updater forward and a pending context queue
        self._orig_updater_fn = None
        self._updater = None
        self._pending_contexts: List[Dict[str, Any]] = []

    def attach(self, model: Any) -> None:
        """Attach tracker to ``model``.

        Parameters
        ----------
        model : TaHForCausalLM
            Model to track. ``model._process_sparse_iteration`` will be wrapped
            so that logits from every iteration are logged.
        """
        if self._model is not None:
            raise RuntimeError("Tracker already attached to a model")

        self._model = model
        self._orig_fn = model._process_sparse_iteration

        def wrapper(*args, **kwargs):
            outputs = self._orig_fn(*args, **kwargs)
            # Cache minimal context for the subsequent output_updater call
            iter_depth = kwargs.get("iter_depth")
            if iter_depth is None and len(args) > 5:
                iter_depth = args[5]

            valid_mask = kwargs.get("valid_mask")
            if valid_mask is None and len(args) > 2:
                valid_mask = args[2]

            cache = kwargs.get("past_key_values")
            if cache is None and len(args) > 6:
                cache = args[6]

            # Queue context to be consumed by output_updater wrapper
            self._pending_contexts.append({
                "iter_depth": iter_depth,
                "valid_mask": valid_mask,
                "cache": cache,
            })
            return outputs

        model._process_sparse_iteration = wrapper

        # Also wrap the output_updater to record active_updated_cumulative_logits
        if hasattr(model, "output_updater") and model.output_updater is not None:
            self._updater = model.output_updater
            self._orig_updater_fn = model.output_updater.forward

            def updater_wrapper(*u_args, **u_kwargs):
                updated_logits = self._orig_updater_fn(*u_args, **u_kwargs)

                # Obtain the most recent pending context if available
                context = None
                if self._pending_contexts:
                    context = self._pending_contexts.pop(0)

                if context is not None and updated_logits is not None:
                    k = min(self.top_k, updated_logits.size(-1))
                    last_token_logits = updated_logits[:, -1, :]
                    values, indices = torch.topk(last_token_logits, k=k, dim=-1)
                    perplexity, entropy = self.logits_to_perplexity_entropy(last_token_logits)

                    iter_depth = context.get("iter_depth")
                    valid_mask = context.get("valid_mask")
                    cache = context.get("cache")

                    for batch_idx in range(updated_logits.size(0)):
                        # Skip if last position is padding/inactive
                        if valid_mask is not None and valid_mask[batch_idx, -1] == 0:
                            continue

                        record = {
                            "batch_idx": batch_idx,
                            "call_index": self._call_idx,
                            "iter_depth": iter_depth,
                            "step_index": (cache.get_seq_length() if cache is not None else None),
                            "perplexity": perplexity[batch_idx].item(),
                            "entropy": entropy[batch_idx].item(),
                            "topk_values": values.detach().cpu()[batch_idx, :].tolist(),
                            "topk_indices": indices.detach().cpu()[batch_idx, :].tolist(),
                        }
                        self.records.append(record)
                    self._call_idx += 1

                return updated_logits

            model.output_updater.forward = updater_wrapper

    def detach(self) -> None:
        """Remove hooks and restore the original model method."""
        if self._model is not None and self._orig_fn is not None:
            self._model._process_sparse_iteration = self._orig_fn
        if self._updater is not None and self._orig_updater_fn is not None:
            self._updater.forward = self._orig_updater_fn
        self._model = None
        self._orig_fn = None
        self._updater = None
        self._orig_updater_fn = None

    def clear(self) -> None:
        """Clear all recorded states."""
        self.records.clear()
        self._call_idx = 0
        self._pending_contexts.clear()

    @staticmethod
    def logits_to_perplexity_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute the perplexity of a logits tensor.

        Perplexity is calculated as :math:`\exp(H(p))` where
        :math:`H(p)` is the entropy of the probability distribution obtained
        by applying softmax to the logits.

        The returned value is the exponential of the entropy over
        all tokens in ``logits``, keeping the same ... dimensions as logits except for the vocab dimension.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor of shape ``(..., vocab_size)`` containing pre-softmax
            activations.

        Returns
        -------
        torch.Tensor
            The computed perplexity, shape ``...`` (same as logits without vocab dimension).
        torch.Tensor
            The computed entropy, shape ``...`` (same as logits without vocab dimension).
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        # Compute log-probabilities in a numerically stable way
        log_probs = torch.log_softmax(logits, dim=-1)
        # Entropy per token: -sum(p * log p)
        entropy = -(probs * log_probs).sum(dim=-1)
        # Perplexity = exp(entropy)
        return torch.exp(entropy).detach().cpu(), entropy.detach().cpu()

    def to_pandas(self, selected_keys: List[str] | None = None):
        """Convert tracked records to a ``pandas.DataFrame``.

        Parameters
        ----------
        selected_keys : List[str] | None
            Optional list of keys to keep from each record. If omitted all
            keys are included.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the requested fields for every tracked call.
        """
        if not self.records:
            return pd.DataFrame()

        # Determine which keys to include
        ALL_KEYS = list(self.records[0].keys())

        if selected_keys is None:
            keys = ALL_KEYS
        else:
            keys = list(selected_keys)

        # Build data limited to desired keys, falling back to None when key missing
        data = [{k: rec.get(k) for k in keys} for rec in self.records]

        return pd.DataFrame(data)[keys]
        