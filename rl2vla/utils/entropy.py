"""Entropy utilities for tokenized action policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Aggregation = Literal["mean", "sum"]


@dataclass(frozen=True)
class EntropyConfig:
    """Configuration for token-level action entropy."""

    aggregation: Aggregation = "mean"
    normalize_by_vocab: bool = True
    eps: float = 1e-8


def categorical_entropy_from_logits(
    logits: torch.Tensor,
    config: EntropyConfig | None = None,
) -> torch.Tensor:
    """Compute aggregated entropy for tokenized categorical action logits.

    Args:
        logits: Tensor with shape ``[batch, action_tokens, vocab]``.
        config: Entropy aggregation and normalization settings.

    Returns:
        Tensor with shape ``[batch, 1]`` containing the aggregated entropy.
    """

    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, T, V], got {tuple(logits.shape)}")

    config = config or EntropyConfig()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)

    if config.normalize_by_vocab:
        vocab = logits.shape[-1]
        normalizer = torch.log(torch.tensor(float(vocab), device=logits.device, dtype=logits.dtype))
        token_entropy = token_entropy / normalizer.clamp_min(config.eps)

    if config.aggregation == "mean":
        entropy = token_entropy.mean(dim=-1, keepdim=True)
    elif config.aggregation == "sum":
        entropy = token_entropy.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unsupported entropy aggregation: {config.aggregation}")

    return entropy
