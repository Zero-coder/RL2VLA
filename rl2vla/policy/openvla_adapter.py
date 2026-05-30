"""Adapters for OpenVLA-style tokenized action policies.

This module documents the interface RL2VLA expects from a VLA policy while
allowing smoke tests to run without downloading large OpenVLA checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class PolicyOutput:
    action_tokens: torch.Tensor
    action_logits: torch.Tensor
    action_logprobs: torch.Tensor
    action_hidden: torch.Tensor


class TokenizedVLAPolicy(Protocol):
    """Protocol expected by RL2VLA training code."""

    def act(self, batch: dict[str, torch.Tensor]) -> PolicyOutput:
        """Return sampled action tokens and token-level policy outputs."""


class DummyOpenVLAAdapter(torch.nn.Module):
    """Small policy used for smoke tests and interface validation."""

    def __init__(self, obs_dim: int = 64, hidden_dim: int = 128, action_tokens: int = 7, vocab_size: int = 256) -> None:
        super().__init__()
        self.action_tokens = action_tokens
        self.vocab_size = vocab_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.policy = torch.nn.Linear(hidden_dim, action_tokens * vocab_size)

    def act(self, batch: dict[str, torch.Tensor]) -> PolicyOutput:
        obs = batch["obs"]
        hidden = self.encoder(obs)
        logits = self.policy(hidden).view(obs.shape[0], self.action_tokens, self.vocab_size)
        dist = torch.distributions.Categorical(logits=logits)
        tokens = dist.sample()
        logprobs = dist.log_prob(tokens).sum(dim=-1)
        action_hidden = hidden.unsqueeze(1).expand(-1, self.action_tokens, -1).contiguous()
        return PolicyOutput(tokens, logits, logprobs, action_hidden)
