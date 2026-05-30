"""Entropy-aware Value Injection modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EVIConfig:
    """Configuration for FiLM-based entropy-aware value injection."""

    value_dim: int
    entropy_dim: int = 1
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "silu"
    detach_descriptor: bool = True
    gamma_init: float = 1.0
    beta_init: float = 0.0


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class EntropyFiLM(nn.Module):
    """Map entropy descriptors to FiLM parameters for critic features."""

    def __init__(self, config: EVIConfig) -> None:
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.entropy_dim
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(_activation(config.activation))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, 2 * config.value_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            with torch.no_grad():
                final.bias[: self.config.value_dim].fill_(self.config.gamma_init)
                final.bias[self.config.value_dim :].fill_(self.config.beta_init)

    def forward(self, value_features: torch.Tensor, entropy_descriptor: torch.Tensor) -> torch.Tensor:
        """Apply entropy-conditioned FiLM to critic features."""

        if value_features.ndim != 2:
            raise ValueError("value_features must have shape [B, D]")
        if entropy_descriptor.ndim != 2:
            raise ValueError("entropy_descriptor must have shape [B, E]")
        if self.config.detach_descriptor:
            entropy_descriptor = entropy_descriptor.detach()

        gamma_beta = self.net(entropy_descriptor.to(dtype=value_features.dtype))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * value_features + beta
