"""Value heads used by RL2VLA ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from rl2vla.models.evi import EVIConfig, EntropyFiLM


ValueConditioning = Literal["none", "film", "concat"]


@dataclass(frozen=True)
class ValueHeadConfig:
    input_dim: int
    hidden_dim: int = 1024
    conditioning: ValueConditioning = "film"
    evi_hidden_dim: int = 256
    evi_layers: int = 2
    detach_descriptor: bool = True


class RL2VLAValueHead(nn.Module):
    """Critic head with optional entropy conditioning."""

    def __init__(self, config: ValueHeadConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.input_dim
        trunk_input = input_dim + 1 if config.conditioning == "concat" else input_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.film: EntropyFiLM | None = None
        if config.conditioning == "film":
            self.film = EntropyFiLM(
                EVIConfig(
                    value_dim=config.hidden_dim,
                    hidden_dim=config.evi_hidden_dim,
                    num_layers=config.evi_layers,
                    detach_descriptor=config.detach_descriptor,
                )
            )
        self.value = nn.Linear(config.hidden_dim, 1)

    def forward(self, state_features: torch.Tensor, entropy_descriptor: torch.Tensor | None = None) -> torch.Tensor:
        if self.config.conditioning in {"film", "concat"} and entropy_descriptor is None:
            raise ValueError("entropy_descriptor is required for conditioned value heads")

        if self.config.conditioning == "concat":
            if self.config.detach_descriptor:
                entropy_descriptor = entropy_descriptor.detach()
            state_features = torch.cat([state_features, entropy_descriptor.to(state_features.dtype)], dim=-1)

        hidden = self.trunk(state_features)
        if self.film is not None:
            hidden = self.film(hidden, entropy_descriptor)
        return self.value(hidden).squeeze(-1)
