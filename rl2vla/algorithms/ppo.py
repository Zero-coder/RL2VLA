"""PPO and GAE utilities for RL2VLA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    entropy_value_loss_scale: float = 0.0


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns."""

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[t].float()
        next_values = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    config: PPOConfig,
) -> dict[str, torch.Tensor]:
    """Compute PPO actor, critic, and entropy losses."""

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    log_ratio = new_logprobs - old_logprobs
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_error = F.mse_loss(values, returns, reduction="none")
    if config.entropy_value_loss_scale > 0:
        value_weight = 1.0 + config.entropy_value_loss_scale * entropy.detach().squeeze(-1)
        value_error = value_error * value_weight
    value_loss = value_error.mean()
    entropy_loss = -entropy.mean()
    total = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

    approx_kl = ((ratio - 1.0) - log_ratio).mean().detach()
    clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean().detach()
    return {
        "loss": total,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.mean().detach(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
