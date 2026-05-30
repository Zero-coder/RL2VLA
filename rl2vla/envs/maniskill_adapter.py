"""Environment adapter interfaces for ManiSkill/SimplerEnv-style tasks."""

from __future__ import annotations

from typing import Protocol


class VectorEnv(Protocol):
    """Minimal vectorized environment protocol used by training scripts."""

    num_envs: int

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError


def make_env_from_config(config: dict):
    """Create an environment from config.

    This scaffold intentionally avoids importing ManiSkill at module import time.
    Install ManiSkill/SimplerEnv and replace this function with the benchmark
    adapter used in your experiments.
    """

    backend = config.get("backend", "dummy")
    if backend != "dummy":
        raise NotImplementedError(
            "Install the target benchmark and implement make_env_from_config for "
            f"backend={backend!r}."
        )
    return None
