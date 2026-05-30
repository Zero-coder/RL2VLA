"""Metric helpers for RL2VLA experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n
        self.history.append(float(value))

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count
