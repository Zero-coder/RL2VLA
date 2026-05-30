"""Measure EVI parameter count and forward latency."""

from __future__ import annotations

import argparse
import time

import torch

from rl2vla.models.value_heads import RL2VLAValueHead, ValueHeadConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    baseline = RL2VLAValueHead(
        ValueHeadConfig(input_dim=args.input_dim, hidden_dim=args.hidden_dim, conditioning="none")
    ).to(device)
    evi = RL2VLAValueHead(
        ValueHeadConfig(input_dim=args.input_dim, hidden_dim=args.hidden_dim, conditioning="film")
    ).to(device)
    features = torch.randn(args.batch_size, args.input_dim, device=device)
    entropy = torch.rand(args.batch_size, 1, device=device)

    for _ in range(20):
        baseline(features)
        evi(features, entropy)

    start = time.perf_counter()
    for _ in range(args.steps):
        baseline(features)
    baseline_time = (time.perf_counter() - start) / args.steps

    start = time.perf_counter()
    for _ in range(args.steps):
        evi(features, entropy)
    evi_time = (time.perf_counter() - start) / args.steps

    print(
        {
            "baseline_params": count_params(baseline),
            "evi_params": count_params(evi),
            "extra_params": count_params(evi) - count_params(baseline),
            "baseline_latency_s": baseline_time,
            "evi_latency_s": evi_time,
            "extra_latency_s": evi_time - baseline_time,
        }
    )


if __name__ == "__main__":
    main()
