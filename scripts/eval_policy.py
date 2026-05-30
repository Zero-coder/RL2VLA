"""Evaluation launcher for RL2VLA policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rl2vla.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(yaml.safe_dump({"config": config, "checkpoint": str(args.checkpoint)}, sort_keys=False))
    raise NotImplementedError("Implement benchmark-specific evaluation adapter.")


if __name__ == "__main__":
    main()
