"""Config-driven PPO launcher for RL2VLA.

This script provides the command-line surface for the paper code. The actual
large-scale OpenVLA/ManiSkill integration should be implemented inside the
policy and environment adapters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rl2vla.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    raise NotImplementedError(
        "Connect TokenizedVLAPolicy and ManiSkill/SimplerEnv adapters before running full PPO."
    )


if __name__ == "__main__":
    main()
