"""Export compact Markdown tables from experiment CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV is empty")
    columns = rows[0].keys()
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join(row[col] for col in columns) + " |")


if __name__ == "__main__":
    main()
