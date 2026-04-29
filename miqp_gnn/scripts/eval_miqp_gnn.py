"""Evaluate trained MIQP HGT models on held-out datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "datasets"
EXPERIMENT_DIR = ROOT / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIQP HGT model")
    parser.add_argument("dataset")
    parser.add_argument("--method", choices=["pla", "mccormick", "adaptive"], default="pla")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--metrics", nargs="*", default=["gap_closed", "delta_z_miqp", "epsilon"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = DATA_DIR / args.dataset
    test_file = dataset_dir / f"test_{args.method}.npz"
    if not test_file.exists():
        raise FileNotFoundError(f"Missing dataset file {test_file}; build it first.")

    report_dir = EXPERIMENT_DIR / f"eval_{args.dataset}_{args.method}"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "metrics.json"

    print(f"[MIQP] Evaluating placeholder checkpoint {args.checkpoint}")
    print(f"[MIQP] Metrics: {args.metrics} → {report_path}")
    report_path.write_text("{}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
