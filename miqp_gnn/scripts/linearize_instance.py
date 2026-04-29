"""Linearize MIQP instances via PLA / McCormick strategies.

Reads raw instances from data/raw, applies the strategy defined by CLI/config,
outputs MILP files under data/linearized along with auxiliary-variable metadata and
linearization error summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "configs" / "datasets.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linearize MIQP instances")
    parser.add_argument("dataset", help="Dataset name, e.g., portfolio_small")
    parser.add_argument("split", choices=["train", "valid", "test"], help="Data split")
    parser.add_argument("--method", choices=["pla", "mccormick", "adaptive"], default="pla")
    parser.add_argument("--segments", type=int, default=10, help="PLA segments when method=pla")
    parser.add_argument("--max-workers", type=int, default=1)
    return parser.parse_args()


def load_linearization_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)["linearization"]


def main() -> None:
    args = parse_args()
    raw_dir = DATA_DIR / "raw" / args.dataset / args.split
    out_dir = DATA_DIR / "linearized" / args.dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = DATA_DIR / "metadata" / args.dataset
    meta_dir.mkdir(parents=True, exist_ok=True)

    raw_instances = sorted(raw_dir.glob("*.json")) + sorted(raw_dir.glob("*.mps"))
    if not raw_instances:
        print(f"[MIQP] No raw instances found in {raw_dir}; generate them first.")
        return

    lin_cfg = load_linearization_config()
    lin_meta_path = meta_dir / f"linearization_{args.split}.jsonl"

    print(f"[MIQP] Linearizing {len(raw_instances)} instances with method={args.method}...")
    print("[MIQP] Placeholder implementation only — integrate solver APIs later.")

    with lin_meta_path.open("a", encoding="utf-8") as fp:
        for inst_path in raw_instances:
            instance_id = inst_path.stem
            lin_path = out_dir / f"{instance_id}_{args.method}.lp"
            lin_path.touch(exist_ok=True)
            metadata = {
                "instance_id": instance_id,
                "method": args.method,
                "segments": args.segments,
                "max_workers": args.max_workers,
                "linearization_config": lin_cfg,
                "status": "TODO:implement",
            }
            fp.write(json.dumps(metadata, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
