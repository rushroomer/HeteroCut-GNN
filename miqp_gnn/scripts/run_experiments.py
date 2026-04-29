"""Experiment orchestrator producing results comparable to milp_gnn.tex tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments.yaml"
EXPERIMENT_DIR = ROOT / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIQP experiment suites")
    parser.add_argument("--experiments", nargs="*", default=[
        "linearization_comparison",
        "cut_selection_benchmark",
        "end_to_end_performance",
        "ablation",
        "generalization",
        "efficiency_scaling",
    ])
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    EXPERIMENT_DIR.mkdir(exist_ok=True)

    for exp_name in args.experiments:
        if exp_name not in cfg["experiments"]:
            raise KeyError(f"Experiment {exp_name} missing in configs/experiments.yaml")
        exp_cfg = cfg["experiments"][exp_name]
        out_path = EXPERIMENT_DIR / f"{exp_name}.json"
        print(f"[MIQP] Scheduling experiment {exp_name} → {out_path}")
        placeholder = {
            "experiment": exp_name,
            "config": exp_cfg,
            "status": "TODO:implement",
        }
        out_path.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
