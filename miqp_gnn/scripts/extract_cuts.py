"""Extract candidate cut planes from linearized MILP instances.

For each linearized model, run the solver in root-node mode to gather
candidate cuts, compute MILP and MIQP bound improvements, and sample
cut subsets with labels (\Delta z^{MILP}, \Delta z^{MIQP}, \epsilon).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate cuts from linearized MILP")
    parser.add_argument("dataset")
    parser.add_argument("split", choices=["train", "valid", "test"])
    parser.add_argument("--method", choices=["pla", "mccormick", "adaptive"], default="pla")
    parser.add_argument("--max-cuts", type=int, default=120)
    parser.add_argument("--subset-samples", type=int, default=15)
    parser.add_argument("--solver", choices=["cplex", "gurobi", "scip"], default="cplex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lin_dir = DATA_DIR / "linearized" / args.dataset / args.split
    out_dir = DATA_DIR / "cuts" / args.dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    lin_instances = sorted(lin_dir.glob(f"*_{args.method}.lp"))
    if not lin_instances:
        print(f"[MIQP] No linearized instances found in {lin_dir}")
        return

    print(f"[MIQP] Extracting cuts via {args.solver} (placeholder implementation)...")
    for inst in lin_instances:
        instance_id = inst.stem
        cut_path = out_dir / f"{instance_id}_cuts.jsonl"
        with cut_path.open("w", encoding="utf-8") as fp:
            header = {
                "instance_id": instance_id,
                "method": args.method,
                "solver": args.solver,
                "max_cuts": args.max_cuts,
                "subset_samples": args.subset_samples,
                "status": "TODO:solver_integration",
            }
            fp.write(json.dumps(header, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
