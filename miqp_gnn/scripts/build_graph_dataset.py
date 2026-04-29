"""Convert MIQP JSON instances into placeholder graph tensors for GNN training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CONFIG_PATH = ROOT / "configs" / "miqp_feature_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GNN datasets from MIQP instances")
    parser.add_argument("dataset")
    parser.add_argument("split", choices=["train", "valid", "test"])
    parser.add_argument("--method", choices=["pla", "mccormick", "adaptive"], default="pla")
    parser.add_argument("--output", default=None, help="Optional override for dataset output file")
    return parser.parse_args()


def load_feature_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_instances(dataset: str, split: str) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    raw_dir = RAW_DIR / dataset / split
    if not raw_dir.exists():
        return instances
    for json_file in sorted(raw_dir.glob("*.json")):
        if json_file.stat().st_size == 0:
            continue
        with json_file.open("r", encoding="utf-8") as fp:
            try:
                instances.append(json.load(fp))
            except json.JSONDecodeError:
                print(f"[MIQP] Skipping invalid JSON file {json_file}")
    return instances


def summarize_instance(data: dict[str, Any]) -> tuple[np.ndarray, float, float, float]:
    bounds = data.get("bounds", [])
    n_vars = len(bounds) or data.get("n_cont", 0)
    n_int = len(data.get("integer_variables", []))
    constraints = data.get("constraints", [])
    num_constraints = len(constraints)

    Q = data.get("quadratic_objective", {}).get("Q", [])
    if Q:
        total = len(Q) * len(Q[0])
        nnz = sum(1 for row in Q for val in row if abs(val) > 1e-9)
        q_density = nnz / total if total else 0.0
    else:
        q_density = 0.0

    feature_vec = np.array([n_vars, n_int, num_constraints, q_density], dtype=np.float32)
    delta_z_milp = 0.0  # placeholder labels
    delta_z_miqp = 0.0
    epsilon = 0.0
    return feature_vec, delta_z_milp, delta_z_miqp, epsilon


def main() -> None:
    args = parse_args()
    feature_cfg = load_feature_config()
    out_dir = DATA_DIR / "datasets" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else out_dir / f"{args.split}_{args.method}.npz"

    instances = load_instances(args.dataset, args.split)
    if not instances:
        print(f"[MIQP] No raw instances found for {args.dataset}/{args.split}")
        return

    instance_ids = []
    feature_vectors = []
    delta_milp = []
    delta_miqp = []
    epsilons = []

    for inst in instances:
        feature_vec, dz_milp, dz_miqp, eps = summarize_instance(inst)
        instance_ids.append(inst.get("instance_id", f"{args.dataset}_{len(instance_ids)}"))
        feature_vectors.append(feature_vec)
        delta_milp.append(dz_milp)
        delta_miqp.append(dz_miqp)
        epsilons.append(eps)

    feature_matrix = np.stack(feature_vectors)
    np.savez(
        output_path,
        instance_ids=np.array(instance_ids),
        graph_features=feature_matrix,
        delta_z_milp=np.array(delta_milp, dtype=np.float32),
        delta_z_miqp=np.array(delta_miqp, dtype=np.float32),
        epsilon=np.array(epsilons, dtype=np.float32),
    )
    print(f"[MIQP] Saved placeholder dataset with {len(instance_ids)} instances to {output_path}")

    schema_path = out_dir / "feature_schema.yaml"
    if not schema_path.exists():
        with schema_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(feature_cfg, fp, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
