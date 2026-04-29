"""MIQP instance generator using random parameter sampling (GCNN-style)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
METADATA_DIR = ROOT / "data" / "metadata"


@dataclass
class ProblemConfig:
    name: str
    n_cont: tuple[int, int]
    n_int: tuple[int, int]
    q_density: tuple[float, float]
    bound_range: tuple[float, float]
    rhs_range: tuple[float, float]


PROBLEM_CONFIGS: dict[str, ProblemConfig] = {
    "portfolio_small": ProblemConfig("portfolio_small", (50, 100), (20, 50), (0.05, 0.15), (0.0, 1.0), (0.5, 1.5)),
    "portfolio_medium": ProblemConfig("portfolio_medium", (100, 200), (50, 100), (0.05, 0.15), (0.0, 1.0), (0.5, 1.5)),
    "qap_small": ProblemConfig("qap_small", (100, 400), (100, 400), (0.6, 0.9), (0.0, 1.0), (10, 100)),
    "qap_medium": ProblemConfig("qap_medium", (400, 900), (400, 900), (0.7, 0.95), (0.0, 1.0), (10, 100)),
    "qcqp": ProblemConfig("qcqp", (50, 150), (30, 80), (0.2, 0.4), (0.0, 1.0), (0.5, 2.0)),
    "boxqp": ProblemConfig("boxqp", (20, 60), (0, 0), (0.6, 0.9), (0.0, 1.0), (0.5, 2.0)),
}


def random_symmetric_matrix(size: int, density: float, rng: np.random.Generator) -> np.ndarray:
    mat = rng.random((size, size))
    mask = rng.random((size, size)) < density
    mat *= mask
    mat = np.triu(mat)
    return mat + mat.T - np.diag(np.diag(mat))


def generate_portfolio(cfg: ProblemConfig, rng: np.random.Generator) -> dict:
    n = rng.integers(*cfg.n_cont)
    p = rng.integers(*cfg.n_int) if cfg.n_int[1] > 0 else 0
    q_density = rng.uniform(*cfg.q_density)
    Q = random_symmetric_matrix(n, q_density, rng) * 0.2
    mu = rng.uniform(-0.3, -0.1, size=n)
    rhs = 1.0
    asset_bounds = [[i, 0.0, 1.0] for i in range(n)]
    constraints = [
        {"name": "budget", "sense": "=", "rhs": rhs, "coefs": [float(rng.random()) for _ in range(n)]},
    ]
    return {
        "instance_type": "portfolio",
        "n_cont": int(n),
        "n_int": int(p),
        "quadratic_objective": {"Q": Q.tolist(), "q": mu.tolist()},
        "constraints": constraints,
        "bounds": asset_bounds,
        "integer_variables": list(range(n - p, n)) if p > 0 else [],
    }


def generate_qap(cfg: ProblemConfig, rng: np.random.Generator) -> dict:
    n = rng.integers(*cfg.n_cont)
    flow = random_symmetric_matrix(n, rng.uniform(*cfg.q_density), rng)
    dist = random_symmetric_matrix(n, rng.uniform(*cfg.q_density), rng)
    Q = (flow @ dist).tolist()
    return {
        "instance_type": "qap",
        "n_cont": int(n),
        "n_int": int(n),
        "quadratic_objective": {"Q": Q, "q": [0.0] * n},
        "constraints": [],
        "bounds": [[i, 0, 1] for i in range(n)],
        "integer_variables": list(range(n)),
    }


def generate_qcqp(cfg: ProblemConfig, rng: np.random.Generator) -> dict:
    n = rng.integers(*cfg.n_cont)
    m = rng.integers(20, 60)
    Q = random_symmetric_matrix(n, rng.uniform(*cfg.q_density), rng)
    constraints = [
        {
            "name": f"quad_constr_{i}",
            "sense": "<=",
            "rhs": float(rng.uniform(*cfg.rhs_range)),
            "quad_matrix": random_symmetric_matrix(n, rng.uniform(0.1, 0.3), rng).tolist(),
        }
        for i in range(m)
    ]
    return {
        "instance_type": "qcqp",
        "n_cont": int(n),
        "n_int": int(rng.integers(*cfg.n_int)),
        "quadratic_objective": {"Q": Q.tolist(), "q": rng.uniform(-1, 1, size=n).tolist()},
        "constraints": constraints,
        "bounds": [[i, 0, 1] for i in range(n)],
        "integer_variables": list(range(cfg.n_int[0])),
    }


def generate_boxqp(cfg: ProblemConfig, rng: np.random.Generator) -> dict:
    n = rng.integers(*cfg.n_cont)
    Q = random_symmetric_matrix(n, rng.uniform(*cfg.q_density), rng)
    c = rng.uniform(-1, 1, size=n)
    bounds = [[i, 0, 1] for i in range(n)]
    return {
        "instance_type": "boxqp",
        "n_cont": int(n),
        "n_int": 0,
        "quadratic_objective": {"Q": Q.tolist(), "q": c.tolist()},
        "constraints": [],
        "bounds": bounds,
        "integer_variables": [],
    }


def get_generator(name: str) -> Callable[[ProblemConfig, np.random.Generator], dict]:
    if "portfolio" in name:
        return generate_portfolio
    if "qap" in name:
        return generate_qap
    if name == "qcqp":
        return generate_qcqp
    if name == "boxqp":
        return generate_boxqp
    raise ValueError(f"Unsupported dataset: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random MIQP instance generator")
    parser.add_argument("--dataset", required=True, choices=list(PROBLEM_CONFIGS.keys()))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PROBLEM_CONFIGS[args.dataset]
    rng = np.random.default_rng(args.seed)
    generator = get_generator(args.dataset)

    out_dir = RAW_DIR / args.dataset / args.split
    meta_dir = METADATA_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = meta_dir / f"{args.split}_metadata.jsonl"

    for idx in range(args.num_instances):
        instance_id = f"{args.dataset}_{args.split}_{idx:05d}"
        instance = generator(cfg, rng)
        instance["instance_id"] = instance_id
        with (out_dir / f"{instance_id}.json").open("w", encoding="utf-8") as fp:
            json.dump(instance, fp)
        metadata = {
            "instance_id": instance_id,
            "dataset": args.dataset,
            "split": args.split,
            "n_cont": instance.get("n_cont"),
            "n_int": instance.get("n_int"),
            "q_density": cfg.q_density,
        }
        with metadata_path.open("a", encoding="utf-8") as meta_fp:
            meta_fp.write(json.dumps(metadata) + "\n")
    print(f"[MIQP] Generated {args.num_instances} {args.dataset}/{args.split} instances.")


if __name__ == "__main__":
    main()
