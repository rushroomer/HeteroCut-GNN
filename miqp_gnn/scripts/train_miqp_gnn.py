"""Train the MIQP-aware HGT model using placeholder datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "datasets"
EXPERIMENT_DIR = ROOT / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIQP HGT model")
    parser.add_argument("dataset", help="Dataset name (portfolio_small, qap_small, ...)")
    parser.add_argument("--method", choices=["pla", "mccormick", "adaptive"], default="pla")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    args = parse_args()
    dataset_dir = DATA_DIR / args.dataset
    train_file = dataset_dir / f"train_{args.method}.npz"
    valid_file = dataset_dir / f"valid_{args.method}.npz"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing dataset file {train_file}; run build_graph_dataset.py")

    train_data = load_dataset(train_file)
    valid_data = load_dataset(valid_file) if valid_file.exists() else None

    EXPERIMENT_DIR.mkdir(exist_ok=True)
    run_dir = EXPERIMENT_DIR / f"train_{args.dataset}_{args.method}"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / "training.log"

    features = train_data["graph_features"]
    stats = {
        "num_instances": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "avg_vars": float(features[:, 0].mean()),
        "avg_int": float(features[:, 1].mean()),
        "avg_constraints": float(features[:, 2].mean()),
        "avg_q_density": float(features[:, 3].mean()),
    }

    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}\n")
        fp.write(f"Device request: {args.device}\n")
        fp.write(f"Train stats: {stats}\n")
        if valid_data is None:
            fp.write("Warning: validation split missing, proceeding with train-only placeholder.\n")
        else:
            fp.write(f"Valid instances: {valid_data['graph_features'].shape[0]}\n")
        fp.write("NOTE: Placeholder trainer; integrate real HGT model later.\n")

    print(f"[MIQP] Logged placeholder training stats to {log_path}")


if __name__ == "__main__":
    main()
