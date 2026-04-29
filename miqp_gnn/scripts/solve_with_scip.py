"""Solve generated MIQP instances with PySCIPOpt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyscipopt import Model, quicksum

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SOLVE_DIR = ROOT / "experiments" / "scip_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve MIQP instances with SCIP")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--instance-id", default=None, help="Specific instance ID (default: iterate over split)")
    parser.add_argument("--time-limit", type=float, default=60.0)
    return parser.parse_args()


def load_instance_paths(dataset: str, split: str, instance_id: str | None) -> list[Path]:
    folder = RAW_DIR / dataset / split
    if instance_id:
        path = folder / f"{instance_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No instances found in {folder}")
    return json_files


def build_model(data: dict) -> Model:
    model = Model("miqp")
    n = data.get("n_cont", len(data.get("bounds", [])))
    bounds = {}
    for b in data.get("bounds", []):
        if isinstance(b, dict):
            var_idx = b.get("var")
            bounds[var_idx] = (b.get("lb", 0.0), b.get("ub", 1.0))
        else:
            var_idx, lb, ub = b
            bounds[var_idx] = (lb, ub)
    int_vars = set(data.get("integer_variables", []))

    vars_list = []
    for i in range(n):
        lb, ub = bounds.get(i, (0.0, 1.0))
        vtype = "INTEGER" if i in int_vars else "CONTINUOUS"
        var = model.addVar(name=f"x_{i}", lb=lb, ub=ub, vtype=vtype)
        vars_list.append(var)

    quad = data.get("quadratic_objective", {})
    Q = quad.get("Q")
    q = quad.get("q", [0.0] * n)
    lin_expr = quicksum(q[i] * vars_list[i] for i in range(min(len(q), n)))
    quad_expr = 0
    if Q:
        for i in range(len(Q)):
            for j in range(len(Q[i])):
                coef = Q[i][j]
                if abs(coef) > 1e-9:
                    quad_expr += coef * vars_list[i] * vars_list[j]
    objvar = model.addVar(name="objvar")
    model.addCons(objvar == lin_expr + quad_expr)
    model.setObjective(objvar, sense="minimize")

    for constr in data.get("constraints", []):
        if "quad_matrix" in constr:
            quad_mat = constr["quad_matrix"]
            expr = quicksum(
                quad_mat[i][j] * vars_list[i] * vars_list[j]
                for i in range(len(quad_mat))
                for j in range(len(quad_mat[i]))
            )
            model.addCons(expr <= constr.get("rhs", 0.0), name=constr.get("name"))
        else:
            coefs = constr.get("coefs", [0.0] * n)
            expr = sum(coefs[i] * vars_list[i] for i in range(min(len(coefs), n)))
            sense = constr.get("sense", "<=")
            rhs = constr.get("rhs", 0.0)
            if sense == "<=":
                model.addCons(expr <= rhs, name=constr.get("name"))
            elif sense == ">=":
                model.addCons(expr >= rhs, name=constr.get("name"))
            else:
                model.addCons(expr == rhs, name=constr.get("name"))
    return model


def main() -> None:
    args = parse_args()
    paths = load_instance_paths(args.dataset, args.split, args.instance_id)
    SOLVE_DIR.mkdir(parents=True, exist_ok=True)

    for path in paths:
        instance_id = path.stem
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        model = build_model(data)
        model.setParam("limits/time", args.time_limit)
        model.optimize()

        status = model.getStatus()
        obj = model.getObjVal() if model.getNSols() > 0 else None
        out_path = SOLVE_DIR / f"{instance_id}_solve.txt"
        with out_path.open("w", encoding="utf-8") as fp:
            fp.write(f"status: {status}\n")
            fp.write(f"objective: {obj}\n")
        print(f"[MIQP] {instance_id}: status={status}, objective={obj}")
        if status == "optimal" or model.getNSols() > 0:
            break


if __name__ == "__main__":
    main()
