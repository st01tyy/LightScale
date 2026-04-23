#!/usr/bin/env python3
"""Compare gradient-accumulation artifacts across two runs."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from tests.grad_accum_test_utils import load_json


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _find_ranks(output_dir: str) -> List[int]:
    ranks = []
    for name in os.listdir(output_dir):
        if name.startswith("stats_rank") and name.endswith(".json"):
            rank_str = name[len("stats_rank") : -len(".json")]
            ranks.append(int(rank_str))
    return sorted(ranks)


def _compare_stats(left: Dict, right: Dict, atol: float, rtol: float) -> List[str]:
    errors = []
    left_losses = left.get("losses", [])
    right_losses = right.get("losses", [])
    left_norms = left.get("grad_norms", [])
    right_norms = right.get("grad_norms", [])
    if len(left_losses) != len(right_losses):
        errors.append("loss length mismatch")
    if len(left_norms) != len(right_norms):
        errors.append("grad_norm length mismatch")

    for idx, (lval, rval) in enumerate(zip(left_losses, right_losses), start=1):
        diff = abs(lval - rval)
        tol = atol + rtol * abs(rval)
        if diff > tol:
            errors.append(f"loss step {idx} diff {diff} > tol {tol}")

    for idx, (lval, rval) in enumerate(zip(left_norms, right_norms), start=1):
        diff = abs(lval - rval)
        tol = atol + rtol * abs(rval)
        if diff > tol:
            errors.append(f"grad_norm step {idx} diff {diff} > tol {tol}")

    return errors


def _compare_grads(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Tuple[float, float]:
    if left.keys() != right.keys():
        missing_left = sorted(set(right.keys()) - set(left.keys()))
        missing_right = sorted(set(left.keys()) - set(right.keys()))
        raise RuntimeError(
            "Gradient keys mismatch. "
            f"Missing in left: {missing_left}. Missing in right: {missing_right}."
        )
    max_diff = 0.0
    max_ref = 0.0
    for key in left:
        diff = np.max(np.abs(left[key] - right[key]))
        ref = np.max(np.abs(right[key]))
        max_diff = max(max_diff, float(diff))
        max_ref = max(max_ref, float(ref))
    return max_diff, max_ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare grad-accum run artifacts")
    parser.add_argument("--left", required=True, help="Left run output directory")
    parser.add_argument("--right", required=True, help="Right run output directory")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_ranks = _find_ranks(args.left)
    right_ranks = _find_ranks(args.right)
    if left_ranks != right_ranks:
        raise RuntimeError(f"Rank mismatch: left={left_ranks}, right={right_ranks}")

    failed = False
    for rank in left_ranks:
        left_stats = load_json(os.path.join(args.left, f"stats_rank{rank}.json"))
        right_stats = load_json(os.path.join(args.right, f"stats_rank{rank}.json"))
        stat_errors = _compare_stats(left_stats, right_stats, args.atol, args.rtol)
        if stat_errors:
            failed = True
            print(f"[rank {rank}] Stats mismatch:")
            for err in stat_errors:
                print(f"  - {err}")

        steps = left_stats.get("steps", 0)
        for step in range(1, steps + 1):
            left_grads = _load_npz(os.path.join(args.left, f"main_grads_step{step}_rank{rank}.npz"))
            right_grads = _load_npz(os.path.join(args.right, f"main_grads_step{step}_rank{rank}.npz"))
            max_diff, max_ref = _compare_grads(left_grads, right_grads)
            tol = args.atol + args.rtol * max_ref
            if max_diff > tol:
                failed = True
                print(f"[rank {rank}] Step {step} grad max diff {max_diff} > tol {tol}")

    if failed:
        raise SystemExit(1)
    print("All comparisons passed.")


if __name__ == "__main__":
    main()
