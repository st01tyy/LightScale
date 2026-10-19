#!/usr/bin/env python3
"""Compare gradient-accumulation artifacts across two runs."""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np

# This script is executed as `python tests/compare_grad_accum_runs.py`, so its
# containing directory is already on sys.path. A `tests.` package import is not
# reliable because this directory intentionally has no __init__.py.
from grad_accum_test_utils import load_json


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


def _compare_stats(
    left: Dict, right: Dict, atol: float, rtol: float, compare_grad_norms: bool = True
) -> List[str]:
    errors = []
    left_losses = left.get("losses", [])
    right_losses = right.get("losses", [])
    left_norms = left.get("grad_norms", [])
    right_norms = right.get("grad_norms", [])
    if len(left_losses) != len(right_losses):
        errors.append("loss length mismatch")
    if compare_grad_norms and len(left_norms) != len(right_norms):
        errors.append("grad_norm length mismatch")

    for idx, (lval, rval) in enumerate(zip(left_losses, right_losses), start=1):
        if not math.isfinite(lval) or not math.isfinite(rval):
            errors.append(f"loss step {idx} is non-finite: left={lval}, right={rval}")
            continue
        diff = abs(lval - rval)
        tol = atol + rtol * abs(rval)
        if diff > tol:
            errors.append(f"loss step {idx} diff {diff} > tol {tol}")

    if compare_grad_norms:
        for idx, (lval, rval) in enumerate(zip(left_norms, right_norms), start=1):
            if not math.isfinite(lval) or not math.isfinite(rval):
                errors.append(f"grad_norm step {idx} is non-finite: left={lval}, right={rval}")
                continue
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
        if not np.isfinite(left[key]).all() or not np.isfinite(right[key]).all():
            raise RuntimeError(
                f"Non-finite gradient values for parameter '{key}': "
                f"left_finite={np.isfinite(left[key]).all()}, "
                f"right_finite={np.isfinite(right[key]).all()}"
            )
        diff = np.max(np.abs(left[key] - right[key]))
        ref = np.max(np.abs(right[key]))
        max_diff = max(max_diff, float(diff))
        max_ref = max(max_ref, float(ref))
    return max_diff, max_ref


def _check_special_loss_grads(grads: Dict[str, np.ndarray], label: str, require_moe_aux: bool, require_mtp: bool) -> None:
    required_groups = []
    if require_moe_aux:
        required_groups.append(("MoE router / auxiliary-loss", [name for name in grads if ".router." in name]))
    if require_mtp:
        required_groups.append(("MTP", [name for name in grads if ".mtp." in name]))
    for group_name, names in required_groups:
        if not names:
            raise RuntimeError(f"{label} has no {group_name} gradients")
        if not any(np.any(grads[name] != 0) for name in names):
            raise RuntimeError(f"{label} has only zero {group_name} gradients")


def _compare_special_metrics(
    left: Dict, right: Dict, atol: float, rtol: float, require_moe_aux: bool, require_mtp: bool
) -> List[str]:
    errors = []
    for metric_name, required in (("moe", require_moe_aux), ("mtp", require_mtp)):
        if not required:
            continue
        left_metric = left.get("loss_metrics", {}).get(metric_name, {})
        right_metric = right.get("loss_metrics", {}).get(metric_name, {})
        left_values = np.asarray(
            list(left_metric.values()) if metric_name == "moe" else left_metric, dtype=np.float64
        )
        right_values = np.asarray(
            list(right_metric.values()) if metric_name == "moe" else right_metric, dtype=np.float64
        )
        if left_values.size == 0 or right_values.size == 0:
            errors.append(f"missing {metric_name} loss metrics")
        elif left_values.shape != right_values.shape or not np.allclose(
            left_values, right_values, atol=atol, rtol=rtol
        ):
            errors.append(f"{metric_name} loss metrics differ")
    return errors


def _aggregate_grads(output_dir: str, ranks: List[int], step: int) -> Dict[str, np.ndarray]:
    """Sum local parameter gradients from one CP group.

    BSHD uses CP slicing after fixed-length sample padding, while THD slices each
    real sample before packing. Those layouts distribute a logical sample's token
    gradients differently across CP ranks. The optimizer-relevant quantity is the
    CP-group sum, not an individual rank's local gradient buffer.
    """
    aggregate: Dict[str, np.ndarray] = {}
    for rank in ranks:
        grads = _load_npz(os.path.join(output_dir, f"main_grads_step{step}_rank{rank}.npz"))
        if not aggregate:
            aggregate = {name: grad.copy() for name, grad in grads.items()}
            continue
        if aggregate.keys() != grads.keys():
            raise RuntimeError(f"Gradient keys mismatch within CP group {ranks} on rank {rank}")
        for name, grad in grads.items():
            aggregate[name] += grad
    return aggregate


def _compute_grad_norm(grads: Dict[str, np.ndarray]) -> float:
    total = 0.0
    for grad in grads.values():
        total += float(np.sum(np.square(grad, dtype=np.float64)))
    return math.sqrt(total)


def _make_cp_groups(ranks: List[int], cp_size: int) -> List[List[int]]:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if len(ranks) % cp_size != 0:
        raise ValueError(
            f"Number of ranks ({len(ranks)}) must be divisible by cp_size ({cp_size})"
        )
    for expected, rank in enumerate(ranks):
        if rank != expected:
            raise ValueError(
                "CP artifact aggregation requires consecutive global ranks starting at zero, "
                f"got ranks={ranks}"
            )
    return [ranks[start : start + cp_size] for start in range(0, len(ranks), cp_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare grad-accum run artifacts")
    parser.add_argument("--left", required=True, help="Left run output directory")
    parser.add_argument("--right", required=True, help="Right run output directory")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument(
        "--cp-size",
        type=int,
        default=1,
        help="Context-parallel size; gradients are summed within each consecutive CP group before comparison",
    )
    parser.add_argument("--require-moe-aux", action="store_true", help="Require non-zero MoE router gradients.")
    parser.add_argument("--require-mtp", action="store_true", help="Require non-zero MTP gradients.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_ranks = _find_ranks(args.left)
    right_ranks = _find_ranks(args.right)
    if left_ranks != right_ranks:
        raise RuntimeError(f"Rank mismatch: left={left_ranks}, right={right_ranks}")
    cp_groups = _make_cp_groups(left_ranks, args.cp_size)

    failed = False
    for rank in left_ranks:
        left_stats = load_json(os.path.join(args.left, f"stats_rank{rank}.json"))
        right_stats = load_json(os.path.join(args.right, f"stats_rank{rank}.json"))
        if not left_stats.get("update_successful") or not right_stats.get("update_successful"):
            failed = True
            print(f"[rank {rank}] Optimizer update did not complete in both runs")
        stat_errors = _compare_stats(
            left_stats, right_stats, args.atol, args.rtol, compare_grad_norms=args.cp_size == 1
        )
        stat_errors.extend(
            _compare_special_metrics(
                left_stats, right_stats, args.atol, args.rtol, args.require_moe_aux, args.require_mtp
            )
        )
        if stat_errors:
            failed = True
            print(f"[rank {rank}] Stats mismatch:")
            for err in stat_errors:
                print(f"  - {err}")

    steps = load_json(os.path.join(args.left, f"stats_rank{left_ranks[0]}.json")).get("steps", 0)
    for cp_group in cp_groups:
        for step in range(1, steps + 1):
            left_grads = _aggregate_grads(args.left, cp_group, step)
            right_grads = _aggregate_grads(args.right, cp_group, step)
            _check_special_loss_grads(left_grads, f"left CP group {cp_group}", args.require_moe_aux, args.require_mtp)
            _check_special_loss_grads(right_grads, f"right CP group {cp_group}", args.require_moe_aux, args.require_mtp)
            left_norm = _compute_grad_norm(left_grads)
            right_norm = _compute_grad_norm(right_grads)
            norm_diff = abs(left_norm - right_norm)
            norm_tol = args.atol + args.rtol * abs(right_norm)
            if norm_diff > norm_tol:
                failed = True
                print(
                    f"[CP group {cp_group}] Step {step} grad norm diff "
                    f"{norm_diff} > tol {norm_tol}"
                )
            max_diff, max_ref = _compare_grads(left_grads, right_grads)
            tol = args.atol + args.rtol * max_ref
            if max_diff > tol:
                failed = True
                print(f"[CP group {cp_group}] Step {step} grad max diff {max_diff} > tol {tol}")

    for rank in left_ranks:
        left_params = _load_npz(os.path.join(args.left, f"params_step1_rank{rank}.npz"))
        right_params = _load_npz(os.path.join(args.right, f"params_step1_rank{rank}.npz"))
        max_diff, max_ref = _compare_grads(left_params, right_params)
        tol = args.atol + args.rtol * max_ref
        if max_diff > tol:
            failed = True
            print(f"[rank {rank}] Updated parameter max diff {max_diff} > tol {tol}")

    if failed:
        raise SystemExit(1)
    print("All comparisons passed.")


if __name__ == "__main__":
    main()
