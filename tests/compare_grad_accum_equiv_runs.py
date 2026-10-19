#!/usr/bin/env python3
"""Compare disk-backed full-batch and chunked gradient-accumulation runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full and chunked Megatron grad-accum artifacts")
    parser.add_argument("--full", required=True, help="Output directory from --grad-test-mode full")
    parser.add_argument("--chunked", required=True, help="Output directory from --grad-test-mode chunked")
    parser.add_argument("--atol", type=float, default=3e-4, help="Absolute tensor tolerance")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tensor tolerance")
    parser.add_argument(
        "--require-moe-aux",
        action="store_true",
        help="Require non-zero MoE router gradients and recorded auxiliary-loss metrics.",
    )
    parser.add_argument(
        "--require-mtp",
        action="store_true",
        help="Require non-zero MTP gradients and recorded MTP-loss metrics.",
    )
    return parser.parse_args()


def _ranks(directory: str) -> list[int]:
    ranks = []
    for path in glob.glob(os.path.join(directory, "grads_rank*.npz")):
        match = re.search(r"grads_rank(\d+)\.npz$", path)
        if match:
            ranks.append(int(match.group(1)))
    if not ranks:
        raise RuntimeError(f"No gradient artifacts in {directory}")
    return sorted(ranks)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as artifact:
        return {name: artifact[name] for name in artifact.files}


def _compare_tensors(label: str, full: Dict[str, np.ndarray], chunked: Dict[str, np.ndarray], atol: float, rtol: float) -> None:
    if full.keys() != chunked.keys():
        raise RuntimeError(f"{label} keys differ: full-only={sorted(full.keys() - chunked.keys())}, "
                           f"chunked-only={sorted(chunked.keys() - full.keys())}")
    max_diff = 0.0
    worst_name = ""
    for name in full:
        if not np.isfinite(full[name]).all() or not np.isfinite(chunked[name]).all():
            raise RuntimeError(f"{label} contains non-finite values for {name}")
        diff = float(np.max(np.abs(full[name] - chunked[name])))
        tolerance = atol + rtol * float(np.max(np.abs(chunked[name])))
        if diff > tolerance:
            raise RuntimeError(f"{label} mismatch for {name}: {diff} > {tolerance}")
        if diff > max_diff:
            max_diff, worst_name = diff, name
    print(f"{label}: max |delta|={max_diff:.3e} ({worst_name})")


def _check_loss_paths(grads: Dict[str, np.ndarray], rank: int, require_moe_aux: bool, require_mtp: bool) -> None:
    groups = []
    if require_moe_aux:
        groups.append(("MoE router / auxiliary-loss", [name for name in grads if ".router." in name]))
    if require_mtp:
        groups.append(("MTP", [name for name in grads if ".mtp." in name]))
    for label, names in groups:
        if not names:
            raise RuntimeError(f"rank {rank} has no {label} gradients")
        if not any(np.any(grads[name] != 0) for name in names):
            raise RuntimeError(f"rank {rank} has only zero {label} gradients")


def _compare_metrics(
    full_path: str,
    chunked_path: str,
    atol: float,
    rtol: float,
    require_moe_aux: bool,
    require_mtp: bool,
) -> None:
    with open(full_path, encoding="utf-8") as f:
        full = json.load(f)
    with open(chunked_path, encoding="utf-8") as f:
        chunked = json.load(f)
    if not full["update_successful"] or not chunked["update_successful"]:
        raise RuntimeError("An optimizer step did not complete")
    path_names = []
    if require_moe_aux:
        path_names.append("moe")
    if require_mtp:
        path_names.append("mtp")
    for path_name in path_names:
        full_values = np.asarray(
            list(full["loss_metrics"][path_name].values()) if path_name == "moe" else full["loss_metrics"][path_name],
            dtype=np.float64,
        )
        chunked_values = np.asarray(
            list(chunked["loss_metrics"][path_name].values()) if path_name == "moe" else chunked["loss_metrics"][path_name],
            dtype=np.float64,
        )
        if full_values.size == 0 or chunked_values.size == 0:
            raise RuntimeError(f"Missing recorded {path_name} loss metrics")
        if full_values.shape != chunked_values.shape or not np.allclose(full_values, chunked_values, atol=atol, rtol=rtol):
            raise RuntimeError(f"{path_name} loss metrics differ")


def main() -> None:
    args = _parse_args()
    full_ranks, chunked_ranks = _ranks(args.full), _ranks(args.chunked)
    if full_ranks != chunked_ranks:
        raise RuntimeError(f"Rank artifacts differ: full={full_ranks}, chunked={chunked_ranks}")
    for rank in full_ranks:
        full_grads = _load_npz(os.path.join(args.full, f"grads_rank{rank}.npz"))
        chunked_grads = _load_npz(os.path.join(args.chunked, f"grads_rank{rank}.npz"))
        _check_loss_paths(full_grads, rank, args.require_moe_aux, args.require_mtp)
        _check_loss_paths(chunked_grads, rank, args.require_moe_aux, args.require_mtp)
        _compare_tensors(f"rank {rank} gradients", full_grads, chunked_grads, args.atol, args.rtol)
        _compare_tensors(
            f"rank {rank} parameters after one update",
            _load_npz(os.path.join(args.full, f"params_rank{rank}.npz")),
            _load_npz(os.path.join(args.chunked, f"params_rank{rank}.npz")),
            args.atol,
            args.rtol,
        )
        _compare_metrics(
            os.path.join(args.full, f"summary_rank{rank}.json"),
            os.path.join(args.chunked, f"summary_rank{rank}.json"),
            args.atol,
            args.rtol,
            args.require_moe_aux,
            args.require_mtp,
        )
    print("Full-batch and chunked gradient accumulation are equivalent.")


if __name__ == "__main__":
    main()