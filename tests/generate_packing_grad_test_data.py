#!/usr/bin/env python3
"""Generate matched BSHD and THD packing test datasets.

The two output files contain exactly the same logical samples, but are intended for
different layout runners:
1. `packing_bshd.npz` stores sample-aligned rows for the BSHD baseline.
2. `packing_thd.npz` also stores sample-aligned rows, and the THD runner later turns
    them into local packed buckets using the real runtime order:
    DP slice -> per-sample CP slice -> local packing.

This design keeps the source logical samples identical across the two modes and also
allows the THD path to support `cp_size > 1`, where packing must happen after CP slicing.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np


PAD_TOKEN_ID = 0


def _plan_equal_sample_dp_shards(sample_seq_lens: List[int], dp_world_size: int) -> List[List[int]]:
    total_samples = len(sample_seq_lens)
    if total_samples % dp_world_size != 0:
        raise ValueError(
            f"num_samples={total_samples} must be divisible by dp_world_size={dp_world_size}"
        )

    per_shard_samples = total_samples // dp_world_size
    shard_indices: List[List[int]] = [[] for _ in range(dp_world_size)]
    shard_total_lens = [0 for _ in range(dp_world_size)]

    for sample_idx in sorted(range(total_samples), key=lambda idx: sample_seq_lens[idx], reverse=True):
        candidate_shards = [
            shard_id for shard_id in range(dp_world_size)
            if len(shard_indices[shard_id]) < per_shard_samples
        ]
        target_shard = min(
            candidate_shards,
            key=lambda shard_id: (len(shard_indices[shard_id]), shard_total_lens[shard_id], shard_id),
        )
        shard_indices[target_shard].append(sample_idx)
        shard_total_lens[target_shard] += sample_seq_lens[sample_idx]

    return shard_indices


def _build_sample_arrays(rng: np.random.Generator, sample_seq_len: int, seq_length: int, vocab_size: int) -> Dict[str, np.ndarray]:
    token_storage = rng.integers(1, vocab_size, size=(sample_seq_len + 1,), dtype=np.int64)

    input_ids = np.full((seq_length,), PAD_TOKEN_ID, dtype=np.int64)
    labels = np.full((seq_length,), PAD_TOKEN_ID, dtype=np.int64)
    loss_mask = np.zeros((seq_length,), dtype=np.float32)

    input_ids[:sample_seq_len] = token_storage[:-1]
    labels[:sample_seq_len] = token_storage[1:]
    loss_mask[:sample_seq_len] = 1.0

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "sample_seq_len": np.int64(sample_seq_len),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matched BSHD/THD packing grad-accum test data")
    parser.add_argument("--output-dir", required=True, help="Directory to store the generated .npz files")
    parser.add_argument("--num-samples", type=int, default=32, help="Total number of logical samples")
    parser.add_argument("--dp-world-size", type=int, default=8, help="DP world size used by the 8-GPU launch script")
    parser.add_argument("--cp-world-size", type=int, default=1, help="CP world size used by the 8-GPU launch script")
    parser.add_argument("--seq-length", type=int, default=256, help="Target seq_length used by both BSHD and THD runs")
    parser.add_argument("--min-sample-seq-len", type=int, default=32, help="Minimum logical sample length after shift")
    parser.add_argument("--max-sample-seq-len", type=int, default=192, help="Maximum logical sample length after shift")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size for synthetic token generation")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sample_seq_len > args.seq_length:
        raise ValueError("max-sample-seq-len must not exceed seq-length")
    if args.cp_world_size < 1:
        raise ValueError("cp-world-size must be >= 1")
    if args.seq_length % args.cp_world_size != 0:
        raise ValueError("seq-length must be divisible by cp-world-size")
    if args.cp_world_size > 1 and args.seq_length % (2 * args.cp_world_size) != 0:
        raise ValueError("when cp-world-size > 1, seq-length must be divisible by 2 * cp-world-size")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cp_multiple = 1 if args.cp_world_size == 1 else args.cp_world_size * 2

    raw_sample_seq_lens = rng.integers(
        args.min_sample_seq_len,
        args.max_sample_seq_len + 1,
        size=(args.num_samples,),
        dtype=np.int64,
    )
    sample_seq_lens = ((raw_sample_seq_lens + cp_multiple - 1) // cp_multiple) * cp_multiple
    sample_seq_lens = np.minimum(sample_seq_lens, args.seq_length)
    samples = [
        _build_sample_arrays(rng, int(sample_seq_len), args.seq_length, args.vocab_size)
        for sample_seq_len in sample_seq_lens.tolist()
    ]

    shard_plan = _plan_equal_sample_dp_shards(sample_seq_lens.tolist(), args.dp_world_size)
    ordered_sample_indices = [sample_idx for shard in shard_plan for sample_idx in shard]
    per_shard_samples = args.num_samples // args.dp_world_size

    bshd_input_ids = np.stack([samples[idx]["input_ids"] for idx in ordered_sample_indices], axis=0)
    bshd_labels = np.stack([samples[idx]["labels"] for idx in ordered_sample_indices], axis=0)
    bshd_loss_mask = np.stack([samples[idx]["loss_mask"] for idx in ordered_sample_indices], axis=0)
    bshd_sample_seq_lens = np.asarray([samples[idx]["sample_seq_len"] for idx in ordered_sample_indices], dtype=np.int64)

    bshd_path = os.path.join(args.output_dir, "packing_bshd.npz")
    np.savez(
        bshd_path,
        input_ids=bshd_input_ids,
        labels=bshd_labels,
        loss_mask=bshd_loss_mask,
        sample_seq_lens=bshd_sample_seq_lens,
        global_num_microbatches=np.asarray([per_shard_samples], dtype=np.int64),
    )

    thd_path = os.path.join(args.output_dir, "packing_thd.npz")
    np.savez(
        thd_path,
        input_ids=bshd_input_ids,
        labels=bshd_labels,
        loss_mask=bshd_loss_mask,
        sample_seq_lens=bshd_sample_seq_lens,
        global_num_microbatches=np.asarray([per_shard_samples], dtype=np.int64),
    )

    print(f"Wrote BSHD dataset to {bshd_path}")
    print(f"Wrote THD dataset to {thd_path}")
    print(
        f"Logical samples={args.num_samples}, dp_world_size={args.dp_world_size}, "
        f"cp_world_size={args.cp_world_size}, samples_per_dp_shard={per_shard_samples}, "
        f"aligned_sample_multiple={cp_multiple}"
    )


if __name__ == "__main__":
    main()