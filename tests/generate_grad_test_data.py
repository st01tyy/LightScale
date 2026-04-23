#!/usr/bin/env python3
"""Utility to build deterministic micro-batch tensors for grad_accum_equiv.py.

The resulting ``.npz`` file contains the exact keys/shape conventions that
``tests/grad_accum_equiv.py`` expects:

    tokens         : [num_microbatches, micro_batch_size, seq_length]
    labels         : same shape as ``tokens``
    loss_mask      : same shape as ``tokens`` (float32)
    attention_mask : [num_microbatches, micro_batch_size, 1, seq_length, seq_length]
    position_ids   : [num_microbatches, micro_batch_size, seq_length]

All arrays are generated on CPU using NumPy so the file can be produced on any
machine.  You can optionally load an existing ``tokens`` tensor from disk to
make the synthetic batch resemble real data; otherwise the script will draw
random token ids.

Example:
    python tests/generate_grad_test_data.py \
        --output /workspace/fixed_batch.npz \
        --num-microbatches 10 \
        --micro-batch-size 2 \
        --seq-length 2048 \
        --vocab-size 50304 \
        --seed 1234
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np


PAD_TOKEN_ID = 0


def _load_tokens(path: str, *, num_microbatches: int, micro_batch_size: int, seq_length: int) -> np.ndarray:
    array = np.load(path)
    if array.shape != (num_microbatches, micro_batch_size, seq_length):
        raise ValueError(
            f"Precomputed tokens must have shape {(num_microbatches, micro_batch_size, seq_length)},"
            f" but got {array.shape}."
        )
    if array.dtype not in (np.int32, np.int64):
        raise ValueError("Token array must be int32/int64.")
    return array.astype(np.int64, copy=False)


def _random_tokens(*, vocab_size: int, num_microbatches: int, micro_batch_size: int, seq_length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=1, high=vocab_size, size=(num_microbatches, micro_batch_size, seq_length), dtype=np.int64)


def _build_loss_mask(tokens: np.ndarray) -> np.ndarray:
    mask = np.ones_like(tokens, dtype=np.float32)
    mask[tokens == PAD_TOKEN_ID] = 0.0
    return mask


def _build_attention_mask(tokens: np.ndarray) -> np.ndarray:
    num_microbatches, micro_batch_size, seq_length = tokens.shape
    causal = np.tril(np.ones((seq_length, seq_length), dtype=np.float32))
    causal = np.broadcast_to(causal, (num_microbatches, micro_batch_size, seq_length, seq_length)).copy()
    causal = causal[:, :, np.newaxis, :, :]  # shape -> [mb, batch, 1, s, s]
    return causal


def _build_position_ids(tokens: np.ndarray) -> np.ndarray:
    seq_length = tokens.shape[-1]
    position = np.arange(seq_length, dtype=np.int64)
    position = np.broadcast_to(position, tokens.shape)
    return position.copy()


def _shift_labels(tokens: np.ndarray) -> np.ndarray:
    # Standard next-token prediction target: shift by one position.
    labels = tokens.copy()
    labels[..., :-1] = tokens[..., 1:]
    labels[..., -1] = PAD_TOKEN_ID
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic inputs for grad_accum_equiv.py")
    parser.add_argument("--output", required=True, help="Path to the output .npz file")
    parser.add_argument("--num-microbatches", type=int, required=True, help="Number of microbatches")
    parser.add_argument("--micro-batch-size", type=int, required=True, help="Micro batch size per microbatch")
    parser.add_argument("--seq-length", type=int, required=True, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=50304, help="Vocabulary size for random token generation")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--tokens-npy", type=str, default=None, help="Optional .npy file containing precomputed tokens with shape [num_microbatches, micro_batch_size, seq_length]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    rng = np.random.default_rng(args.seed)

    if args.tokens_npy:
        tokens = _load_tokens(
            args.tokens_npy,
            num_microbatches=args.num_microbatches,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
        )
    else:
        tokens = _random_tokens(
            vocab_size=args.vocab_size,
            num_microbatches=args.num_microbatches,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            rng=rng,
        )

    labels = _shift_labels(tokens)
    loss_mask = _build_loss_mask(tokens)
    attention_mask = _build_attention_mask(tokens)
    position_ids = _build_position_ids(tokens)

    np.savez(
        args.output,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    print(f"Wrote dataset to {args.output}")


if __name__ == "__main__":
    main()
