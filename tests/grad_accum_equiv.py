#!/usr/bin/env python3
"""Utilities for numerically validating custom gradient accumulation schedules.

This script compares parameter gradients produced by Megatron when:
    1. Running a single forward/backward call that covers the entire batch.
    2. Splitting the same batch into several smaller calls (e.g., for uneven
       gradient accumulation) and skipping the optimizer step between them.

The comparison is based on a fixed batch of pre-tokenized data that is loaded
from a local ``.npz`` file so both executions are perfectly deterministic.

The script expects to run under ``torchrun`` on a single node (up to 8 GPUs) and
reuses Megatron's initialization / model-building utilities.  It does **not**
modify Megatron's training loop – it simply prepares identical data iterators
for both execution modes and inspects the accumulated gradients afterwards.

Example:
    torchrun --nproc_per_node=8 tests/grad_accum_equiv.py \
        --grad-test-data /workspace/fixed_batch.npz \
        --grad-test-schedule 4,4,2 \
        --grad-test-output /workspace/grad_equiv_out \
        --micro-batch-size 2 --global-batch-size 16 --seq-length 2048 \
        --num-layers 2 --hidden-size 2048 --num-attention-heads 16 \
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2

The ``tests/grad_accum_equiv.py`` script assumes the user has patched Megatron's
pipeline schedules to accept a ``global_num_microbatches`` (or equivalent)
argument so that the loss scaling factor can be adjusted to match the overall
batch.  If that hook is missing, the script will raise a descriptive ``TypeError``.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.training.global_vars import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def _import_symbol(path: str):
    mod_name, attr = path.split(":", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


@dataclass
class FixedBatchDataset:
    """Container that exposes a deterministic sequence of microbatches."""

    data: Dict[str, torch.Tensor]
    num_microbatches: int

    @classmethod
    def from_npz(cls, path: str) -> "FixedBatchDataset":
        required = ["tokens", "labels", "loss_mask", "attention_mask", "position_ids"]
        np_data = np.load(path)
        for key in required:
            if key not in np_data:
                raise ValueError(f"Missing '{key}' in {path}; found {list(np_data.keys())}")
        tensors: Dict[str, torch.Tensor] = {}
        num_microbatches = None
        for key, value in np_data.items():
            tensor = torch.from_numpy(value)
            if tensor.dim() < 2:
                raise ValueError(f"Field '{key}' must have at least 2 dims (microbatch, batch, ...)")
            if num_microbatches is None:
                num_microbatches = tensor.shape[0]
            elif tensor.shape[0] != num_microbatches:
                raise ValueError("All tensors must share the same first dimension (microbatches)")
            tensors[key] = tensor
        assert num_microbatches is not None
        return cls(tensors, num_microbatches)

    def microbatch(self, index: int) -> Dict[str, torch.Tensor]:
        return {k: v[index].clone() for k, v in self.data.items()}


class MicrobatchIterator(Iterator[Dict[str, torch.Tensor]]):
    def __init__(self, dataset: FixedBatchDataset, indices: Sequence[int]):
        self._dataset = dataset
        self._indices = list(indices)
        self._pos = 0

    def __next__(self):
        if self._pos >= len(self._indices):
            raise StopIteration
        idx = self._indices[self._pos]
        self._pos += 1
        return self._dataset.microbatch(idx)


# --------------------------------------------------------------------------------------
# Forward / loss helpers
# --------------------------------------------------------------------------------------


def _build_lm_forward_step(device: torch.device):
    """Create a Megatron-compatible forward_step_func for causal LM tests."""

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, *, non_loss_data=False):
        assert not non_loss_data, "collect_non_loss_data path is unsupported in this test"
        losses = output_tensor.float() # (batch, length)
        mask = loss_mask.float() # (batch, length)
        loss = torch.mean(losses * mask, dim=1) # (batch,)
        loss = torch.mean(loss, dim=0) # scalar
        total_tokens = mask.sum()
        # loss = torch.sum(losses.view(-1) * mask) / (total_tokens + 1e-12)
        report = torch.stack([loss.detach(), total_tokens.detach()])
        dist.all_reduce(report, op=dist.ReduceOp.AVG, group=mpu.get_data_parallel_group())
        return loss, {"lm loss": report}

    def forward_step(data_iterator: Iterator[Dict[str, torch.Tensor]], model, *unused, **_unused_kwargs):
        batch = next(data_iterator)
        tokens = position_ids = attention_mask = None
        labels = loss_mask = None
        if mpu.is_pipeline_first_stage():
            tokens = batch["tokens"].to(device, non_blocking=True)
            position_ids = batch["position_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        if mpu.is_pipeline_last_stage():
            labels = batch["labels"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
        output = model(tokens, position_ids, attention_mask, labels=labels)
        if mpu.is_pipeline_last_stage():
            return output, lambda out, non_loss_data=False: loss_func(loss_mask, out, non_loss_data=non_loss_data)
        return output, lambda *args, **kwargs: (torch.tensor(0.0, device=device), {})

    return forward_step


# --------------------------------------------------------------------------------------
# Gradient capture helpers
# --------------------------------------------------------------------------------------


def _zero_model_grads(model: List[torch.nn.Module]) -> None:
    for model_chunk in model:
        if hasattr(model_chunk, "zero_grad_buffer"):
            model_chunk.zero_grad_buffer()
        for param in model_chunk.parameters():
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad.zero_()
            if param.grad is not None:
                param.grad = None
    optimizer = get_args().optimizer
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)


def _extract_grads(model: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for chunk_id, module in enumerate(model):
        prefix = f"chunk{chunk_id}"
        for name, param in module.named_parameters():
            grad = None
            if hasattr(param, "main_grad") and param.main_grad is not None:
                grad = param.main_grad
            elif param.grad is not None:
                grad = param.grad
            if grad is None:
                print(f"{name} grad is None")
                continue
            grads[f"{prefix}.{name}"] = grad.detach().float().cpu().clone()
            # print(grads[f"{prefix}.{name}"], flush=True)
    return grads


def _write_npz(path: str, payload: Dict[str, torch.Tensor]) -> None:
    cpu_np = {k: v.numpy() for k, v in payload.items()}
    np.savez(path, **cpu_np)


def _compare(local_a: Dict[str, torch.Tensor], local_b: Dict[str, torch.Tensor]) -> float:
    if local_a.keys() != local_b.keys():
        missing_a = sorted(set(local_a.keys()) - set(local_b.keys()))
        missing_b = sorted(set(local_b.keys()) - set(local_a.keys()))
        raise RuntimeError(f"Gradient dicts mismatch. Only in single: {missing_a}. Only in chunked: {missing_b}.")
    max_diff = 0.0
    for key in local_a:
        # print(key, local_a[key], local_b[key], flush=True)
        diff = torch.max(torch.abs(local_a[key] - local_b[key])).item()
        if diff > max_diff:
            print(key)
        max_diff = max(max_diff, diff)
    tensor = torch.tensor(max_diff, device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------


def _extra_args_provider(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Gradient Equivalence Test")
    group.add_argument("--grad-test-data", required=True, help="Path to .npz file with fixed microbatches")
    group.add_argument("--grad-test-output", required=True, help="Directory for gradient dumps and summary")
    group.add_argument(
        "--grad-test-schedule",
        required=True,
        help="Comma-separated list of microbatch counts for each accumulation chunk (e.g. 4,4,2)",
    )
    group.add_argument(
        "--grad-test-model-provider",
        default="pretrain_gpt:model_provider",
        help="Dotted path to the model_provider function used by setup_model_and_optimizer",
    )
    group.add_argument(
        "--grad-test-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance threshold for reporting success",
    )

    return parser


def _prepare_megatron(model_provider_path: str) -> List[torch.nn.Module]:
    model_provider = _import_symbol(model_provider_path)
    model, optimizer, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    get_args().optimizer = optimizer
    return model


def _run_schedule(
    *,
    schedule: Sequence[int],
    dataset: FixedBatchDataset,
    forward_step_func,
    model: List[torch.nn.Module],
    total_microbatches: int,
):
    args = get_args()
    fb_func = get_forward_backward_func()
    fb_signature = inspect.signature(fb_func)
    start = 0
    for idx, chunk in enumerate(schedule):
        indices = list(range(start, start + chunk))
        start += chunk
        iterator = MicrobatchIterator(dataset, indices)
        fb_kwargs = dict(
            forward_step_func=forward_step_func,
            data_iterator=iterator,
            model=model,
            num_microbatches=chunk,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=getattr(args, "decoder_seq_length", None),
            forward_only=False,
        )
        if "global_num_microbatches" in fb_signature.parameters:
            fb_kwargs["global_num_microbatches"] = total_microbatches
        else:
            raise TypeError(
                "Megatron schedules must accept 'global_num_microbatches'."
                " Please apply the pipeline-schedule patch before running this test."
            )
        if dist.get_rank() == 0:
            print(fb_kwargs['num_microbatches'], fb_kwargs['global_num_microbatches'])
        fb_func(**fb_kwargs)
        dist.barrier()

def _read_npz(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing gradient dump: {path}")
    np_data = np.load(path)
    return {k: torch.from_numpy(np_data[k].copy()) for k in np_data.files}


def main() -> None:
    initialize_megatron(extra_args_provider=_extra_args_provider)
    args = get_args()
    os.makedirs(args.grad_test_output, exist_ok=True)

    dataset = FixedBatchDataset.from_npz(args.grad_test_data)
    schedule = [int(item) for item in args.grad_test_schedule.split(",") if item.strip()]
    total_microbatches = sum(schedule)
    if dataset.num_microbatches != total_microbatches:
        raise ValueError(
            f"Dataset provides {dataset.num_microbatches} microbatches,"
            f" but schedule sums to {total_microbatches}."
        )

    model = _prepare_megatron(args.grad_test_model_provider)
    device = torch.cuda.current_device()
    forward_step = _build_lm_forward_step(torch.device(device))

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()

    single_schedule = [total_microbatches]
    # for chunk_id, module in enumerate(model):
    #     if hasattr(module, "zero_grad_buffer"):
    #         module.zero_grad_buffer()
    # _zero_model_grads(model)
    # get_args().optimizer.zero_grad()
    # torch.set_rng_state(cpu_rng_state)
    # torch.cuda.set_rng_state_all(cuda_rng_state)
    # _run_schedule(
    #     schedule=single_schedule,
    #     dataset=dataset,
    #     forward_step_func=forward_step,
    #     model=model,
    #     total_microbatches=total_microbatches,
    # )
    # single_grads = _extract_grads(model)
    # _write_npz(
    #     os.path.join(args.grad_test_output, f"single_rank{dist.get_rank()}.npz"),
    #     single_grads,
    # )
    # dist.barrier()
    # exit(0)
    

    # for chunk_id, module in enumerate(model):
    #     if hasattr(module, "zero_grad_buffer"):
    #         module.zero_grad_buffer()
    _zero_model_grads(model)
    get_args().optimizer.zero_grad()
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state_all(cuda_rng_state)
    _run_schedule(
        schedule=schedule,
        dataset=dataset,
        forward_step_func=forward_step,
        model=model,
        total_microbatches=total_microbatches,
    )
    chunked_grads = _extract_grads(model)
    _write_npz(
        os.path.join(args.grad_test_output, f"chunked_rank{dist.get_rank()}.npz"),
        chunked_grads,
    )

    single_grads = _read_npz(os.path.join(args.grad_test_output, f"single_rank{dist.get_rank()}.npz"))
    max_diff = _compare(single_grads, chunked_grads)
    status = "PASS" if max_diff <= args.grad_test_atol else "FAIL"
    summary = {
        "status": status,
        "max_abs_diff": max_diff,
        "tolerance": args.grad_test_atol,
        "schedule": schedule,
        "data_path": os.path.abspath(args.grad_test_data),
    }
    if dist.get_rank() == 0:
        with open(os.path.join(args.grad_test_output, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[grad-accum-test] {status}: max |Δ| = {max_diff:.3e}")
    dist.barrier()


if __name__ == "__main__":
    main()
