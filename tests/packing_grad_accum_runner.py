#!/usr/bin/env python3
"""Run one gradient-accumulation check in either BSHD or THD packing mode.

This script is intentionally mode-based instead of comparing two layouts inside one
process. The companion shell script launches it twice:
1. once on `packing_bshd.npz`
2. once on `packing_thd.npz`

Both runs dump loss / grad-norm summaries and per-rank gradient tensors. The existing
`tests/compare_grad_accum_runs.py` script can then compare the two output directories.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.training.arguments import parse_and_validate_args
from megatron.training.global_vars import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from light_scale import dist_utils
from light_scale.gkd_utils import _TPSumForwardIdentityBackward


def _import_symbol(path: str):
    mod_name, attr = path.split(":", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


class LayoutDataset:
    def __init__(self, npz_path: str, mode: str):
        data = np.load(npz_path)
        self.mode = mode
        self.input_ids = torch.from_numpy(data["input_ids"].astype(np.int64, copy=False))
        self.labels = torch.from_numpy(data["labels"].astype(np.int64, copy=False))
        self.loss_mask = torch.from_numpy(data["loss_mask"].astype(np.float32, copy=False))
        global_num_microbatches = data["global_num_microbatches"]
        self.global_num_microbatches = int(global_num_microbatches.reshape(-1)[0])
        self.sample_seq_lens = None
        if "sample_seq_lens" in data:
            self.sample_seq_lens = torch.from_numpy(data["sample_seq_lens"].astype(np.int64, copy=False))

    def _local_sample_rows(self) -> List[Dict[str, torch.Tensor]]:
        if self.sample_seq_lens is None:
            raise ValueError("sample_seq_lens must be present for packing grad-accum tests")

        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        if self.input_ids.shape[0] % dp_world_size != 0:
            raise ValueError(
                f"dataset rows={self.input_ids.shape[0]} must be divisible by dp_world_size={dp_world_size}"
            )
        local_count = self.input_ids.shape[0] // dp_world_size
        start = dp_rank * local_count
        end = start + local_count

        rows: List[Dict[str, torch.Tensor]] = []
        for idx in range(start, end):
            rows.append(
                {
                    "input_ids": self.input_ids[idx].clone(),
                    "labels": self.labels[idx].clone(),
                    "loss_mask": self.loss_mask[idx].clone(),
                    "sample_seq_len": self.sample_seq_lens[idx].clone(),
                }
            )
        return rows

    def _build_bshd_microbatches(self) -> List[Dict[str, torch.Tensor]]:
        cp_world_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        microbatches: List[Dict[str, torch.Tensor]] = []
        for row in self._local_sample_rows():
            input_ids = row["input_ids"].unsqueeze(0)
            labels = row["labels"].unsqueeze(0)
            loss_mask = row["loss_mask"].unsqueeze(0)
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
            if cp_world_size > 1:
                input_ids = dist_utils.slice_2D_tensor_for_context_parallelism(input_ids, cp_rank, cp_world_size)
                labels = dist_utils.slice_2D_tensor_for_context_parallelism(labels, cp_rank, cp_world_size)
                loss_mask = dist_utils.slice_2D_tensor_for_context_parallelism(loss_mask, cp_rank, cp_world_size)
                position_ids = dist_utils.slice_2D_tensor_for_context_parallelism(
                    position_ids, cp_rank, cp_world_size
                )
            microbatches.append(
                {
                    "input_ids": input_ids.contiguous(),
                    "labels": labels.contiguous(),
                    "loss_mask": loss_mask.contiguous(),
                    "position_ids": position_ids.contiguous(),
                }
            )
        return microbatches

    def _build_thd_microbatches(self) -> List[Dict[str, torch.Tensor]]:
        cp_world_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        target_global_seq_len = self.input_ids.shape[1]
        cp_alignment = 1 if cp_world_size == 1 else 2 * cp_world_size
        if target_global_seq_len % cp_alignment != 0:
            raise ValueError(
                "THD CP requires the global packed length to be divisible by "
                f"{cp_alignment}, got {target_global_seq_len}"
            )
        
        # 1. 收集全局样本及其原始（全局）长度，先不切片
        global_samples = []
        for row in self._local_sample_rows():
            sample_seq_len = int(row["sample_seq_len"].item())
            global_samples.append({
                "length": sample_seq_len, # 保持全局长度
                "row": row
            })

        # 2. 使用全局长度进行一致的排序和分桶
        global_samples.sort(key=lambda item: item["length"], reverse=True)
        buckets = []
        current_bucket = []
        current_bucket_len = 0
        for sample in global_samples:
            sample_len = sample["length"]
            if sample_len % cp_alignment != 0:
                raise ValueError(
                    "Each packed sample must be divisible by the THD CP alignment "
                    f"{cp_alignment}, got {sample_len}"
                )
            if current_bucket and current_bucket_len + sample_len > target_global_seq_len:
                buckets.append(current_bucket)
                current_bucket = []
                current_bucket_len = 0
            current_bucket.append(sample)
            current_bucket_len += sample_len
        if current_bucket:
            buckets.append(current_bucket)

        # 3. 为每个 Bucket 构建当前 Rank 对应的微批次（在 Bucket 内部切片）
        target_local_seq_len = self.input_ids.shape[1] // cp_world_size
        local_bucket_count = len(buckets)
        bucket_count_tensor = torch.tensor(
            [local_bucket_count], dtype=torch.int64, device=dist_utils.get_device()
        )
        dist.all_reduce(bucket_count_tensor, op=dist.ReduceOp.MAX, group=mpu.get_data_parallel_group())
        padded_bucket_count = int(bucket_count_tensor.item())

        microbatches: List[Dict[str, torch.Tensor]] = []
        for bucket in buckets:
            bucket_lengths_global = [item["length"] for item in bucket]
            
            # 分别对 Bucket 内的各个样本在当前 CP Rank 进行切片
            sliced_input_ids = []
            sliced_labels = []
            sliced_loss_mask = []
            sliced_position_ids = []
            bucket_lengths_local = []
            
            for item in bucket:
                row = item["row"]
                sample_seq_len = item["length"]
                input_ids = row["input_ids"][:sample_seq_len].unsqueeze(0)
                labels = row["labels"][:sample_seq_len].unsqueeze(0)
                loss_mask = row["loss_mask"][:sample_seq_len].unsqueeze(0)
                position_ids = torch.arange(sample_seq_len, dtype=torch.long).unsqueeze(0)

                if cp_world_size > 1:
                    input_ids = dist_utils.slice_2D_tensor_for_context_parallelism(input_ids, cp_rank, cp_world_size)
                    labels = dist_utils.slice_2D_tensor_for_context_parallelism(labels, cp_rank, cp_world_size)
                    loss_mask = dist_utils.slice_2D_tensor_for_context_parallelism(loss_mask, cp_rank, cp_world_size)
                    position_ids = dist_utils.slice_2D_tensor_for_context_parallelism(
                        position_ids, cp_rank, cp_world_size
                    )

                input_ids = input_ids.squeeze(0).contiguous()
                labels = labels.squeeze(0).contiguous()
                loss_mask = loss_mask.squeeze(0).contiguous()
                position_ids = position_ids.squeeze(0).contiguous()
                
                bucket_lengths_local.append(int(input_ids.shape[0]))
                sliced_input_ids.append(input_ids)
                sliced_labels.append(labels)
                sliced_loss_mask.append(loss_mask)
                sliced_position_ids.append(position_ids)

            # 拼接切片后的数据并进行 Local Padding
            bucket_total_len_local = sum(bucket_lengths_local)
            input_ids = torch.cat(sliced_input_ids, dim=0)
            labels = torch.cat(sliced_labels, dim=0)
            loss_mask = torch.cat(sliced_loss_mask, dim=0)
            position_ids = torch.cat(sliced_position_ids, dim=0)
            
            padding_needed = target_local_seq_len - bucket_total_len_local
            if padding_needed > 0:
                input_ids = torch.cat([input_ids, torch.zeros((padding_needed,), dtype=torch.int64)], dim=0)
                labels = torch.cat([labels, torch.zeros((padding_needed,), dtype=torch.int64)], dim=0)
                loss_mask = torch.cat([loss_mask, torch.zeros((padding_needed,), dtype=torch.float32)], dim=0)
                position_ids = torch.cat([position_ids, torch.zeros((padding_needed,), dtype=torch.long)], dim=0)

            # Transformer Engine receives global metadata on every CP rank. Its
            # regular and padded cu_seqlens must have identical sequence counts.
            # Represent unused bucket capacity as one logical-zero dummy sequence:
            #   regular: [..., real_total, real_total]
            #   padded:  [..., real_total, target_global_seq_len]
            # The dummy is intentionally absent from cu_local and the sample count,
            # so it never contributes to the OPD sample-mean objective.
            cu_global = [0]
            running_global = 0
            for sample_len in bucket_lengths_global:
                running_global += sample_len
                cu_global.append(running_global)
            if running_global > target_global_seq_len:
                raise ValueError(
                    f"Packed bucket length {running_global} exceeds target {target_global_seq_len}"
                )

            tail_padding_global = target_global_seq_len - running_global
            cu_global_padded = list(cu_global)
            if tail_padding_global:
                if tail_padding_global % cp_alignment != 0:
                    raise ValueError(
                        "THD CP dummy sequence must be divisible by the CP alignment "
                        f"{cp_alignment}, got {tail_padding_global}"
                    )
                cu_global.append(running_global)
                cu_global_padded.append(target_global_seq_len)

            max_seqlen_global = max(max(bucket_lengths_global), tail_padding_global)
            if len(cu_global) != len(cu_global_padded):
                raise AssertionError("TE regular and padded cu_seqlens must have equal lengths")
            if cu_global[-1] != running_global or cu_global_padded[-1] != target_global_seq_len:
                raise AssertionError("Invalid global THD cumulative sequence lengths")

            # 构建局部的 cu_seqlens，用于供当前 Rank 的 Loss 函数切片计算样本 Loss
            cu_local = [0]
            running_local = 0
            for sample_len in bucket_lengths_local:
                running_local += sample_len
                cu_local.append(running_local)
            if running_local + padding_needed != target_local_seq_len:
                raise AssertionError("Local packed length does not match its CP shard length")

            microbatches.append(
                {
                    "input_ids": input_ids.unsqueeze(0).contiguous(),
                    "labels": labels.unsqueeze(0).contiguous(),
                    "loss_mask": loss_mask.unsqueeze(0).contiguous(),
                    "position_ids": position_ids.unsqueeze(0).contiguous(),
                    # 直接在 Dataset 阶段传递真实的全局 Metadata，避免多卡上估计不准
                    "cu_seqlens_global": torch.tensor(cu_global, dtype=torch.int32),
                    "cu_seqlens_global_padded": torch.tensor(cu_global_padded, dtype=torch.int32),
                    "max_seqlen_global": torch.tensor(max_seqlen_global, dtype=torch.int32),
                    "num_real_samples_in_microbatch": torch.tensor(len(bucket_lengths_global), dtype=torch.int32),
                    # 保留用于 Loss 计算的 local cu_seqlens
                    "cu_seqlens": torch.tensor(cu_local, dtype=torch.int32),
                }
            )

        while len(microbatches) < padded_bucket_count:
            microbatches.append(
                {
                    "input_ids": torch.zeros((1, target_local_seq_len), dtype=torch.int64),
                    "labels": torch.zeros((1, target_local_seq_len), dtype=torch.int64),
                    "loss_mask": torch.zeros((1, target_local_seq_len), dtype=torch.float32),
                    "position_ids": torch.zeros((1, target_local_seq_len), dtype=torch.long),
                    
                    # 补充对应的全局（Global）和局部（Local）字段
                    # A DP-alignment bucket has no real sample. Use the same
                    # logical-zero dummy convention as a partially filled bucket.
                    "cu_seqlens_global": torch.tensor([0, 0], dtype=torch.int32),
                    "cu_seqlens_global_padded": torch.tensor([0, target_global_seq_len], dtype=torch.int32),
                    "max_seqlen_global": torch.tensor(target_global_seq_len, dtype=torch.int32),
                    
                    # 局部的 cu_seqlens 用于 Loss Mask 等本地计算
                    "cu_seqlens": torch.tensor([0], dtype=torch.int32),
                    "num_real_samples_in_microbatch": torch.tensor(0, dtype=torch.int32),
                }
            )
        return microbatches

    def local_microbatches(self) -> List[Dict[str, torch.Tensor]]:
        if self.mode == "thd":
            return self._build_thd_microbatches()
        return self._build_bshd_microbatches()


class ListIterator(Iterator[Dict[str, torch.Tensor]]):
    def __init__(self, items: Sequence[Dict[str, torch.Tensor]]):
        self._items = list(items)
        self._pos = 0

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._pos >= len(self._items):
            raise StopIteration
        item = self._items[self._pos]
        self._pos += 1
        return item


def _extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Packing Gradient Accumulation Test")
    group.add_argument("--packing-grad-data", required=True, help="Path to packing_bshd.npz or packing_thd.npz")
    group.add_argument("--packing-grad-mode", choices=["bshd", "thd"], required=True, help="Layout mode to run")
    group.add_argument("--packing-grad-output", required=True, help="Directory for per-rank stats and gradients")
    group.add_argument(
        "--grad-test-model-provider",
        default="pretrain_gpt:model_provider",
        help="Import path to the Megatron model_provider function",
    )
    return parser


def _prepare_megatron(model_provider_path: str):
    model_provider = _import_symbol(model_provider_path)
    model, optimizer, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    return model, optimizer


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
                continue
            grads[f"{prefix}.{name}"] = grad.detach().float().cpu().clone()
    return grads


def _extract_params(model: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    for chunk_id, module in enumerate(model):
        prefix = f"chunk{chunk_id}"
        for name, param in module.named_parameters():
            params[f"{prefix}.{name}"] = param.detach().float().cpu().clone()
    return params


def _loss_metrics() -> Dict[str, object]:
    moe_metrics = {
        name: entry.values.detach().float().cpu().tolist()
        for name, entry in get_moe_metrics_tracker().metrics.items()
    }
    mtp_values = MTPLossLoggingHelper.tracker.get("values")
    return {
        "moe": moe_metrics,
        "mtp": [] if mtp_values is None else mtp_values.detach().float().cpu().tolist(),
    }


def _write_npz(path: str, payload: Dict[str, torch.Tensor]) -> None:
    np.savez(path, **{k: v.numpy() for k, v in payload.items()})


def _compute_grad_norm(grads: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for grad in grads.values():
        total += float(torch.sum(grad.float() * grad.float()).item())
    return math.sqrt(total)


def _build_packed_seq_params(batch: Dict[str, torch.Tensor], device: torch.device) -> PackedSeqParams:
    # TE receives global packed metadata even though input_ids is CP-local. This
    # is also valid for CP=1 and keeps the tail-padding convention identical.
    cu_seqlens = batch["cu_seqlens_global"].to(device=device, dtype=torch.int32, non_blocking=True)
    cu_seqlens_padded = batch["cu_seqlens_global_padded"].to(
        device=device, dtype=torch.int32, non_blocking=True
    )
    max_seqlen = int(batch["max_seqlen_global"].item())

    assert cu_seqlens.ndim == cu_seqlens_padded.ndim == 1
    assert len(cu_seqlens) == len(cu_seqlens_padded)
    assert int(cu_seqlens[0].item()) == int(cu_seqlens_padded[0].item()) == 0
    assert int(cu_seqlens[-1].item()) <= int(cu_seqlens_padded[-1].item())

    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def _build_forward_step(mode: str, device: torch.device):
    def _bshd_loss_func(output_tensor: torch.Tensor, loss_mask: torch.Tensor, non_loss_data: bool = False):
        assert not non_loss_data
        flat_loss = output_tensor.float().view(-1)
        flat_mask = loss_mask.float().view(-1)
        stats = torch.stack(
            [torch.sum(flat_loss * flat_mask), torch.sum(flat_mask)], dim=0
        )
        if mpu.get_context_parallel_world_size() > 1:
            stats = _TPSumForwardIdentityBackward.apply(stats, mpu.get_context_parallel_group())
        sample_mean = stats[0] / (stats[1] + 1e-8)
        report = torch.stack(
            [sample_mean.detach(), torch.tensor(1.0, dtype=torch.float32, device=device)], dim=0
        )
        dist.all_reduce(report, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
        return sample_mean, {"metrics": report}

    def _thd_loss_func(
        output_tensor: torch.Tensor,
        loss_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_real_samples_in_microbatch: torch.Tensor,
        non_loss_data: bool = False,
    ):
        assert not non_loss_data
        num_real_samples = int(num_real_samples_in_microbatch.item())
        if num_real_samples == 0:
            numerator = output_tensor.float().sum() * 0.0
            report = torch.stack(
                [numerator.detach(), torch.tensor(0.0, dtype=torch.float32, device=device)], dim=0
            )
            dist.all_reduce(report, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
            return numerator, {"metrics": report}

        flat_loss = (output_tensor.float() * loss_mask.float()).reshape(-1)
        flat_mask = loss_mask.float().reshape(-1)
        cu = cu_seqlens[: num_real_samples + 1].to(device=device, dtype=torch.long)
        prefix_loss = torch.cat([flat_loss.new_zeros((1,)), torch.cumsum(flat_loss, dim=0)], dim=0)
        prefix_mask = torch.cat([flat_mask.new_zeros((1,)), torch.cumsum(flat_mask, dim=0)], dim=0)
        sample_loss_sum = prefix_loss[cu[1:]] - prefix_loss[cu[:-1]]
        sample_token_count = prefix_mask[cu[1:]] - prefix_mask[cu[:-1]]
        stats = torch.stack([sample_loss_sum, sample_token_count], dim=0)
        if mpu.get_context_parallel_world_size() > 1:
            stats = _TPSumForwardIdentityBackward.apply(stats, mpu.get_context_parallel_group())
        sample_means = stats[0] / (stats[1] + 1e-8)
        numerator = sample_means.sum()
        report = torch.stack(
            [numerator.detach(), torch.tensor(float(num_real_samples), dtype=torch.float32, device=device)], dim=0
        )
        dist.all_reduce(report, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
        return numerator, {"metrics": report}

    def forward_step(data_iterator, model, *unused_args, **unused_kwargs):
        batch = next(data_iterator)
        input_ids = batch["input_ids"].to(device=device, dtype=torch.int64, non_blocking=True)
        labels = batch["labels"].to(device=device, dtype=torch.int64, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device=device, dtype=torch.float32, non_blocking=True)
        position_ids = batch["position_ids"].to(device=device, dtype=torch.long, non_blocking=True)
        padding_mask = loss_mask == 0

        packed_seq_params = None
        if mode == "thd":
            packed_seq_params = _build_packed_seq_params(batch, device)

        output_tensor = model(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=None,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
        )

        if mode == "thd":
            return output_tensor, lambda out, non_loss_data=False: _thd_loss_func(
                out,
                loss_mask=loss_mask,
                cu_seqlens=batch["cu_seqlens"].to(device=device, dtype=torch.int32, non_blocking=True),
                num_real_samples_in_microbatch=batch["num_real_samples_in_microbatch"].to(
                    device=device, dtype=torch.int32, non_blocking=True
                ),
                non_loss_data=non_loss_data,
            )

        return output_tensor, lambda out, non_loss_data=False: _bshd_loss_func(
            out, loss_mask=loss_mask, non_loss_data=non_loss_data
        )

    return forward_step


def main() -> None:
    parse_and_validate_args(extra_args_provider=_extra_args_provider)
    initialize_megatron()
    args = get_args()
    os.makedirs(args.packing_grad_output, exist_ok=True)

    dataset = LayoutDataset(args.packing_grad_data, args.packing_grad_mode)
    microbatches = dataset.local_microbatches()
    global_num_microbatches = dataset.global_num_microbatches
    num_microbatches = len(microbatches)
    if num_microbatches < 2:
        raise RuntimeError(
            "Packing gradient accumulation requires at least two physical microbatches before one update."
        )

    model, optimizer = _prepare_megatron(args.grad_test_model_provider)
    for module in model:
        if hasattr(module, "zero_grad_buffer"):
            module.zero_grad_buffer()
    optimizer.zero_grad()

    device = torch.device("cuda", torch.cuda.current_device())
    forward_step = _build_forward_step(args.packing_grad_mode, device)
    fb_func = get_forward_backward_func()

    fb_kwargs = dict(
        forward_step_func=forward_step,
        data_iterator=ListIterator(microbatches),
        model=model,
        num_microbatches=num_microbatches,
        forward_only=False,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=getattr(args, "decoder_seq_length", None),
    )
    fb_kwargs["global_num_microbatches"] = global_num_microbatches

    reduced_losses = fb_func(**fb_kwargs)

    total_numerator = 0.0
    total_count = 0.0
    for item in reduced_losses:
        total_numerator += float(item["metrics"][0].item())
        total_count += float(item["metrics"][1].item())
    loss_value = total_numerator / max(total_count, 1e-8)

    grads = _extract_grads(model)
    grad_norm = _compute_grad_norm(grads)
    loss_metrics = _loss_metrics()
    update_successful, optimizer_grad_norm, num_zeros_in_grad = optimizer.step()
    if not update_successful:
        raise RuntimeError("Optimizer step was skipped or failed.")
    params = _extract_params(model)

    rank = dist.get_rank()
    _write_npz(os.path.join(args.packing_grad_output, f"main_grads_step1_rank{rank}.npz"), grads)
    _write_npz(os.path.join(args.packing_grad_output, f"params_step1_rank{rank}.npz"), params)
    with open(
        os.path.join(args.packing_grad_output, f"stats_rank{rank}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "steps": 1,
                "mode": args.packing_grad_mode,
                "losses": [loss_value],
                "grad_norms": [grad_norm],
                "num_microbatches": num_microbatches,
                "accumulation_steps": num_microbatches,
                "global_num_microbatches": global_num_microbatches,
                "update_successful": True,
                "optimizer_grad_norm": None if optimizer_grad_norm is None else float(optimizer_grad_norm),
                "num_zeros_in_grad": None if num_zeros_in_grad is None else float(num_zeros_in_grad),
                "loss_metrics": loss_metrics,
            },
            f,
            indent=2,
        )

    if rank == 0:
        print(
            f"[packing-grad-test] mode={args.packing_grad_mode} "
            f"loss={loss_value:.8f} grad_norm={grad_norm:.8f} "
            f"num_microbatches={num_microbatches} global_num_microbatches={global_num_microbatches}"
        )
    dist.barrier()


if __name__ == "__main__":
    main()