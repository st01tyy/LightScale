import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Iterator, List, Union, Tuple, Optional
from light_scale.data import BatchExperience
from light_scale import dist_utils
from megatron.core import mpu
from light_scale.logger_utils import setup_logger
from megatron.training.global_vars import get_args, get_tokenizer
import bisect


@dataclass(frozen=True)
class MicrobatchPlan:
    """Deterministic assignment of one logical batch to DP-aligned microbatches."""

    ordered_indices: List[int]
    microbatch_offsets: List[List[int]]

    @property
    def num_microbatches(self) -> int:
        return len(self.microbatch_offsets[0]) - 1


def build_microbatch_plan(
    sample_seq_lens: List[int],
    dp_world_size: int,
    packed_seq_length: int,
) -> MicrobatchPlan:
    """Plan equal-count, nonempty, capacity-valid microbatches for every DP shard.

    The returned order groups samples by DP shard, then by planned microbatch. A
    shard that naturally needs fewer buckets is split into additional nonempty
    buckets instead of receiving a whole empty alignment bucket.
    """
    total_samples = len(sample_seq_lens)
    assert dp_world_size > 0
    assert total_samples > 0 and total_samples % dp_world_size == 0, (
        f"sample count {total_samples} must be positive and divisible by DP size {dp_world_size}"
    )
    assert packed_seq_length > 0
    assert all(0 < sample_len <= packed_seq_length for sample_len in sample_seq_lens), (
        f"all sample lengths must be in (0, {packed_seq_length}]"
    )

    samples_per_shard = total_samples // dp_world_size
    shard_indices: List[List[int]] = [[] for _ in range(dp_world_size)]
    shard_total_lens = [0 for _ in range(dp_world_size)]

    for sample_idx in sorted(range(total_samples), key=lambda idx: (-sample_seq_lens[idx], idx)):
        candidate_shards = [
            shard_id
            for shard_id in range(dp_world_size)
            if len(shard_indices[shard_id]) < samples_per_shard
        ]
        shard_id = min(
            candidate_shards,
            key=lambda current: (len(shard_indices[current]), shard_total_lens[current], current),
        )
        shard_indices[shard_id].append(sample_idx)
        shard_total_lens[shard_id] += sample_seq_lens[sample_idx]

    shard_buckets: List[List[List[int]]] = []
    for shard in shard_indices:
        buckets: List[List[int]] = []
        current_bucket: List[int] = []
        current_length = 0
        for sample_idx in sorted(shard, key=lambda idx: (-sample_seq_lens[idx], idx)):
            sample_length = sample_seq_lens[sample_idx]
            if current_bucket and current_length + sample_length > packed_seq_length:
                buckets.append(current_bucket)
                current_bucket = []
                current_length = 0
            current_bucket.append(sample_idx)
            current_length += sample_length
        if current_bucket:
            buckets.append(current_bucket)
        shard_buckets.append(buckets)

    num_microbatches = max(len(buckets) for buckets in shard_buckets)
    assert num_microbatches <= samples_per_shard
    for buckets in shard_buckets:
        while len(buckets) < num_microbatches:
            # Splitting a real bucket preserves capacity while replacing the old
            # empty-bucket alignment scheme with another nonempty microbatch.
            split_bucket_index = max(
                (index for index, bucket in enumerate(buckets) if len(bucket) > 1),
                key=lambda index: (sum(sample_seq_lens[idx] for idx in buckets[index]), len(buckets[index]), -index),
            )
            split_bucket = buckets[split_bucket_index]
            moved_sample_index = min(split_bucket, key=lambda idx: (sample_seq_lens[idx], idx))
            split_bucket.remove(moved_sample_index)
            buckets.append([moved_sample_index])

    ordered_indices: List[int] = []
    microbatch_offsets: List[List[int]] = []
    for buckets in shard_buckets:
        offsets = [0]
        for bucket in buckets:
            assert bucket
            assert sum(sample_seq_lens[idx] for idx in bucket) <= packed_seq_length
            ordered_indices.extend(bucket)
            offsets.append(offsets[-1] + len(bucket))
        assert offsets[-1] == samples_per_shard
        microbatch_offsets.append(offsets)

    assert len(ordered_indices) == total_samples
    assert len(set(ordered_indices)) == total_samples
    return MicrobatchPlan(
        ordered_indices=ordered_indices,
        microbatch_offsets=microbatch_offsets,
    )


class PlannedBatchIterator(Iterator[Dict[str, torch.Tensor]]):
    """Materialize one DP shard's planned THD microbatches on demand."""

    def __init__(self, batch: BatchExperience):
        assert batch.input_ids is not None and batch.labels is not None and batch.loss_mask is not None
        assert batch.is_thd_packed, "PlannedBatchIterator requires is_thd_packed=True"
        assert batch.sample_seq_lens is not None, "planned THD path requires per-sample sequence lengths"
        assert batch.microbatch_offsets is not None, "planned THD path requires microbatch offsets"
        assert batch.is_opd == (batch.teacher_logps is not None)
        if batch.expert_ids is not None:
            assert batch.expert_ids.dtype == torch.int32
            assert batch.expert_ids.ndim == 4
            assert batch.expert_ids.shape[0] == batch.input_ids.shape[0]
            assert batch.expert_ids.shape[2] == batch.input_ids.shape[1]

        args = get_args()
        self.batch = batch
        self.token_field_names = [
            name
            for name in (
                "input_ids",
                "labels",
                "loss_mask",
                "teacher_logps",
                "old_actor_logps",
                "ref_logps",
                "advantages",
                "token_rewards",
                "kls",
            )
            if getattr(batch, name) is not None
        ]
        self.cp_world_size = mpu.get_context_parallel_world_size()
        self.cp_rank = mpu.get_context_parallel_rank()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.packing_mode = str(getattr(args, "packing_mode", "flash")).lower()
        if self.packing_mode not in ("flash", "fused"):
            raise ValueError(
                f"packing_mode must be 'flash' or 'fused', got {self.packing_mode!r}"
            )
        self.target_global_seq_len = int(args.seq_length)
        self.target_local_seq_len = self.target_global_seq_len // self.cp_world_size
        self.cp_alignment = 1 if self.cp_world_size == 1 else 2 * self.cp_world_size
        assert self.target_global_seq_len % self.cp_alignment == 0

        offsets = batch.microbatch_offsets.to(dtype=torch.long, device="cpu")
        assert offsets.ndim == 2 and offsets.shape[0] == self.dp_world_size
        self.local_offsets = offsets[self.dp_rank].tolist()
        assert len(self.local_offsets) >= 2
        assert self.local_offsets[0] == 0
        assert all(left < right for left, right in zip(self.local_offsets, self.local_offsets[1:]))
        self.samples_per_shard = int(self.local_offsets[-1])
        assert batch.input_ids.shape[0] == self.samples_per_shard * self.dp_world_size
        self.global_row_start = self.dp_rank * self.samples_per_shard
        self.pad_token_id = self._get_pad_token_id()
        self._next_microbatch = 0

    def _get_pad_token_id(self) -> int:
        tokenizer = get_tokenizer()
        pad_token_id = getattr(getattr(tokenizer, "_tokenizer", None), "pad_token_id", None)
        if pad_token_id is None:
            try:
                pad_token_id = tokenizer.pad
            except NotImplementedError:
                pad_token_id = None
        assert pad_token_id is not None, "THD packing requires tokenizer pad_token_id"
        return int(pad_token_id)

    def __iter__(self) -> "PlannedBatchIterator":
        return self

    def __len__(self) -> int:
        return len(self.local_offsets) - 1

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._next_microbatch >= len(self):
            raise StopIteration

        local_start = self.local_offsets[self._next_microbatch]
        local_end = self.local_offsets[self._next_microbatch + 1]
        self._next_microbatch += 1
        global_start = self.global_row_start + local_start
        global_end = self.global_row_start + local_end

        local_samples = []
        local_expert_ids = []
        local_router_padding_masks = []
        for row_idx in range(global_start, global_end):
            global_length = int(self.batch.sample_seq_lens[row_idx].item())
            assert global_length % self.cp_alignment == 0
            sample_tensors = {
                name: getattr(self.batch, name)[row_idx : row_idx + 1, :global_length]
                for name in self.token_field_names
            }
            if self.cp_world_size > 1:
                sample_tensors = {
                    name: dist_utils.slice_2D_tensor_for_context_parallelism(
                        tensor, self.cp_rank, self.cp_world_size
                    )
                    for name, tensor in sample_tensors.items()
                }
            sample_length = int(sample_tensors["input_ids"].shape[1])
            assert sample_length * self.cp_world_size == global_length
            local_samples.append((global_length, sample_length, sample_tensors))

            if self.batch.expert_ids is not None:
                sample_expert_ids = self.batch.expert_ids[row_idx, :, :global_length, :]
                if self.cp_world_size > 1:
                    num_moe_layers, _, topk = sample_expert_ids.shape
                    sample_expert_ids = dist_utils.slice_2D_tensor_for_context_parallelism(
                        sample_expert_ids.permute(0, 2, 1).reshape(num_moe_layers * topk, -1),
                        self.cp_rank,
                        self.cp_world_size,
                    ).reshape(num_moe_layers, topk, sample_length).permute(0, 2, 1)
                assert sample_expert_ids.shape[1] == sample_length
                router_padding_mask = (sample_expert_ids == -1).all(dim=(0, 2))
                valid_router_tokens = (sample_expert_ids >= 0).all(dim=(0, 2))
                assert torch.all(router_padding_mask | valid_router_tokens)
                local_expert_ids.append(sample_expert_ids)
                local_router_padding_masks.append(router_padding_mask)

        bucket_local_length = sum(sample[1] for sample in local_samples)
        bucket_global_length = sum(sample[0] for sample in local_samples)
        assert local_samples and bucket_local_length <= self.target_local_seq_len
        assert bucket_global_length <= self.target_global_seq_len

        packed = {}
        for name in self.token_field_names:
            tensor = torch.cat([sample[2][name].squeeze(0) for sample in local_samples], dim=0)
            padding_length = self.target_local_seq_len - bucket_local_length
            if padding_length:
                fill_value = self.pad_token_id if name in ("input_ids", "labels") else 0.0
                tensor = torch.cat(
                    [
                        tensor,
                        torch.full(
                            (padding_length,), fill_value, dtype=tensor.dtype, device=tensor.device
                        ),
                    ],
                    dim=0,
                )
            packed[name] = tensor.unsqueeze(0).contiguous()

        packed_expert_ids = None
        router_padding_mask = None
        if self.batch.expert_ids is not None:
            packed_expert_ids = torch.cat(local_expert_ids, dim=1)
            router_padding_mask = torch.cat(local_router_padding_masks, dim=0)
            padding_length = self.target_local_seq_len - bucket_local_length
            if padding_length:
                packed_expert_ids = torch.cat(
                    [
                        packed_expert_ids,
                        torch.full(
                            (packed_expert_ids.shape[0], padding_length, packed_expert_ids.shape[2]),
                            -1,
                            dtype=torch.int32,
                            device=packed_expert_ids.device,
                        ),
                    ],
                    dim=1,
                )
                router_padding_mask = torch.cat(
                    [
                        router_padding_mask,
                        torch.ones(
                            (padding_length,), dtype=torch.bool, device=router_padding_mask.device
                        ),
                    ],
                    dim=0,
                )
            assert packed_expert_ids.shape[1] == self.target_local_seq_len
            assert router_padding_mask.shape == (self.target_local_seq_len,)
            assert torch.equal(
                router_padding_mask,
                (packed_expert_ids == -1).all(dim=(0, 2)),
            )

        local_lengths = [sample[1] for sample in local_samples]
        global_lengths = [sample[0] for sample in local_samples]
        local_cu_seqlens = [0]
        global_cu_seqlens = [0]
        for local_length, global_length in zip(local_lengths, global_lengths):
            local_cu_seqlens.append(local_cu_seqlens[-1] + local_length)
            global_cu_seqlens.append(global_cu_seqlens[-1] + global_length)

        # Ordinary CUs describe logical samples. Padded CUs additionally describe
        # their physical locations in the fixed-size THD tensor. FusedAttention
        # consumes both representations; FlashAttention only consumes ordinary CUs.
        global_cu_seqlens_padded = list(global_cu_seqlens)
        tail_global_length = self.target_global_seq_len - bucket_global_length
        if tail_global_length:
            global_cu_seqlens.append(bucket_global_length)
            global_cu_seqlens_padded.append(self.target_global_seq_len)
            local_cu_seqlens.append(self.target_local_seq_len)

        if self.packing_mode == "flash":
            # FA2 has no padded-CU interface. Make its ordinary CU cover every
            # physical token, including the fixed-size tail padding.
            attention_global_cu_seqlens = global_cu_seqlens_padded
            attention_max_seqlen_global = max(
                right - left
                for left, right in zip(
                    attention_global_cu_seqlens, attention_global_cu_seqlens[1:]
                )
            )
            attention_max_seqlen_local = max(
                right - left for left, right in zip(local_cu_seqlens, local_cu_seqlens[1:])
            )
        else:
            # FusedAttention receives logical lengths and physical offsets
            # separately, so tail padding does not enlarge logical max_seqlen.
            attention_global_cu_seqlens = global_cu_seqlens
            attention_max_seqlen_global = max(global_lengths)
            attention_max_seqlen_local = max(local_lengths)

        device = packed["input_ids"].device
        assert self.batch.outcome_rewards is not None
        sample_outcome_rewards = self.batch.outcome_rewards[global_start:global_end]
        assert sample_outcome_rewards.numel() == len(local_samples)
        return {
            **packed,
            "expert_ids": packed_expert_ids,
            "router_padding_mask": router_padding_mask,
            "cu_seqlens": torch.tensor([local_cu_seqlens], dtype=torch.int32, device=device),
            "cu_seqlens_global": torch.tensor(
                [attention_global_cu_seqlens], dtype=torch.int32, device=device
            ),
            "cu_seqlens_global_padded": torch.tensor(
                [global_cu_seqlens_padded], dtype=torch.int32, device=device
            ),
            "max_seqlen": torch.tensor(
                [attention_max_seqlen_local],
                dtype=torch.int32,
                device=device,
            ),
            "max_seqlen_global": torch.tensor(
                [attention_max_seqlen_global], dtype=torch.int32, device=device
            ),
            "num_real_samples_in_microbatch": torch.tensor(
                [len(local_samples)], dtype=torch.int32, device=device
            ),
            "is_thd_packed": torch.tensor(1, dtype=torch.int32, device=device),
            "is_opd": torch.tensor(
                int(self.batch.is_opd), dtype=torch.int32, device=device
            ),
            "sample_outcome_rewards": sample_outcome_rewards.to(device=device, non_blocking=True),
            "idx": torch.tensor([[local_start]], dtype=torch.int64, device=device),
        }

class DistributedTensorDataset(Dataset):
    def __init__(self, batches: List[Union[BatchExperience]]):
        super().__init__()
        logger = setup_logger("light_scale")
        self.logger = logger

        tensor_batches: List[BatchExperience] = [None] * len(batches)
        
        for i, batch in enumerate(batches):
            if batch.is_thd_packed:
                raise RuntimeError(
                    "DistributedTensorDataset does not support is_thd_packed; "
                    "the async packed route must use PlannedBatchIterator."
                )

            # 可选：蒸馏对齐的 DP 段信息（[(start, length), ...]），若不存在则按原逻辑平均 DP 切分
            distill_segments: Optional[List[Tuple[int, int]]] = getattr(batch, 'distill_segments', None)
            tensor_batch = BatchExperience(
                input_ids=self._prepare_tensor_for_rank(batch.input_ids, segments=distill_segments),
                labels=self._prepare_tensor_for_rank(batch.labels, segments=distill_segments),
                loss_mask=self._prepare_tensor_for_rank(batch.loss_mask, segments=distill_segments),
                teacher_logps=self._prepare_tensor_for_rank(batch.teacher_logps),
                old_actor_logps=self._prepare_tensor_for_rank(batch.old_actor_logps, segments=distill_segments),
                ref_logps=self._prepare_tensor_for_rank(batch.ref_logps, segments=distill_segments),
                advantages=self._prepare_tensor_for_rank(batch.advantages, segments=distill_segments),
                outcome_rewards=self._prepare_tensor_for_rank(batch.outcome_rewards, apply_cp_slice=False, is_outcome_rewards=True, segments=distill_segments),
                is_thd_packed=False,
                is_opd=batch.is_opd,
                teacher_logits=batch.teacher_logits,
                teacher_topk_indices=batch.teacher_topk_indices,
                teacher_topk_values=batch.teacher_topk_values,
            )
            tensor_batches[i] = tensor_batch
        
        self.tensor_batches = tensor_batches
        # 前缀和
        prefix_sums = [None] * (len(tensor_batches) + 1)
        prefix_sums[0] = 0
        for i, tensor_batch in enumerate(tensor_batches):
            prefix_sums[i + 1] = prefix_sums[i] + tensor_batch.input_ids.shape[0]
        self.prefix_sums = prefix_sums
        logger.debug(f"dataset length: {prefix_sums[-1]}")
        
    
    def __len__(self):
        return self.prefix_sums[-1]
    
    def __getitem__(self, idx):
        self.logger.debug(f"raw idx: {idx}")
        batch_id = bisect.bisect_right(self.prefix_sums, idx) - 1
        idx = idx - self.prefix_sums[batch_id]
        self.logger.debug(f"batch_id: {batch_id}, idx: {idx}")
        batch: BatchExperience = self.tensor_batches[batch_id]
        item = {
            "input_ids": batch.input_ids[idx] if batch.input_ids is not None else None,
            "labels": batch.labels[idx] if batch.labels is not None else None,
            "loss_mask": batch.loss_mask[idx] if batch.loss_mask is not None else None,
            "teacher_logps": batch.teacher_logps[idx] if batch.teacher_logps is not None else None,
            "old_actor_logps": batch.old_actor_logps[idx] if batch.old_actor_logps is not None else None,
            "ref_logps": batch.ref_logps[idx] if batch.ref_logps is not None else None,
            "advantages": batch.advantages[idx] if batch.advantages is not None else None,
            "outcome_rewards": batch.outcome_rewards[idx] if batch.outcome_rewards is not None else None,
            "is_thd_packed": torch.tensor(0, dtype=torch.int32),
            "is_opd": torch.tensor(int(batch.is_opd), dtype=torch.int32),
            "idx": torch.LongTensor([idx]),
            "teacher_logits": batch.teacher_logits[idx] if batch.teacher_logits is not None else None,
            "teacher_topk_indices": batch.teacher_topk_indices[idx] if batch.teacher_topk_indices is not None else None,
            "teacher_topk_values": batch.teacher_topk_values[idx] if batch.teacher_topk_values is not None else None,
        }
        return item

    def _prepare_tensor_for_rank(self, current_tensor, apply_cp_slice=True, is_outcome_rewards=False, segments: Optional[List[Tuple[int, int]]] = None):
        if current_tensor is None:
            return None
        if is_outcome_rewards: # outcome_rewards 特殊处理：先 unsqueeze
            current_tensor = current_tensor.unsqueeze(dim=1)

        # 先按可选的 distill 段选择样本（对齐 teacher→student logits 的重组顺序）
        if segments is not None and len(segments) > 0:
            self.logger.debug(f"segments: {segments}")
            # segments: List[(start, length)]，沿 batch 维拼接
            pieces = []
            for (start, length) in segments:
                if length <= 0:
                    continue
                end = start + length
                assert 0 <= start < current_tensor.shape[0] and end <= current_tensor.shape[0], \
                    f"segment [{start}:{end}) 超界 (batch={current_tensor.shape[0]})"
                pieces.append(current_tensor[start:end, :])
            if len(pieces) == 0:
                # 若段为空，返回一个空的占位（形状兼容），避免后续栈失败
                empty = current_tensor.new_empty((0, current_tensor.shape[1]))
                sliced_by_segments = empty
            else:
                sliced_by_segments = torch.cat(pieces, dim=0).contiguous()
            sliced_dp = sliced_by_segments
        else:
            sliced_dp = dist_utils.slice_2D_tensor_for_data_parallelism(
                current_tensor,
                mpu.get_data_parallel_rank(),
                mpu.get_data_parallel_world_size()
            )

        if apply_cp_slice:
            sliced_cp_dp = dist_utils.slice_2D_tensor_for_context_parallelism(
                sliced_dp,
                mpu.get_context_parallel_rank(),
                mpu.get_context_parallel_world_size()
            )
            return sliced_cp_dp
        return sliced_dp

def collate_fn(batches: list):
    batched_item = dict()
    for key in batches[0].keys():
        tensors = [batch[key] for batch in batches]
        if tensors[0] is None:
            batched_item[key] = None
            continue
        batched_item[key] = torch.stack(tensors, dim=0).contiguous()
    return batched_item

def create_distributed_dataloader(batches: List[BatchExperience], batch_size):
    dataset = DistributedTensorDataset(batches)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0
    )
    return dataloader