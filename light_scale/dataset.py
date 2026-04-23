import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Tuple, Optional
from light_scale.data import BatchExperience
from light_scale import dist_utils
from megatron.core import mpu
from light_scale.logger_utils import setup_logger
import bisect

class DistributedTensorDataset(Dataset):
    def __init__(self, batches: List[Union[BatchExperience]]):
        super().__init__()
        logger = setup_logger("light_scale")
        self.logger = logger

        tensor_batches: List[BatchExperience] = [None] * len(batches)
        
        for i, batch in enumerate(batches):
            # 可选：蒸馏对齐的 DP 段信息（[(start, length), ...]），若不存在则按原逻辑平均 DP 切分
            distill_segments: Optional[List[Tuple[int, int]]] = getattr(batch, 'distill_segments', None)
            tensor_batch = BatchExperience(
                input_ids=self._prepare_tensor_for_rank(batch.input_ids, segments=distill_segments),
                labels=self._prepare_tensor_for_rank(batch.labels, segments=distill_segments),
                loss_mask=self._prepare_tensor_for_rank(batch.loss_mask, segments=distill_segments),
                old_actor_logps=self._prepare_tensor_for_rank(batch.old_actor_logps, segments=distill_segments),
                ref_logps=self._prepare_tensor_for_rank(batch.ref_logps, segments=distill_segments),
                advantages=self._prepare_tensor_for_rank(batch.advantages, segments=distill_segments),
                outcome_rewards=self._prepare_tensor_for_rank(batch.outcome_rewards, apply_cp_slice=False, is_outcome_rewards=True, segments=distill_segments),
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
            "old_actor_logps": batch.old_actor_logps[idx] if batch.old_actor_logps is not None else None,
            "ref_logps": batch.ref_logps[idx] if batch.ref_logps is not None else None,
            "advantages": batch.advantages[idx] if batch.advantages is not None else None,
            "outcome_rewards": batch.outcome_rewards[idx] if batch.outcome_rewards is not None else None,
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