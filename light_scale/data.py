from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class Resource:
    type: str
    name: str
    base_url: str
    # port: int
    num_gpus: int = 0

@dataclass
class Message:
    content: str
    is_masked: bool

@dataclass
class MultiResponseSample:
    prompt: str
    dataset_type: str
    ground_truth: str
    sample_id: int = None
    problem: Optional[str] = None
    responses: List[str] = None
    group_messages: List[List[Message]] = None
    rewards: List[float] = None
    reward_metrics_list: List[dict] = None
    avg_reward_metrics: List[str] = None
    normed_rewards: List[float] = None
    completion_tokens: int = None
    total_tokens: int = None
    group_content_ids: Optional[List[np.ndarray]] = None
    group_loss_mask: Optional[List[np.ndarray]] = None
    group_teacher_log_probs: Optional[List[np.ndarray]] = None
    # Per response, global MoE-layer routing IDs aligned with group_content_ids.
    group_expert_ids: Optional[List[np.ndarray]] = None
    message_type: str = "sample"

    @classmethod
    def build_end_of_rollout(cls) -> "MultiResponseSample":
        return cls(
            prompt="",
            dataset_type="",
            ground_truth="",
            message_type="end_of_rollout",
        )

    def is_end_of_rollout(self) -> bool:
        return self.message_type == "end_of_rollout"

@dataclass
class Sample:
    prompt: str = None
    response: str = None
    messages: List[Message] = None
    content_ids: np.ndarray = None
    loss_mask: np.ndarray = None
    teacher_log_probs: np.ndarray = None
    # Global MoE-layer routing IDs aligned with content_ids: [moe_layers, tokens, topk].
    expert_ids: np.ndarray = None
    reward: float = None
    normed_reward: float = None
    completion_tokens: float = None
    reward_metrics: dict = None
    ground_truth: str = None
    dataset_type: str = None
    sample_id: int = None
    # reward_metrics = None

@dataclass
class BatchExperience:
    input_ids: torch.Tensor = None
    labels: torch.Tensor = None
    loss_mask: torch.Tensor = None
    teacher_logps: torch.Tensor = None
    # Global logical routing IDs: [batch, global_moe_layers, seq, topk].
    # Valid tokens contain global expert IDs; physical/logical padding is -1.
    expert_ids: torch.Tensor = None
    # THD packing 的 CP-local cumulative boundaries，仅供逐真实 sample 的 loss 聚合使用。
    cu_seqlens: Optional[object] = None
    # THD packing 的全局 logical cumulative boundaries，供 TE packed attention 使用。
    cu_seqlens_global: Optional[object] = None
    # THD packing 的全局 physical cumulative boundaries，显式描述尾部 padding dummy sequence。
    cu_seqlens_global_padded: Optional[object] = None
    # CP-local packed sequence 的最大 interval 长度，仅与 local `cu_seqlens` 对应。
    max_seqlen: Optional[object] = None
    # 全局 packed sequence 的最大 interval 长度，与 TE global metadata 对应。
    max_seqlen_global: Optional[object] = None
    # 当前 packed microbatch 中真实 sample 的数量，不包含 padding-only dummy sequence。
    num_real_samples_in_microbatch: torch.Tensor = None
    # packing 前每个 sample 的全局有效长度；DP shard 后用于先 CP slice 再 pack。
    sample_seq_lens: torch.Tensor = None
    # 每个 DP shard 的 planned microbatch 行边界，形状为 [dp_size, num_microbatches + 1]。
    # 同一列区间对应所有 DP shard 的一次真实 sample-packed 前后向，不包含 whole-empty bucket。
    microbatch_offsets: torch.Tensor = None
    # 标识该 batch 使用 THD packed 数据布局。
    is_thd_packed: bool = False
    # 标识该 batch 使用 teacher-logp OPD 目标函数。
    is_opd: bool = False
    outcome_rewards: torch.Tensor = None
    actor_logps: torch.Tensor = None
    old_actor_logps: torch.Tensor = None
    ref_logps: torch.Tensor = None
    advantages: torch.Tensor = None
    token_rewards: torch.Tensor = None
    kls: torch.Tensor = None
    batch_samples: List[Sample] = None
    teacher_logits: torch.Tensor = None
    teacher_topk_indices: torch.Tensor = None
    teacher_topk_values: torch.Tensor = None