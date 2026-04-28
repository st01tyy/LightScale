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

@dataclass
class Sample:
    prompt: str = None
    response: str = None
    messages: List[Message] = None
    content_ids: np.ndarray = None
    loss_mask: np.ndarray = None
    teacher_log_probs: np.ndarray = None
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