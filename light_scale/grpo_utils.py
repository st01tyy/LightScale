from typing import Optional, Tuple, Union, Iterator, Dict

import torch

from light_scale.logp_utils import from_parallel_logits_to_logprobs
from megatron.core import mpu
from light_scale import dist_utils
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.training.global_vars import get_args
import torch.distributed as dist
from functools import partial
from megatron.core import tensor_parallel
from light_scale.gkd_utils import get_tp_vocab_range, build_global_topk_plus_label_from_shard_log_probs, distributed_log_softmax

import numpy as np
import os
from typing import List

def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_token_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Optional[torch.Tensor],
    loss_mask: torch.Tensor,
    reward_clip_range: Tuple[float, float] = None,
) -> torch.Tensor:
    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    eos_indices = loss_mask.size(1) - 1 - loss_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(loss_mask).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(r.dtype))
    token_reward = last_reward

    if kl is not None:
        kl_reward = -kl_coef * kl
        token_reward = token_reward + kl_reward  #TODO: padding部分，kl_reward必须要是0

    return token_reward


def compute_returns(
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Function that computes advantages and returns from rewards using REINFORCE.
    REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

    Input:
    - rewards: Tensor of shape (batch_size, response_size)
    - action_mask: Tensor of shape (batch_size, response_size), binary mask
    - gamma: discount factor

    Output:
    - returns: Tensor of shape (batch_size, response_size)
    """
    response_length = rewards.size(1)
    returns = torch.zeros_like(rewards)
    cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

    # Mask invalid responses if action_mask is provided
    if action_mask is not None:
        rewards = action_mask * rewards

    # Calculate returns by accumulating discounted rewards
    for t in reversed(range(response_length)):
        cumulative_return = rewards[:, t] + gamma * cumulative_return
        returns[:, t] = cumulative_return

    return returns


def masked_mean(
    tensor: torch.Tensor, 
    mask: Optional[torch.Tensor], 
    dim: int = None
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / (mask.sum(dim=dim) + float(1e-8))

def compute_batch_logp(model, data_iterator: Optional[Iterator], num_microbatches: int, micro_batch_size: int, iter_num: int = None) -> Optional[torch.Tensor]:
    def logp_func(output_tensor: torch.Tensor, labels: torch.Tensor, non_loss_data=True, debug_i: int = 0):
        assert non_loss_data
        # 无需准备loss mask, target_mask = (target < vocab_start_index) | (target >= vocab_end_index),
        logp = from_parallel_logits_to_logprobs(output_tensor, labels, inference_only=True, higher_stability=True, ignore_last=False)
        return {"logp": logp}

    def megatron_forward_step(data_iterator, model):
        forward_args = {
            "input_ids": None,
            "position_ids": None,
            "attention_mask": None
        }
        logp_func_args = {
            "labels": None
        }
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            assert data_iterator is not None
            batched_item: Dict[str, torch.Tensor] = next(data_iterator)
            if mpu.is_pipeline_first_stage():
                forward_args["input_ids"] = batched_item["input_ids"].to(device=dist_utils.get_device(), non_blocking=True)
            if mpu.is_pipeline_last_stage():
                logp_func_args = {
                    "labels": batched_item["labels"].to(device=dist_utils.get_device(), non_blocking=True),
                    "debug_i": batched_item["idx"]
                }
        # if int(os.environ.get("MRL_DUMP_FLAG", 0)) == 1 and dist.get_rank() == 0:
        #     # for debug
        #     dump_path = os.environ.get("MRL_DUMP_PATH", None)
        #     if dump_path is None:
        #         raise RuntimeError("MRL_DUMP_PATH is None")
        #     np.save(f"{dump_path}/iter_{iter_num}_logp_func_micro_batch_{i}_cp_{mpu.get_context_parallel_rank()}_input_ids.npy", local_batch_inputs["input_ids"].detach().cpu().numpy())
        #     np.save(f"{dump_path}/iter_{iter_num}_logp_func_micro_batch_{i}_cp_{mpu.get_context_parallel_rank()}_labels.npy", local_batch_inputs["labels"].detach().cpu().numpy())
        
        output_tensor = model(**forward_args)  # (batch, length, vocab/tp)

        return output_tensor, partial(logp_func, **logp_func_args)

    merged_global_batch_logp = None
    with torch.no_grad():
        logp_list = get_forward_backward_func()(
            forward_step_func=megatron_forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=get_args().seq_length, # This is ignored if variable_seq_lengths in the config is True.
            micro_batch_size=micro_batch_size,
            collect_non_loss_data=True,
        )
        merged_logp_list = []
        for logp in logp_list:
            local_logp_tensor = logp["logp"]
            if mpu.is_pipeline_last_stage():
                if mpu.get_tensor_model_parallel_rank() == 0:
                    merged_logp_tensor = dist_utils.gather_and_merge_cp_sharded_tensor(local_logp_tensor, merge_dim=-1)
                    merged_logp_list.append(merged_logp_tensor)
                dist.barrier(group=mpu.get_tensor_model_parallel_group())
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0:
            merged_batch_logp = torch.cat(merged_logp_list, dim=0)
            merged_global_batch_logp = dist_utils.gather_and_merge_dp_sharded_tensor(merged_batch_logp, merge_dim=0)              
    dist.barrier()
    # 至此，logp应该完成所有合并，合并后的结果在dp[0]pp[-1]cp[0]tp[0]，p2p到rank0
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
            logp_shape = torch.tensor([merged_global_batch_logp.shape[0], merged_global_batch_logp.shape[1]], dtype=torch.int64, device=dist_utils.get_device())
            dist.send(logp_shape, dst=0, group=mpu.get_pipeline_model_parallel_group())
            dist.send(merged_global_batch_logp, dst=0, group=mpu.get_pipeline_model_parallel_group())
        elif dist.get_rank() == 0:
            logp_shape = torch.zeros((2,), dtype=torch.int64, device=dist_utils.get_device())
            dist.recv(logp_shape, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
            merged_global_batch_logp = torch.zeros((logp_shape[0], logp_shape[1]), dtype=torch.float32, device=dist_utils.get_device())
            dist.recv(merged_global_batch_logp, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        dist.barrier()
    return merged_global_batch_logp

class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    # copy from verl: https://github.com/volcengine/verl/blob/992ac065a16c314e8fe7068230289dd7102da939/verl/utils/megatron/tensor_parallel.py#L142
    # shot out to the verl team
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (batch_size, seq_length, vocab_size // tp_size)

    Returns: (batch_size, seq_length,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)

def compute_batch_logits(
    model,
    data_iterator: Optional[Iterator],
    num_microbatches: int,
    micro_batch_size: int,
    iter_num: int = None,
) -> List[torch.Tensor]:
    # 用于教师模型计算一个batch的logits
    margs = get_args()
    use_sparse_topk = bool(getattr(margs, "gkd_sparse_topk_enabled", False))

    def logits_func(output_tensor: torch.Tensor, labels: torch.Tensor = None, non_loss_data=True, debug_i: int = 0):
        assert non_loss_data
        if not use_sparse_topk:
            return {"logits": output_tensor}

        # sparse 模式：先在全词表上计算 teacher log_probs，再做 dense->sparse。
        assert labels is not None
        teacher_temperature = float(getattr(margs, "teacher_temperature", 1.0))
        teacher_log_probs = distributed_log_softmax(output_tensor / teacher_temperature)
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = get_tp_vocab_range(
            padded_vocab_size=margs.padded_vocab_size,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        topk_indices, topk_values = build_global_topk_plus_label_from_shard_log_probs(
            log_probs=teacher_log_probs,
            labels=labels,
            topK=int(margs.gkd_topk),
            vocab_start=int(vocab_start),
            vocab_end=int(vocab_end),
            group=mpu.get_tensor_model_parallel_group(),
        )
        return {
            "topk_indices": topk_indices,
            "topk_values": topk_values,
        }
    
    def megatron_forward_step(data_iterator, model):
        forward_args = {
            "input_ids": None,
            "position_ids": None,
            "attention_mask": None
        }

        logits_func_args = {
            "debug_i": None,
            "labels": None,
        }

        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            assert data_iterator is not None
            batched_item: Dict[str, torch.Tensor] = next(data_iterator)
            if mpu.is_pipeline_first_stage():
                forward_args["input_ids"] = batched_item["input_ids"].to(device=dist_utils.get_device(), non_blocking=True)
            if mpu.is_pipeline_last_stage():
                logits_func_args["debug_i"] = batched_item["idx"]
                if use_sparse_topk:
                    logits_func_args["labels"] = batched_item["labels"].to(device=dist_utils.get_device(), non_blocking=True)
        
        output_tensor = model(**forward_args)  # (batch, length, vocab/tp)

        return output_tensor, partial(logits_func, **logits_func_args)
    
    with torch.no_grad():
        logits_list = get_forward_backward_func()(
            forward_step_func=megatron_forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=get_args().seq_length, # This is ignored if variable_seq_lengths in the config is True.
            micro_batch_size=micro_batch_size,
            collect_non_loss_data=True,
        )

    return logits_list