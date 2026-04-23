# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copied from NeMo-Aligner/nemo_aligner/utils/distributed.py

import torch
import torch.distributed
from megatron.core import tensor_parallel
from megatron.core import mpu as parallel_state

@torch.no_grad()
def _compute_distributed_softmax(vocab_parallel_logits):
    """Expects a size B x S x V//TP tensor, computes a stable distributed softmax
        return shape B x S x V//TP but softmaxed across the V dimension
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    exp_logits = vocab_parallel_logits.exp_()

    sum_exp_logits = exp_logits.sum(-1)

    torch.distributed.all_reduce(
        sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group(),
    )
    # exp_logits becomes the softmax output
    exp_logits.div_(sum_exp_logits.unsqueeze(-1))

    return exp_logits


@torch.no_grad()
def _compute_distributed_log_softmax(vocab_parallel_logits):
    """Expects a size B x S x V//TP tensor, computes a stable distributed softmax
        return shape B x S x V//TP but softmaxed across the V dimension. More stable than just computing softmax
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group(),
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)

class DistributedLogprob(torch.autograd.Function):
    """Function to get logprobs out and differentiate through it
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, inference_only=False, higher_stability=False):
        get_vocab_range = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = parallel_state.get_tensor_model_parallel_rank()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        # higher stability uses a more numerically stable distributed log_softmax instead of softmax
        # however, it uses more VRAM because there is an unavoidable exp() OP on the entire logits tensor
        # some models (like DPO) will get -inf in the resulting logprobs unless you set higher_stability=True
        if higher_stability:
            log_softmax_output = _compute_distributed_log_softmax(vocab_parallel_logits)
            log_probs = log_softmax_output.clone()
            softmax_output = log_softmax_output.exp_()
        else:
            softmax_output = _compute_distributed_softmax(vocab_parallel_logits)
            # if we only do inference, then do the log in place
            log_probs = softmax_output.log_() if inference_only else softmax_output.log()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group(),
        )

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        # 1 if it's the chosen log prob, 0 otherwise
        is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(
            masked_target, num_classes=partition_vocab_size
        )

        grad_input = is_chosen.float().sub_(softmax)

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None

def from_parallel_logits_to_logprobs(
    vocab_parallel_logits, target, inference_only=False, higher_stability=False, ignore_last=True
):
    """get log probs out of a B x S x V//TP tensor
        NOTE: this function shifts the target, which means you must give it the unmodified targets

    Returns a B x S-1 tensor
    """

    if ignore_last:
        target = target.roll(shifts=-1, dims=-1)
    probs = DistributedLogprob.apply(vocab_parallel_logits, target, inference_only, higher_stability).contiguous()
    ### ignore_last should be true if labels are not shifted as a data preparation step
    if ignore_last:
        return probs[:, :-1]
    else:
        return probs