import torch
import torch.distributed as dist
from torch.autograd import Function
import torch.multiprocessing as mp
import os
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core import tensor_parallel

# 自定义的all_reduce函数，解决autograd警告问题
class _AllReduce(Function):
    @staticmethod
    def forward(ctx, input_tensor, op, group):
        output = input_tensor.clone()
        dist.all_reduce(output, op=op, group=group)
        ctx.op = op
        ctx.group = group
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # print(f"_AllReduce grad_output is zero: {torch.allclose(grad_output, torch.zeros_like(grad_output), atol=1e-8)}", flush=True)
        # 对于SUM操作，梯度需要分布式求和
        # 对于MAX操作，梯度保持不变（因为只有最大值位置有梯度）
        if ctx.op == dist.ReduceOp.SUM:
            grad_input = grad_output.clone()
            dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        else:  # MAX操作
            raise NotImplementedError
            # grad_input = grad_output.clone()
        return grad_input, None, None

def safe_all_reduce(tensor, op, group):
    """安全的all_reduce函数，支持autograd"""
    return _AllReduce.apply(tensor, op, group)

class _TPSumForwardIdentityBackward(Function):
    """TP 聚合算子：前向做 SUM 聚合，反向保持恒等传递。

    适用场景：
    - 前向需要把 TP 分片结果做求和，数值上与非 TP 情况保持一致；
    - 反向不希望再跨 TP 做一次 SUM（避免梯度按 tp_size 放大）。

    注意：该语义假设每个 TP rank 上都在优化同一个“已聚合后”的目标。
    """

    @staticmethod
    def forward(ctx, input_tensor, group):
        output = input_tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 关键：不再进行跨 TP 的二次 all_reduce，避免梯度额外放大
        return grad_output, None


def tp_sum_forward_identity_backward(tensor, group=None):
    """TP 前向 SUM、反向恒等的聚合接口。"""
    if group is None:
        group = mpu.get_tensor_model_parallel_group()
    return _TPSumForwardIdentityBackward.apply(tensor, group)

class _DistributedLogSoftmax(Function):
    """
    封装了分布式 Log-Softmax 的自定义 Autograd 函数。
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits):
        """
        前向传播函数。
        Args:
            ctx: autograd 上下文对象。
            vocab_parallel_logits: 输入的 logits，形状为 [B, S, V_part]，
                                   其中 V_part = V / TP_world_size。
        """
        # --- 1. 计算全局最大值以保证数值稳定性 ---
        # 计算当前 rank 上的局部最大值
        logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
        # 使用自定义的all_reduce在所有 TP rank 间找到全局最大值
        logits_max = safe_all_reduce(
            logits_max,
            op=dist.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group()
        )

        # --- 2. 减去最大值 ---
        # 这一步可以防止 exp() 溢出
        logits_shifted = vocab_parallel_logits - logits_max

        # --- 3. 计算 exp 和的全局总和 ---
        # 计算当前 rank 上的 exp 和
        sum_exp_logits = logits_shifted.exp().sum(dim=-1, keepdim=True)
        # 使用自定义的all_reduce将所有 TP rank 上的和加起来，得到全局分母
        sum_exp_logits = safe_all_reduce(
            sum_exp_logits,
            op=dist.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group()
        )

        # --- 4. 计算最终的 log_softmax ---
        # log_softmax = logits_shifted - log(global_sum_exp)
        log_softmax_output = logits_shifted - sum_exp_logits.log()

        # --- 5. 保存反向传播所需的张量 ---
        # 根据推导的公式，反向传播需要 log_softmax 的输出 (用于计算 softmax)
        ctx.save_for_backward(log_softmax_output)

        return log_softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"_DistributedLogSoftmax grad_output is zero: {torch.allclose(grad_output, torch.zeros_like(grad_output), atol=1e-8)}", flush=True)
        """
        反向传播函数。
        Args:
            ctx: autograd 上下文对象。
            grad_output: log_softmax 输出的梯度。
        """
        # --- 1. 恢复前向传播保存的张量 ---
        log_softmax_output, = ctx.saved_tensors

        # --- 2. 计算 softmax ---
        # softmax = exp(log_softmax)
        softmax_output = torch.exp(log_softmax_output)

        # --- 3. 计算梯度的全局总和 ---
        # 根据公式 grad_input = grad_output - softmax * sum(grad_output)
        # 我们需要计算 sum(grad_output)
        # 首先计算当前 rank 上的局部和
        grad_sum = grad_output.sum(dim=-1, keepdim=True)
        # 使用自定义的all_reduce将所有 TP rank 上的和加起来，得到全局的 sum(grad_output)
        grad_sum = safe_all_reduce(
            grad_sum,
            op=dist.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group()
        )

        # --- 4. 计算输入的梯度 ---
        grad_input = grad_output - softmax_output * grad_sum

        # forward 方法只有一个输入，所以 backward 只需要返回一个梯度
        # print(f"_DistributedLogSoftmax grad_input is zero: {torch.allclose(grad_input, torch.zeros_like(grad_input), atol=1e-8)}", flush=True)
        return grad_input

# 创建一个用户友好的接口函数
def distributed_log_softmax(vocab_parallel_logits):
    """
    对输入的 logits 执行分布式的 Log-Softmax 计算。

    Args:
        vocab_parallel_logits: 输入的 logits, 形状为 [B, S, V_part]。

    Returns:
        经过分布式 log_softmax 计算后的张量。
    """
    # .apply 是调用 autograd.Function 的标准方式
    return _DistributedLogSoftmax.apply(vocab_parallel_logits)

class _DistributedSparseLogSoftmax(Function):
    """分布式 sparse Log-Softmax（TP shard）。

    目标：
    - 仅在给定的全局索引 `indices_global` 上返回 log_probs，输出形状固定为 [B, L, K]；
    - 避免在前向显式构造完整的 [B, L, V_local] log_probs 张量；
    - 仍与 TP 分片 softmax 分母一致（跨 TP 的 max/sum-exp 聚合）。

    约定：
    - `vocab_parallel_logits` 是当前 TP rank 的词表分片 logits，形状 [B, L, V_local]；
    - `indices_global` 是全局词表 id，形状 [B, L, K]；
    - 对不在本 rank shard 的索引，输出 `fill_value`；
    - 该算子输出是“本地分片视角”的 sparse log_probs（非全 TP 聚合输出）。
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        indices_global: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        fill_value: float,
        group,
    ):
        assert vocab_parallel_logits.dim() == 3
        assert indices_global.dim() == 3
        assert vocab_parallel_logits.shape[:2] == indices_global.shape[:2]

        if group is None:
            group = mpu.get_tensor_model_parallel_group()

        vstart = int(vocab_start)
        vend = int(vocab_end)
        fill_value = float(fill_value)

        # 1) 分布式 log-softmax 分母（max + sumexp）
        logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)

        logits_shifted = vocab_parallel_logits - logits_max
        sum_exp = logits_shifted.exp().sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        log_denom = logits_max + sum_exp.log()  # [B, L, 1]

        # 2) 仅在请求索引上 gather 本地 logits，再减去分母
        in_shard = (indices_global >= vstart) & (indices_global < vend)
        local_pos = indices_global - vstart
        safe_pos = local_pos.clamp(min=0, max=max(vocab_parallel_logits.shape[-1] - 1, 0)).to(torch.long)

        selected_local_logits = torch.gather(vocab_parallel_logits, dim=-1, index=safe_pos)
        selected_log_probs = selected_local_logits - log_denom
        out = torch.where(in_shard, selected_log_probs, torch.full_like(selected_log_probs, fill_value))

        # backward 需要：输入 logits、log_denom、索引与 in_shard 掩码
        ctx.save_for_backward(vocab_parallel_logits, log_denom, indices_global, in_shard)
        ctx.vocab_start = vstart
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, grad_output):
        vocab_parallel_logits, log_denom, indices_global, in_shard = ctx.saved_tensors
        vstart = int(ctx.vocab_start)
        group = ctx.group

        # dL/dz_v = sum_k g_k*(1[v==idx_k] - softmax_v)
        # 其中仅本 shard 索引参与 1[v==idx_k] 项；
        # 但 softmax 项的 sum_k(g_k) 需要是 global topK 的全局和（TP 内 all-reduce）。
        grad_selected = torch.where(in_shard, grad_output, torch.zeros_like(grad_output))
        grad_selected_sum = grad_selected.sum(dim=-1, keepdim=True)  # [B, L, 1] (local partial)
        dist.all_reduce(grad_selected_sum, op=dist.ReduceOp.SUM, group=group)

        # 复用 grad_input 缓冲，避免额外申请 softmax_local [B, L, V_local]
        grad_input = torch.empty_like(vocab_parallel_logits)
        torch.sub(vocab_parallel_logits, log_denom, out=grad_input)
        grad_input.exp_()
        grad_input.mul_(grad_selected_sum)
        grad_input.neg_()

        local_pos = indices_global - vstart
        safe_pos = local_pos.clamp(min=0, max=max(vocab_parallel_logits.shape[-1] - 1, 0)).to(torch.long)
        grad_input.scatter_add_(dim=-1, index=safe_pos, src=grad_selected)

        return grad_input, None, None, None, None, None


def distributed_sparse_log_softmax(
    vocab_parallel_logits: torch.Tensor,
    indices_global: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    fill_value: float = -1e9,
    group=None,
):
    """分布式 sparse Log-Softmax 接口。

    参数：
    - vocab_parallel_logits: [B, L, V_local]，当前 TP rank 的 logits 分片。
    - indices_global: [B, L, K]，全局词表 id。
    - vocab_start/vocab_end: 当前 TP rank 的全局词表范围 [start, end)。
    - fill_value: 不在本地 shard 的索引位置填充值。
    - group: TP 通信组，默认 `mpu.get_tensor_model_parallel_group()`。

    返回：
    - [B, L, K]，本地 shard 视角的 sparse log_probs。
    """
    return _DistributedSparseLogSoftmax.apply(
        vocab_parallel_logits,
        indices_global,
        int(vocab_start),
        int(vocab_end),
        float(fill_value),
        group,
    )

def get_tp_vocab_range(padded_vocab_size, tp_rank, tp_world_size):
    partition_vocab_size = padded_vocab_size // tp_world_size
    get_vocab_range = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
    return get_vocab_range(partition_vocab_size, tp_rank, tp_world_size)


def build_global_topk_plus_label_from_shard_log_probs(
    log_probs,
    labels,
    topK,
    vocab_start,
    vocab_end,
    group=None,
):
    """基于 TP shard 的 log_probs 构造“全局 topK + label 槽”（固定 K+1 形状）。

    约定：
    - 先在 TP 组内聚合得到全局 topK（全局词表 id）；
    - 再无条件追加 label 槽（即使 label 已在 topK）；
    - 输出在同一 TP 组内各 rank 完全一致。
    """
    assert log_probs.dim() == 3
    assert labels.dim() == 2
    assert log_probs.shape[:2] == labels.shape

    if group is None:
        group = mpu.get_tensor_model_parallel_group()

    topK = int(topK)
    vstart = int(vocab_start)
    vend = int(vocab_end)
    global_top_idx, global_top_val = build_global_topk_from_shard_log_probs(
        log_probs=log_probs,
        topK=topK,
        vocab_start=vstart,
        group=group,
    )

    label_global_val = gather_label_global_log_probs_from_shard_log_probs(
        log_probs=log_probs,
        labels=labels,
        vocab_start=vstart,
        vocab_end=vend,
        group=group,
    )

    out_idx, out_val = append_label_slot_after_global_topk(
        global_top_idx=global_top_idx,
        global_top_val=global_top_val,
        labels=labels,
        label_global_val=label_global_val,
    )

    return out_idx, out_val


def build_global_topk_from_shard_log_probs(
    log_probs,
    topK,
    vocab_start,
    group=None,
):
    """从 TP shard 的 log_probs 生成 TP 全局 topK（不含 label 槽）。"""
    assert log_probs.dim() == 3

    if group is None:
        group = mpu.get_tensor_model_parallel_group()

    _B, _L, V_local = log_probs.shape
    topK = int(topK)
    vstart = int(vocab_start)

    k_local = min(topK, int(V_local))
    local_vals, local_idx = torch.topk(log_probs, k=k_local, dim=-1)
    local_gid = local_idx + vstart

    global_top_idx, global_top_val = tp_allgather_global_topk(
        indices=local_gid,
        values=local_vals,
        finalK=topK,
        group=group,
    )
    return global_top_idx, global_top_val


def gather_label_global_log_probs_from_shard_log_probs(
    log_probs,
    labels,
    vocab_start,
    vocab_end,
    group=None,
):
    """从 TP shard 的 log_probs 汇总 labels 的全局 log_prob（TP 内 SUM）。"""
    assert log_probs.dim() == 3
    assert labels.dim() == 2
    assert log_probs.shape[:2] == labels.shape

    if group is None:
        group = mpu.get_tensor_model_parallel_group()

    _, _, V_local = log_probs.shape
    vstart = int(vocab_start)
    vend = int(vocab_end)

    in_shard = (labels >= vstart) & (labels < vend)
    label_local_pos = (labels - vstart).clamp(min=0, max=max(V_local - 1, 0))
    label_local_val = log_probs.gather(dim=-1, index=label_local_pos.unsqueeze(-1)).squeeze(-1)
    label_local_val = torch.where(in_shard, label_local_val, torch.zeros_like(label_local_val))
    label_global_val = label_local_val.clone()
    dist.all_reduce(label_global_val, op=dist.ReduceOp.SUM, group=group)
    return label_global_val


def append_label_slot_after_global_topk(
    global_top_idx,
    global_top_val,
    labels,
    label_global_val,
):
    """把 label 槽附加到全局 topK 末尾，输出固定 K+1 形状。"""
    assert global_top_idx.shape == global_top_val.shape
    assert global_top_idx.dim() == 3
    assert labels.dim() == 2
    assert label_global_val.shape == labels.shape
    assert global_top_idx.shape[:2] == labels.shape

    B, L, topK = global_top_idx.shape
    out_idx = torch.empty((B, L, topK + 1), dtype=torch.long, device=global_top_idx.device)
    out_val = torch.empty((B, L, topK + 1), dtype=global_top_val.dtype, device=global_top_val.device)
    out_idx[:, :, :topK] = global_top_idx
    out_val[:, :, :topK] = global_top_val
    out_idx[:, :, -1] = labels.to(dtype=torch.long)
    out_val[:, :, -1] = label_global_val.to(dtype=global_top_val.dtype)
    return out_idx, out_val


def tp_allgather_global_topk(indices, values, finalK, group=None):
    """在 TP 组内聚合稀疏候选，并做全局 topK。

    输入：
    - indices/values: [B, L, K_local]，通常为 student 侧本 rank 的固定形状候选。

    输出：
    - out_idx/out_val: [B, L, finalK]，全局（TP 组）候选直接取 topK。
    """
    assert indices.shape == values.shape
    assert indices.dim() == 3

    if group is None:
        group = mpu.get_tensor_model_parallel_group()

    _B, _L, _K_local = indices.shape
    finalK = int(finalK)

    tp_world_size = dist.get_world_size(group=group)
    gathered_idx = [torch.empty_like(indices) for _ in range(tp_world_size)]
    gathered_val = [torch.empty_like(values) for _ in range(tp_world_size)]
    dist.all_gather(gathered_idx, indices, group=group)
    dist.all_gather(gathered_val, values, group=group)

    cat_idx = torch.cat(gathered_idx, dim=-1)
    cat_val = torch.cat(gathered_val, dim=-1)

    # 直接在 all_gather 后的候选上取 topK（不做去重，不做哨兵过滤）。
    assert finalK <= int(cat_idx.shape[-1]), (
        f"finalK={finalK} must be <= gathered candidates={int(cat_idx.shape[-1])}"
    )
    topv, topp = torch.topk(cat_val, k=finalK, dim=-1)
    topi = torch.gather(cat_idx, dim=-1, index=topp)
    return topi, topv


def scatter_sparse_to_local_vocab_dense(indices, values, local_vocab_size, vocab_start, fill_value=-1e9):
    # 对重复 index 做稳健处理：同一 (b,t,vocab_id) 保留最大 value。
    # 这样即使上游出现重复写入，也不会依赖“最后一次写入覆盖”的不稳定行为。
    B, L, K = indices.shape
    V_local = int(local_vocab_size)
    out = torch.full((B, L, V_local), float(fill_value), dtype=values.dtype, device=values.device)

    local_pos = indices - int(vocab_start)
    valid = (local_pos >= 0) & (local_pos < V_local)
    if not valid.any():
        return out

    safe_pos = local_pos.clamp(min=0, max=max(V_local - 1, 0)).to(torch.long)
    src = torch.where(valid, values, torch.full_like(values, float(fill_value)))
    out.scatter_reduce_(dim=-1, index=safe_pos, src=src, reduce="amax", include_self=True)
    return out