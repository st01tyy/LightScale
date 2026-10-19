# 异步 Sample-Mean 梯度累积机制

## 背景与目的

LightScale 会异步接收 rollout 产生的样本，并可能把一次 optimizer update 拆分成多次训练调用。由于开启 sequence packing 后，一个物理 microbatch 能容纳的逻辑样本数量并不固定，因此不同调用使用的物理 microbatch 数量也可能不同。

无论物理上如何拆分，我们真正想要优化的目标应当保持不变。对于一次包含 $N$ 个真实样本的 optimizer update，LightScale 优化的是 sample-mean 目标：

$$
\mathcal{L}_{\mathrm{update}}
= \frac{1}{N} \sum_{i=1}^{N} \ell_i,
$$

其中 $\ell_i$ 是第 $i$ 个真实样本的 token-mean 目标。纯 padding 的 token 和纯 padding 的序列（dummy sequence）永远不会计入 $N$，也不会参与 loss 计算。

本文说明这一目标在异步梯度累积、DDP、BSHD 输入以及 THD sequence-packed 输入下是如何被保持一致的。

## 一、Megatron 原生 DDP 的梯度计算与更新机制

### 1.1 一次同步更新

设数据并行（DP）的 world size 为 $D$。假设每个 DP rank 拿到的样本数量相同，各自计算局部目标 $L_d$。在反向传播过程中，Megatron DDP 会在 DP 组内同步参数梯度。按照常规的 DDP 平均语义，同步后的梯度为：

$$
\bar{g}
= \frac{1}{D} \sum_{d=1}^{D} \nabla_\theta L_d.
$$

如果每个 rank 的本地 batch size 相同，且 $L_d$ 本身就是本地的 mean loss，那么上式恰好就是全局 sample mean 的梯度。

Megatron 的 pipeline schedule 在反向传播之前，还会对每次 forward step 返回的 loss 做一次额外缩放。在标准的定长 microbatch 场景下，schedule 会使用 microbatch 数量（`global_num_microbatches`，或本地回退值）进行归一化，从而使得累积 $M$ 个物理 microbatch 的效果，等价于一次逻辑 batch 的整体更新。

### 1.2 按 microbatch 数量归一化的局限性

按 microbatch 数量归一化，隐含的假设是：每个物理 microbatch 代表相同数量的训练数据。一旦开启 sequence packing，这个假设就不再成立：

- 一个 THD packed microbatch 可能包含很多个真实样本；
- 另一个 microbatch 可能只包含一个真实样本；
- 不同 DP shard 天然可能需要不同数量的 packing bucket。

如果此时仍按 packed bucket 数量做归一化，就相当于让某个样本的权重取决于它恰好被打包进了哪个 bucket——这并不是我们想要的目标函数。

## 二、LightScale 的异步 Sample-Mean 梯度累积原理

### 2.1 先累积分子，再除以一个固定的分母

对于一次异步 optimizer update，LightScale 首先确定每个 DP shard 分配到的真实样本总数：

$$
N_d = \left|\mathcal{S}_d\right|.
$$

batch 构造阶段会保证所有 DP shard 拿到相同数量的真实样本，因此对于这次全局 update 而言，$N_d = N / D$。每次 forward step 返回的是一个“分子”（numerator），而不是某个物理 microbatch 自身的均值。对于 DP rank $d$ 上的第 $b$ 个 microbatch：

$$
A_{d,b}
= \sum_{i \in \mathcal{S}_{d,b}} \ell_i.
$$

THD async 路径要求 `micro_batch_size = 1`，并且 trainer 保持一次 optimizer update
包含 `global_batch_size` 个样本。因此它复用
`global_num_microbatches = global_batch_size / DP = N_d` 作为固定分母，缩放每一个返回的分子：

$$
\widetilde{A}_{d,b}
= \frac{A_{d,b}}{N_d}.
$$

当本次 update 对应的所有异步训练调用都完成梯度累积之后，本地梯度结果为：

$$
\sum_b \nabla_\theta \widetilde{A}_{d,b}
= \nabla_\theta
  \left(\frac{1}{N_d} \sum_{i \in \mathcal{S}_d} \ell_i\right).
$$

随后 DDP 会对各 rank 的本地梯度求平均：

$$
\frac{1}{D}\sum_{d=1}^{D}
\nabla_\theta
\left(\frac{1}{N_d}\sum_{i \in \mathcal{S}_d}\ell_i\right)
= \nabla_\theta
\left(\frac{1}{N}\sum_{i=1}^{N}\ell_i\right).
$$

由此可见：不论把同一批真实样本以何种方式拆分累积，其效果都完全等价于对全局 sample-mean 目标做一次同步的 DDP 反向传播与更新。

这里的关键规则是：分母必须是“整个 optimizer update 内，该 DP shard 拥有的真实样本总数”，而**不是**训练调用次数、物理 microbatch 数量、packed bucket 数量，也不是 token 数量。

### 2.2 异步 optimizer 的执行流程

对于每一次收到的训练 batch，async trainer 会执行以下步骤：

1. 构造一个有序的逻辑样本 batch，并把样本分配到各个 DP shard；
2. 为所有 DP shard 确定一个相同的、非空的 planned microbatch 数量；
3. 让这些 planned microbatch 经过 Megatron 的 forward/backward schedule；
4. 将既有的 `global_num_microbatches` 传给 schedule。在 THD 的
  `micro_batch_size = 1` 约束下，它就是整次 update 的每 DP shard 样本总数，使每个
  microbatch 的分子使用同一分母；
5. 每次训练调用结束后调用 `optim.accumulate_grad_step()`；
6. 当本次 update 对应的所有异步调用都完成后，再统一调用一次 `step_with_accumulated_grads()`。

这一设计允许不同的异步 batch size、不同的 packing 结果，而不会改变最终每个样本获得的权重。

## 三、BSHD 模式

BSHD 是常规的稠密数据布局，形状为 `[batch, sequence, hidden]`。在做 context-parallel（CP）切分之前，每一行代表一个逻辑样本。

### 3.1 数据与 loss 行为

- 通用的 `DistributedTensorDataset` 与 `DataLoader` 仍然是 BSHD 的数据路径；
- 各行先按 DP 切分，再按常规的 head-tail 规则做 CP 切分；
- 现有的非 packed loss 函数计算的仍然是标准的 batch/sample mean；
- 当所有物理 microbatch 的真实样本数相同时，Megatron 标准的按 microbatch 数量归一化，本身就已经等价于 sample-mean 归一化。

在 BSHD 与 THD 的等价性验证脚本中，BSHD 提供的是基准布局：每个物理 microbatch 恰好承载一个逻辑样本，并且可以传入相同的样本数分母，从而使比较结果不受物理 microbatch 数量差异的影响。

### 3.2 不涉及 packed-attention 元数据

BSHD 不会构造 `PackedSeqParams`、累积序列长度（cumulative sequence lengths），也不会引入尾部的 padding-only dummy sequence。模型接收到的是常规的稠密张量。

## 四、THD Sequence-Packed 模式

THD 是 TransformerEngine packed attention 所使用的按 token 打包的数据布局。一个物理 microbatch 的序列容量是固定的，但其中可能拼接了多个真实样本。

### 4.1 规划非空且 DP 对齐的 microbatch

`build_microbatch_plan()` 会为一个逻辑 batch 生成一份确定性的规划：

1. 为每个 DP shard 分配数量相同的真实样本，同时尽量让各 shard 的总 token 长度接近；
2. 在每个 shard 内部贪心地组装满足容量限制的 bucket；
3. 令 $B$ 为各 shard 中自然产生的最大 bucket 数量；
4. 对 bucket 数量不足 $B$ 的 shard，拆分其已有的真实 bucket，直到该 shard 也拥有 $B$ 个 bucket。

每一个 planned bucket 都至少包含一个真实样本。该实现不会仅仅为了对齐 DDP 的 forward/backward 调用次数，而额外补充一个整体为空的 bucket。

`PlannedBatchIterator` 会在每个处于激活状态的 pipeline 端点上消费这份规划，并按需逐个 microbatch 地生成 packed 数据。

### 4.2 CP 切分的先后顺序

在开启 context parallel 时，正确的处理顺序是：

1. 先选出规划好的、属于本 DP shard 的样本；
2. 将每个样本裁剪到其自身的全局有效长度；
3. 对每个样本单独执行 CP 的 head-tail 切分；
4. 将各样本切分后的 local 片段拼接进 THD bucket；
5. 在 bucket 尾部补齐到固定的 local 容量。

如果先做 packing 再对整个 bucket 做 CP 切分，则是不正确的：这会打乱逻辑样本的边界，也会使累积长度（cumulative-length）相关的元数据失去意义。

### 4.3 THD 的 loss 分子

Packed OPD loss 会依据 local 累积边界，把每个真实样本重新还原出来。对于样本 $i$，会先在 CP 组内聚合其 token loss 与有效 token 数，再计算 token mean：

$$
\ell_i
= \frac{\sum_{t \in i} m_t\,q_t}
       {\sum_{t \in i} m_t + \epsilon},
$$

其中 $q_t$ 是逐 token 的 loss，$m_t$ 是 loss mask。返回给 Megatron 的 THD loss，是当前 packed microbatch 内所有真实样本的加和：

$$
A_b = \sum_{i \in \mathcal{S}_b} \ell_i.
$$

这里刻意**不会**除以 packed 样本数或 packed bucket 数。真正的、针对整次 update 的唯一一次除法，是由 pipeline schedule 使用 `global_num_microbatches` 来完成的；在 THD 的 `micro_batch_size = 1` 约束下，它等于每个 DP shard 的真实样本数。

### 4.4 TransformerEngine 的 packed 元数据

迭代器会生成两种不同用途的累积长度元数据：

- Local `cu_seqlens`：仅供 loss 在 CP 切分之后，重新还原出真实样本边界时使用；
- 全局的 `cu_seqlens_global` 与 `cu_seqlens_global_padded`：通过 `PackedSeqParams` 提供给 TransformerEngine 的 packed attention 使用。

对于一个真实全局 token 总量为 $T$、物理容量固定为 $P$ 的 bucket，尾部的物理 padding 会被表示为一个逻辑长度为零的终止序列：

$$
\begin{aligned}
\text{regular} &= [0, \ldots, T, T],\\
\text{padded}  &= [0, \ldots, T, P].
\end{aligned}
$$

两个数组包含的区间（interval）数量必须一致，这是 TransformerEngine 的要求。这个作为终止段的 padding-only 序列，既不计入真实样本数，也不参与 sample-mean loss 的计算。

### 4.5 Pipeline 同步与一致性校验

Pipeline 的第一个 stage 会构造全局 TE 元数据，并在模型执行之前把它同步给其他 pipeline stage；最后一个 stage 则保留仅供 loss 使用的 local 累积边界。

Async trainer 会校验：所有处于激活状态的首/尾 pipeline stage rank 的 planned microbatch 数量必须保持一致。

THD OPD 的异步路径目前会显式拒绝与 `distillation_enabled` 同时开启，原因是当前 packed iterator 并未提供蒸馏 loss 路径所需要的 teacher logits / top-k 数据契约。

## 五、验证预期

packing 梯度累积测试脚本会使用同一批逻辑样本，分别验证 BSHD 与 THD 的等价性。在开启 CP 的情况下，各 rank 本地的参数梯度可能存在差异，这是因为 CP 在两种布局下分配 token 贡献的方式并不相同。正确的比较方式，是在每个 CP 组内对参数梯度求和：

$$
G_d = \sum_{c=0}^{C-1} g_{d,c}.
$$

在选定精度（例如 BF16）对应的数值容差范围内，BSHD 与 THD 的 loss，以及按 CP 组求和后的梯度，都应当保持一致。
