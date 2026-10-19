# Async Sample-Mean Gradient Accumulation

## Purpose

LightScale asynchronously receives rollout samples and may split one optimizer
update into multiple training calls. The calls may also use different numbers of
physical microbatches because sequence packing changes how many logical samples
fit into one model input.

The intended objective is independent of that physical partitioning. For one
optimizer update with $N$ real samples, LightScale optimizes the sample-mean
objective

$$
\mathcal{L}_{\mathrm{update}}
= \frac{1}{N} \sum_{i=1}^{N} \ell_i,
$$

where $\ell_i$ is the token-mean objective for real sample $i$. Padding-only
tokens and padding-only sequences never contribute to $N$ or to the loss.

This document describes how that objective is preserved across asynchronous
gradient accumulation, DDP, BSHD inputs, and THD sequence-packed inputs.

## 1. Megatron DDP Gradient and Update Semantics

### 1.1 One synchronized update

Let $D$ be the data-parallel world size. Assume each DP rank receives the same
number of samples and computes the local objective $L_d$. During backward,
Megatron DDP synchronizes parameter gradients across the DP group. With the
usual DDP averaging semantics, the synchronized gradient is

$$
\bar{g}
= \frac{1}{D} \sum_{d=1}^{D} \nabla_\theta L_d.
$$

If every rank has the same local batch size and $L_d$ is its local mean loss,
this is exactly the gradient of the global sample mean.

Megatron pipeline schedules additionally scale the loss returned by a forward
step before backward. In the standard fixed-size microbatch case, the schedule
uses the microbatch count (`global_num_microbatches`, or the local fallback) so
that accumulating $M$ physical microbatches produces the same scale as one
logical batch update.

### 1.2 Limitation of microbatch-count normalization

Microbatch-count normalization assumes that every physical microbatch represents
the same amount of training data. This assumption is no longer valid when
sequence packing is enabled:

- one THD packed microbatch can contain many real samples;
- another can contain only one real sample;
- different DP shards can naturally require different bucket counts.

Dividing by the number of packed buckets would therefore weight a sample by the
bucket in which it happened to be packed. This is not the desired objective.

## 2. LightScale Async Sample-Mean Accumulation

### 2.1 Accumulate numerators, divide by a fixed denominator

For one asynchronous optimizer update, LightScale first determines the total
number of real samples assigned to each DP shard:

$$
N_d = \left|\mathcal{S}_d\right|.
$$

The batch construction path guarantees that all DP shards have the same real
sample count, so $N_d = N / D$ for the global update. Each forward step returns
a *numerator* rather than a physical-microbatch mean. For microbatch $b$ on DP
rank $d$:

$$
A_{d,b}
= \sum_{i \in \mathcal{S}_{d,b}} \ell_i.
$$

The THD async path requires `micro_batch_size = 1`, and the trainer keeps one
optimizer update at `global_batch_size` samples. It therefore reuses
`global_num_microbatches = global_batch_size / DP = N_d` as this fixed
denominator when scaling every returned numerator:

$$
\widetilde{A}_{d,b}
= \frac{A_{d,b}}{N_d}.
$$

After all asynchronous training calls for the update have accumulated gradients,
the local result is

$$
\sum_b \nabla_\theta \widetilde{A}_{d,b}
= \nabla_\theta
  \left(\frac{1}{N_d} \sum_{i \in \mathcal{S}_d} \ell_i\right).
$$

DDP then averages the local gradients:

$$
\frac{1}{D}\sum_{d=1}^{D}
\nabla_\theta
\left(\frac{1}{N_d}\sum_{i \in \mathcal{S}_d}\ell_i\right)
= \nabla_\theta
\left(\frac{1}{N}\sum_{i=1}^{N}\ell_i\right).
$$

Therefore, accumulating any partition of the same real samples is equivalent to
one synchronized DDP backward/update for the global sample-mean objective.

The important rule is that the denominator is the total number of real samples
for the DP shard in the whole optimizer update, **not** the number of calls,
physical microbatches, packed buckets, or tokens.

### 2.2 Async optimizer flow

The async trainer performs the following operations for each received training
batch:

1. Build an ordered logical-sample batch and assign samples to DP shards.
2. Determine a common number of nonempty planned microbatches for all DP shards.
3. Run the planned microbatches through Megatron forward/backward scheduling.
4. Pass the existing `global_num_microbatches` to the schedule. Under the THD
  `micro_batch_size = 1` contract, it is the update-wide sample count per DP
  shard and divides every microbatch numerator.
5. Call `optim.accumulate_grad_step()` after each training call.
6. After all asynchronous calls for the update have completed, call
   `step_with_accumulated_grads()` once.

This design permits different asynchronous batch sizes and different packing
patterns without changing the final sample weighting.

## 3. BSHD Mode

BSHD is the ordinary dense layout with shape `[batch, sequence, hidden]`. Each
row represents one logical sample before context-parallel slicing.

### 3.1 Data and loss behavior

- The generic `DistributedTensorDataset` and `DataLoader` remain the BSHD data
  path.
- Rows are split over DP and then over CP using the normal head-tail CP slicing
  rule.
- The existing non-packed loss functions compute their standard batch/sample
  means.
- When all physical microbatches have the same real-sample count, Megatron's
  standard microbatch normalization is already equivalent to sample-mean
  normalization.

For the BSHD-vs-THD equivalence harness, BSHD provides the reference layout:
one logical sample is materialized per physical microbatch, and the same
sample-count denominator can be supplied to make the comparison independent of
physical microbatch count.

### 3.2 No packed-attention metadata

BSHD does not construct `PackedSeqParams`, cumulative sequence lengths, or
terminal padding-only sequences. The model receives conventional dense tensors.

## 4. THD Sequence-Packed Mode

THD is the packed-token layout used by TransformerEngine packed attention. A
physical microbatch has fixed sequence capacity but may contain multiple real
samples concatenated along the token dimension.

### 4.1 Planning nonempty DP-aligned microbatches

`build_microbatch_plan()` creates a deterministic plan for one logical batch:

1. Assign equal counts of real samples to each DP shard while approximately
   balancing sequence lengths.
2. Greedily form capacity-valid buckets for each shard.
3. Let $B$ be the largest natural bucket count among shards.
4. Split existing real buckets on less crowded shards until each shard has $B$
   buckets.

Every planned bucket contains at least one real sample. The implementation does
not append a whole empty bucket merely to align DDP forward/backward calls.

`PlannedBatchIterator` consumes the plan on every active pipeline endpoint and
materializes one packed microbatch at a time.

### 4.2 CP slicing order

For context parallelism, the required order is:

1. select the planned DP-shard samples;
2. trim every sample to its own global valid length;
3. apply the CP head-tail slice to each sample independently;
4. concatenate the local sample slices into the THD bucket;
5. append physical tail padding to the fixed local capacity.

Packing before per-sample CP slicing is invalid because it can split logical
sample boundaries and invalidate cumulative-length metadata.

### 4.3 THD loss numerator

The packed OPD loss reconstructs each real sample from local cumulative
boundaries. For sample $i$, it first aggregates token loss and valid-token count
across the CP group, then computes its token mean:

$$
\ell_i
= \frac{\sum_{t \in i} m_t\,q_t}
       {\sum_{t \in i} m_t + \epsilon},
$$

where $q_t$ is the token loss and $m_t$ is the loss mask. The THD loss returned
to Megatron is the sum over real samples in the current packed microbatch:

$$
A_b = \sum_{i \in \mathcal{S}_b} \ell_i.
$$

It is deliberately **not** divided by the number of packed samples or packed
buckets. The pipeline schedule performs the single update-wide division by
`global_num_microbatches`, which equals the per-DP-shard sample count under the
THD `micro_batch_size = 1` contract.

### 4.4 TransformerEngine packed metadata

The iterator produces two different forms of cumulative-length metadata:

- Local `cu_seqlens`: used only by the loss to recover real sample boundaries
  after CP slicing.
- Global `cu_seqlens_global` and `cu_seqlens_global_padded`: used by
  TransformerEngine packed attention through `PackedSeqParams`.

For a bucket with real global token total $T$ and fixed physical capacity $P$,
physical tail padding is represented as a terminal logical-zero sequence:

$$
\begin{aligned}
\text{regular} &= [0, \ldots, T, T],\\
\text{padded}  &= [0, \ldots, T, P].
\end{aligned}
$$

Both arrays have the same number of intervals, as required by TransformerEngine.
The terminal padding-only sequence is excluded from the real sample count and
from the sample-mean loss.

### 4.5 Pipeline synchronization and consistency checks

The first pipeline stage builds global TE metadata and synchronizes it to the
other pipeline stages before model execution. The last stage retains local
loss-only cumulative boundaries.

The async trainer verifies that all active first/last pipeline-stage ranks agree
on the planned number of microbatches.

The THD OPD async path currently rejects `distillation_enabled`, because the
packed iterator does not provide the teacher-logit/top-k payload contract needed
by the separate distillation loss path.

## 5. Validation Expectations

The packing gradient-accumulation test harness validates BSHD against THD using
the same logical samples. With CP enabled, rank-local parameter gradients may
differ because CP distributes token contributions differently between layouts.
The correct comparison is the parameter-wise sum within each CP group:

$$
G_d = \sum_{c=0}^{C-1} g_{d,c}.
$$

The BSHD and THD losses and CP-group-summed gradients should agree within the
numerical tolerance appropriate for the selected precision, such as BF16.
