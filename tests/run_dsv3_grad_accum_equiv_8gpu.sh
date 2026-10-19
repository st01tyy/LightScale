#!/bin/bash

set -euo pipefail

# Compare one conventional full-batch Megatron update against multiple
# accumulation chunks followed by one update. Both runs rebuild the same DSV3-style
# MLA MoE + MTP model from the same seed and consume the same generated input.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_TORCH_COMPILE=0

WORK_DIR="${WORK_DIR:-$REPO_ROOT/tests/dsv3_grad_accum_equiv_out}"
DATA_PATH="$WORK_DIR/fixed_batch.npz"
FULL_OUT="$WORK_DIR/full"
CHUNKED_OUT="$WORK_DIR/chunked"
WORLD_SIZE=8
NUM_MICROBATCHES=8
MICRO_BATCH_SIZE=1
ACCUMULATION_SCHEDULE="3,2,3"

rm -rf "$WORK_DIR"
mkdir -p "$FULL_OUT" "$CHUNKED_OUT"

python tests/generate_grad_test_data.py \
    --output "$DATA_PATH" \
    --num-microbatches "$NUM_MICROBATCHES" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --seq-length 128 \
    --vocab-size 32000 \
    --seed 1234

COMMON_ARGS=(
    --grad-test-data "$DATA_PATH"
    --grad-test-schedule "$ACCUMULATION_SCHEDULE"
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$NUM_MICROBATCHES"
    --seq-length 128
    --max-position-embeddings 128
    --num-layers 2
    --hidden-size 128
    --ffn-hidden-size 512
    --num-attention-heads 8
    --num-query-groups 2
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --num-experts 8
    --moe-ffn-hidden-size 256
    --moe-shared-expert-intermediate-size 256
    --moe-router-topk 2
    --moe-router-score-function sigmoid
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type alltoall
    --moe-layer-freq 1
    --multi-latent-attention
    --q-lora-rank 64
    --kv-lora-rank 32
    --qk-head-dim 64
    --qk-pos-emb-head-dim 32
    --v-head-dim 64
    --mtp-num-layers 1
    --position-embedding-type rope
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --seed 1234
    --lr 1e-4
    --train-iters 1
    --tokenizer-type NullTokenizer
    --vocab-size 32000
    --disable-bias-linear
)

torchrun --nproc_per_node="$WORLD_SIZE" tests/grad_accum_equiv.py \
    --grad-test-mode full \
    --grad-test-output "$FULL_OUT" \
    "${COMMON_ARGS[@]}"

torchrun --nproc_per_node="$WORLD_SIZE" tests/grad_accum_equiv.py \
    --grad-test-mode chunked \
    --grad-test-output "$CHUNKED_OUT" \
    "${COMMON_ARGS[@]}"

python tests/compare_grad_accum_equiv_runs.py \
    --full "$FULL_OUT" \
    --chunked "$CHUNKED_OUT" \
    --require-moe-aux \
    --require-mtp \
    --atol 3e-4 \
    --rtol 0.0

echo "DSV3 gradient accumulation equivalence check passed. Artifacts are in: $WORK_DIR"