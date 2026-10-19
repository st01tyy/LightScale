#!/bin/bash

set -euo pipefail

# Compare one conventional dense Qwen2-style Megatron update against multiple
# accumulation chunks followed by one update. Each launch rebuilds the same GQA
# model from the same BF16 initialization and consumes the same fixed input.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_TORCH_COMPILE=0

WORK_DIR="${WORK_DIR:-$REPO_ROOT/tests/qwen2_dense_grad_accum_equiv_out}"
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
    --group-query-attention
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --position-embedding-type rope
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --normalization RMSNorm
    --swiglu
    --add-qkv-bias
    --disable-bias-linear
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --seed 1234
    --lr 1e-4
    --train-iters 1
    --tokenizer-type NullTokenizer
    --vocab-size 32000
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
    --atol 3e-4 \
    --rtol 0.0

echo "Qwen2 dense gradient accumulation equivalence check passed. Artifacts are in: $WORK_DIR"