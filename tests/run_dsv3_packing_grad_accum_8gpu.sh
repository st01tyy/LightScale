#!/bin/bash

set -euo pipefail

# Validate DSV3-style MLA MoE + MTP BSHD/THD packing equivalence. Each run
# accumulates every local physical microbatch before one optimizer update.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_TORCH_COMPILE=0

WORK_DIR="${WORK_DIR:-$REPO_ROOT/tests/dsv3_packing_grad_accum_out}"
DATA_DIR="$WORK_DIR/data"
BSHD_OUT="$WORK_DIR/bshd"
THD_OUT="$WORK_DIR/thd"
WORLD_SIZE=8
CP_SIZE=2
EP_SIZE=4
DP_SIZE=$((WORLD_SIZE / CP_SIZE))

if (( WORLD_SIZE % CP_SIZE != 0 )); then
  echo "CP_SIZE must divide WORLD_SIZE for this test" >&2
  exit 1
fi

if (( DP_SIZE % EP_SIZE != 0 )); then
  echo "EP_SIZE must divide the data-parallel size for this test" >&2
  exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$DATA_DIR" "$BSHD_OUT" "$THD_OUT"

python tests/generate_packing_grad_test_data.py \
  --output-dir "$DATA_DIR" \
  --num-samples 16 \
  --dp-world-size "$DP_SIZE" \
  --cp-world-size "$CP_SIZE" \
  --seq-length 256 \
  --min-sample-seq-len 32 \
  --max-sample-seq-len 192 \
  --vocab-size 32000 \
  --seed 1234

COMMON_ARGS=(
  --micro-batch-size 1
  --global-batch-size 16
  --seq-length 256
  --max-position-embeddings 256
  --num-layers 2
  --hidden-size 128
  --ffn-hidden-size 512
  --num-attention-heads 8
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size "$CP_SIZE"
  --expert-model-parallel-size "$EP_SIZE"
  --num-experts 8
  --moe-ffn-hidden-size 256
  --moe-shared-expert-intermediate-size 256
  --moe-router-topk 2
  --moe-router-score-function sigmoid
  --moe-router-load-balancing-type seq_aux_loss
  --moe-aux-loss-coeff 1e-3
  --moe-token-dispatcher-type alltoall
  --moe-layer-freq 1
  --moe-grouped-gemm
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

torchrun --nproc_per_node="$WORLD_SIZE" tests/packing_grad_accum_runner.py \
  --packing-grad-data "$DATA_DIR/packing_bshd.npz" \
  --packing-grad-mode bshd \
  --packing-grad-output "$BSHD_OUT" \
  "${COMMON_ARGS[@]}"

torchrun --nproc_per_node="$WORLD_SIZE" tests/packing_grad_accum_runner.py \
  --packing-grad-data "$DATA_DIR/packing_thd.npz" \
  --packing-grad-mode thd \
  --packing-grad-output "$THD_OUT" \
  "${COMMON_ARGS[@]}"

python tests/compare_grad_accum_runs.py \
  --left "$BSHD_OUT" \
  --right "$THD_OUT" \
  --cp-size "$CP_SIZE" \
  --require-moe-aux \
  --require-mtp \
  --atol 3e-4 \
  --rtol 0.0

echo "DSV3 BSHD/THD packing gradient accumulation check passed. Artifacts are in: $WORK_DIR"