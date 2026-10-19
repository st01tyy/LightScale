#!/bin/bash

set -euo pipefail

# This script validates that BSHD and THD packing gradient accumulation produce
# the same loss / gradients on a single machine with 8 GPUs.
#
# Workflow:
# 1. Generate one synthetic logical-sample set.
# 2. Materialize it into two layouts: BSHD and THD packing.
# 3. Launch the same tiny Megatron model twice with torchrun:
#    - once on the BSHD file
#    - once on the THD file
# 4. Compare the dumped loss/grad artifacts.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# `set -u` makes reading an unset PYTHONPATH an error. Keep an existing value
# when present, while allowing this script to run in a clean shell.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_DEVICE_MAX_CONNECTIONS=1

export NVTE_TORCH_COMPILE=0

WORK_DIR="${WORK_DIR:-$REPO_ROOT/tests/packing_grad_accum_out}"
DATA_DIR="$WORK_DIR/data"
BSHD_OUT="$WORK_DIR/bshd"
THD_OUT="$WORK_DIR/thd"
# CP_SIZE="${CP_SIZE:-1}"
CP_SIZE=4
WORLD_SIZE=8

if (( WORLD_SIZE % CP_SIZE != 0 )); then
  echo "CP_SIZE=$CP_SIZE must divide WORLD_SIZE=$WORLD_SIZE" >&2
  exit 1
fi

DP_SIZE=$((WORLD_SIZE / CP_SIZE))

rm -rf "$WORK_DIR"
mkdir -p "$DATA_DIR" "$BSHD_OUT" "$THD_OUT"

# Generate one logical sample set, then materialize both BSHD and THD views.
python tests/generate_packing_grad_test_data.py \
  --output-dir "$DATA_DIR" \
  --num-samples 32 \
  --dp-world-size "$DP_SIZE" \
  --cp-world-size "$CP_SIZE" \
  --seq-length 256 \
  --min-sample-seq-len 32 \
  --max-sample-seq-len 192 \
  --vocab-size 32000 \
  --seed 1234

# Common tiny-model arguments for a fast local check.
COMMON_ARGS=(
  --micro-batch-size 1
  --global-batch-size 32
  --seq-length 256
  --max-position-embeddings 256
  --num-layers 2
  --hidden-size 128
  --ffn-hidden-size 512
  --num-attention-heads 8
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size "$CP_SIZE"
  # Use the production-style rotary position embedding path. In THD mode,
  # PackedSeqParams makes RoPE restart at each packed sample boundary.
  --use-rotary-position-embeddings
  # Flash/fused packed-attention implementations require a low-precision dtype.
  # Use BF16 for both layout runs so the attention-backend selection is identical.
  --bf16
  # THD has fewer physical microbatches than BSHD. Disable stochastic layers so
  # both layouts remain numerically comparable under the same model initialization.
  --hidden-dropout 0.0
  --attention-dropout 0.0
  --tokenizer-type NullTokenizer
  --vocab-size 32000
  --train-iters 1
  --lr 1e-4
  --seed 1234
  --disable-bias-linear
)

# Baseline BSHD run: one real sample per microbatch. When CP_SIZE>1, each sample is
# sliced with the same head-tail CP rule before entering the model.
torchrun --nproc_per_node="$WORLD_SIZE" tests/packing_grad_accum_runner.py \
  --packing-grad-data "$DATA_DIR/packing_bshd.npz" \
  --packing-grad-mode bshd \
  --packing-grad-output "$BSHD_OUT" \
  "${COMMON_ARGS[@]}"

# THD run: the same logical samples are first DP-sliced, then CP-sliced per sample,
# and finally packed locally into THD buckets on each rank.
torchrun --nproc_per_node="$WORLD_SIZE" tests/packing_grad_accum_runner.py \
  --packing-grad-data "$DATA_DIR/packing_thd.npz" \
  --packing-grad-mode thd \
  --packing-grad-output "$THD_OUT" \
  "${COMMON_ARGS[@]}"

# Compare per-rank losses and CP-group-summed gradient tensors. BSHD and THD
# distribute logical-sample token gradients differently among CP ranks, so a
# rank-local gradient is not a valid equivalence target when CP is enabled.
python tests/compare_grad_accum_runs.py \
  --left "$BSHD_OUT" \
  --right "$THD_OUT" \
  --cp-size "$CP_SIZE" \
  --atol 3e-4 \
  --rtol 0.0

echo "Packing gradient accumulation check passed for CP_SIZE=$CP_SIZE. Artifacts are in: $WORK_DIR"