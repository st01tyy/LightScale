#!/bin/bash

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
: "${GRAD_TEST_TOKENIZER:?Set GRAD_TEST_TOKENIZER to a local tokenizer directory}"

    # --num-experts 4 \
    # --expert-model-parallel-size 1 \
    # --moe-router-topk 2 \
    # --moe-ffn-hidden-size 512 \
    # --moe-router-load-balancing-type aux_loss \
    # --moe-aux-loss-coeff 1e-3 \

torchrun --nproc_per_node=8 tests/grad_accum_equiv.py \
    --grad-test-data "${GRAD_TEST_DATA:-tests/generated_grad_test_data.npz}" \
    --grad-test-schedule 4,2,2,1,1 \
    --grad-test-output "${GRAD_TEST_OUTPUT:-tests/grad_test_output}" \
    --micro-batch-size 2 --global-batch-size 16 --seq-length 2048 \
    --num-layers 2 --hidden-size 2048 --num-attention-heads 16 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --max-position-embeddings 2048 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$GRAD_TEST_TOKENIZER" \
    --train-iters 10 \
    --lr 1e-4 \
    --disable-bias-linear \
    --num-experts 4 \
    --expert-model-parallel-size 1 \
    --moe-router-topk 2 \
    --moe-ffn-hidden-size 512 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-3
