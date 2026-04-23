#!/bin/bash

export PYTHONPATH="/root/work/externalstorage/gpfsprd/taoyuyang/mrl_gkd/Megatron-RL:$PYTHONPATH"

    # --num-experts 4 \
    # --expert-model-parallel-size 1 \
    # --moe-router-topk 2 \
    # --moe-ffn-hidden-size 512 \
    # --moe-router-load-balancing-type aux_loss \
    # --moe-aux-loss-coeff 1e-3 \

torchrun --nproc_per_node=8 tests/grad_accum_equiv.py \
    --grad-test-data /workspace/fixed_batch.npz \
    --grad-test-schedule 4,2,2,1,1 \
    --grad-test-output /workspace/grad_equiv_out \
    --micro-batch-size 2 --global-batch-size 16 --seq-length 2048 \
    --num-layers 2 --hidden-size 2048 --num-attention-heads 16 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --max-position-embeddings 2048 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /root/work/externalstorage/gpfsprd/taoyuyang/mrl_gkd/75bv2_annealing_cpt12000_lcdiv6_082523_cv3-epoch3 \
    --train-iters 10 \
    --lr 1e-4 \
    --disable-bias-linear \
    --num-experts 4 \
    --expert-model-parallel-size 1 \
    --moe-router-topk 2 \
    --moe-ffn-hidden-size 512 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-3 \