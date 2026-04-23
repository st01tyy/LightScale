#!/bin/bash

bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_moe_single_node.sh \
    hf_models/Qwen3-30B-A3B-Instruct-2507 \
    megatron_models/Qwen3-30B-A3B-Instruct-2507-GEMM \
    false \
    true \
    bf16 \
    1 \
    1 \
    hf_models/Qwen3-30B-A3B-Instruct-2507 \
    8 \
    true

# bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_moe_single_node.sh \
#     megatron_models/Qwen3-30B-A3B-Instruct-2507 \
#     hf_models/Qwen3-30B-A3B-Instruct-2507-converted \
#     true \
#     true \
#     bf16 \
#     1 \
#     1 \
#     hf_models/Qwen3-30B-A3B-Instruct-2507 \
#     8