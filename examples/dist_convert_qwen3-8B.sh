#!/bin/bash

bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_dense_single_node.sh \
    hf_models/Qwen3-8B \
    megatron_models/Qwen3-8B \
    false \
    true \
    bf16 \
    1 \
    4

# bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_dense_single_node.sh \
#     megatron_models/Qwen3-8B \
#     hf_models/Qwen3-8B-converted \
#     true \
#     true \
#     bf16 \
#     1 \
#     4 \
#     hf_models/Qwen3-8B