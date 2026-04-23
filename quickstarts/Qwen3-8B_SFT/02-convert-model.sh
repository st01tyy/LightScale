#!/bin/bash

set -x
set -e

MODEL_PATH="$1"

cd $(dirname "$0")

cd ../../

mkdir -p ./megatron_models/

bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_dense_single_node.sh \
    "$MODEL_PATH" \
    ./megatron_models/Qwen3-8B-Base \
    false \
    true \
    bf16 \
    1 \
    1