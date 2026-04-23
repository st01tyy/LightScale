#!/bin/bash

set -x
set -e

BASE_MODEL_PATH="$1"

cd $(dirname "$0")

mkdir -p ./training_outputs/hf_checkpoints/

cd ../../

bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_dense_single_node.sh \
    ./quickstarts/Qwen3-8B_GRPO/training_outputs/checkpoints \
    ./quickstarts/Qwen3-8B_GRPO/training_outputs/hf_checkpoints/step_300 \
    true \
    true \
    bf16 \
    1 \
    1 \
    "$BASE_MODEL_PATH"