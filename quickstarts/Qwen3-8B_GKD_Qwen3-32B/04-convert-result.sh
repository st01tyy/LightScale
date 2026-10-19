#!/bin/bash

set -x
set -e

BASE_MODEL_PATH="$1"

cd $(dirname "$0")

mkdir -p ./training_outputs/hf_checkpoints/

cd ../../

bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_dense_single_node.sh \
    ./quickstarts/Qwen3-8B_GKD_Qwen3-32B/training_outputs/checkpoints \
    ./quickstarts/Qwen3-8B_GKD_Qwen3-32B/training_outputs/hf_checkpoints/step_600 \
    true \
    true \
    bf16 \
    1 \
    1 \
    "$BASE_MODEL_PATH"