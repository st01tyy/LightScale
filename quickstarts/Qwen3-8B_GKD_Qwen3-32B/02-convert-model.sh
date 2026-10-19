#!/bin/bash

set -x
set -e

STUDENT_PATH="$1"
TEACHER_PATH="$2"

cd $(dirname "$0")

cd ../../

mkdir -p ./hf_models/
mkdir -p ./megatron_models/

ln -s "$STUDENT_PATH" ./hf_models/Qwen3-8B
ln -s "$TEACHER_PATH" ./hf_models/Qwen3-32B

bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_dense_single_node.sh \
    "$STUDENT_PATH" \
    ./megatron_models/Qwen3-8B \
    false \
    true \
    bf16 \
    1 \
    1

bash tools/distributed_checkpoints_convertor/scripts_mrl/run_qwen3_dense_single_node.sh \
    "$TEACHER_PATH" \
    ./megatron_models/Qwen3-32B \
    false \
    true \
    bf16 \
    1 \
    8