#!/bin/bash

set -x
set -e

MODEL_PATH="$1"
DATASET_PATH="$2"

cd $(dirname "$0")

python3 ./convert_metamathqa_to_qwen3.py \
    --input-path "$DATASET_PATH" \
    --output-path "./MetaMathQA_Qwen3" \
    --tokenizer "$MODEL_PATH"

python3 ../../tools/pack_hf_datasets.py \
    --input-path "./MetaMathQA_Qwen3" \
    --target-length-in-k 8 \
    --pad-token-id 151643 \
    --batch-size 4

echo "done"