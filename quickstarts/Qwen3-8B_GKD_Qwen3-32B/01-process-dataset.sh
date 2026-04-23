#!/bin/bash

set -x
set -e

MODEL_PATH="$1"
DATASET_PATH="$2"

cd $(dirname "$0")

python3 ./process_deepscaler_for_gkd.py \
    --input "$DATASET_PATH" \
    --output ./deepscaler_gkd \
    --tokenizer "$MODEL_PATH"