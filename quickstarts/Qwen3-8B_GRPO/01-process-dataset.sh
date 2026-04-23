#!/bin/bash

set -x
set -e

DATASET_PATH="$1"

cd $(dirname "$0")

python3 ./process_deepscaler_for_ds_r1_zero.py \
    --input "$DATASET_PATH" \
    --output ./deepscaler_r1_zero

echo "done"