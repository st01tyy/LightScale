#!/bin/bash

python3 tests/generate_grad_test_data.py \
    --output tests/generated_grad_test_data.npz \
    --num-microbatches 10 \
    --micro-batch-size 2 \
    --seq-length 2048 \
    --vocab-size 50304 \
    --seed 1234