#!/bin/bash

set -x
set -e

# YOU SHOULD DEFINE AND CONSTRUCT ENV NODE_RANK AND NODE_LIST HERE

cd $(dirname "$0")
cd ../../

python3 headquarters_v2.py --config ./quickstarts/Qwen3-8B_GKD_Qwen3-32B/config.yml

