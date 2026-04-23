#!/bin/bash
set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CONVERTOR_DIR}))
# export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250328:${CONVERTOR_DIR}/impl:$PYTHONPATH
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${CONVERTOR_DIR}/impl:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

# Environment variables for distributed training
# NUM_NODES=${WORLD_SIZE:-1}
NUM_NODES=1
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12560}

# Script arguments
LOAD_DIR=$1
SAVE_DIR=$2
MG2HF=$3
USE_CUDA=$4
PR=$5
TENSOR_PARALLEL_SIZE=$6
PIPELINE_PARALLEL_SIZE=$7
HF_DIR=$8

# Validate required arguments
if [ -z "$LOAD_DIR" ] || [ -z "$SAVE_DIR" ] || [ -z "$MG2HF" ] || [ -z "$USE_CUDA" ] || [ -z "$PR" ] || [ -z "$TENSOR_PARALLEL_SIZE" ] || [ -z "$PIPELINE_PARALLEL_SIZE" ]; then
    echo "Usage: $0 <LOAD_DIR> <SAVE_DIR> <MG2HF> <USE_CUDA> <PR> <TP> <PP> [HF_DIR]"
    echo "  LOAD_DIR: Path to load checkpoints from"
    echo "  SAVE_DIR: Path to save converted checkpoints"
    echo "  MG2HF: true/false - Convert from Megatron to HuggingFace format"
    echo "  USE_CUDA: true/false - Use GPU for conversion"
    echo "  PR: fp16/bf16 - Precision"
    echo "  TP: Tensor model parallel size (e.g., 1, 2, 4, 8)"
    echo "  PP: Pipeline model parallel size (e.g., 1, 2, 4, 8)"
    echo "  HF_DIR: HuggingFace model directory (optional, defaults to LOAD_DIR)"
    exit 1
fi

# Set HF_DIR default if not provided
if [ -z "$HF_DIR" ]; then
    HF_DIR=$LOAD_DIR
fi

# Validate parallel configuration
if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || [ "$TENSOR_PARALLEL_SIZE" -lt 1 ]; then
    echo "Error: Tensor parallel size must be a positive integer"
    exit 1
fi

if ! [[ "$PIPELINE_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || [ "$PIPELINE_PARALLEL_SIZE" -lt 1 ]; then
    echo "Error: Pipeline parallel size must be a positive integer"
    exit 1
fi

echo "Parallel Configuration:"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "  Total Model Parallel Size: $((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))"

# Function to check if jq is installed
check_jq() {
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is not installed. Please install jq to parse JSON files."
        echo "On macOS: brew install jq"
        echo "On Ubuntu/Debian: sudo apt-get install jq"
        echo "On CentOS/RHEL: sudo yum install jq"
        exit 1
    fi
}

# Function to read model config from HuggingFace config.json
read_model_config() {
    local config_dir=$1
    local config_file="${config_dir}/config.json"
    
    if [ ! -f "$config_file" ]; then
        echo "Error: config.json not found in $config_dir"
        exit 1
    fi
    
    echo "Reading model configuration from: $config_file"
    
    # Read basic model parameters
    NUM_LAYERS=$(jq -r '.num_hidden_layers // empty' "$config_file")
    HIDDEN_SIZE=$(jq -r '.hidden_size // empty' "$config_file")
    FFN_HIDDEN_SIZE=$(jq -r '.intermediate_size // empty' "$config_file")
    NUM_ATTENTION_HEADS=$(jq -r '.num_attention_heads // empty' "$config_file")
    NUM_KEY_VALUE_HEADS=$(jq -r '.num_key_value_heads // empty' "$config_file")
    MAX_POSITION_EMBEDDINGS=$(jq -r '.max_position_embeddings // empty' "$config_file")
    VOCAB_SIZE=$(jq -r '.vocab_size // empty' "$config_file")
    RMS_NORM_EPS=$(jq -r '.rms_norm_eps // empty' "$config_file")
    ROPE_THETA=$(jq -r '.rope_theta // empty' "$config_file")
    TIE_WORD_EMBEDDINGS=$(jq -r '.tie_word_embeddings // empty' "$config_file")
    
    # Qwen3 specific parameters
    HEAD_DIM=$(jq -r '.head_dim // empty' "$config_file")
    ATTENTION_BIAS=$(jq -r '.attention_bias // empty' "$config_file")

    ATTENTION_DROPOUT=$(jq -r '.attention_dropout // empty' "$config_file")
    
    # Handle rope_scaling if present
    ROPE_SCALING_FACTOR=$(jq -r '.rope_scaling.factor // empty' "$config_file")
    
    # Validate required parameters
    if [ -z "$NUM_LAYERS" ] || [ -z "$HIDDEN_SIZE" ] || [ -z "$FFN_HIDDEN_SIZE" ] || [ -z "$NUM_ATTENTION_HEADS" ]; then
        echo "Error: Missing required model parameters in config.json"
        echo "Required: num_hidden_layers, hidden_size, intermediate_size, num_attention_heads"
        exit 1
    fi
    
    # Set defaults for optional parameters
    NUM_KEY_VALUE_HEADS=${NUM_KEY_VALUE_HEADS:-$NUM_ATTENTION_HEADS}
    MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-32768}
    VOCAB_SIZE=${VOCAB_SIZE:-151936}
    RMS_NORM_EPS=${RMS_NORM_EPS:-1e-6}
    ROPE_THETA=${ROPE_THETA:-10000.0}
    TIE_WORD_EMBEDDINGS=${TIE_WORD_EMBEDDINGS:-false}
    HEAD_DIM=${HEAD_DIM:-128}
    ATTENTION_BIAS=${ATTENTION_BIAS:-false}

    ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.0}
    
    echo "Model Configuration:"
    echo "  Layers: $NUM_LAYERS"
    echo "  Hidden Size: $HIDDEN_SIZE"
    echo "  FFN Hidden Size: $FFN_HIDDEN_SIZE"
    echo "  Attention Heads: $NUM_ATTENTION_HEADS"
    echo "  Key-Value Heads: $NUM_KEY_VALUE_HEADS"
    echo "  Max Position Embeddings: $MAX_POSITION_EMBEDDINGS"
    echo "  Vocab Size: $VOCAB_SIZE"
    echo "  RMS Norm Eps: $RMS_NORM_EPS"
    echo "  RoPE Theta: $ROPE_THETA"
    echo "  Tie Word Embeddings: $TIE_WORD_EMBEDDINGS"
    echo "  Head Dim: $HEAD_DIM"
    echo "  Attention Bias: $ATTENTION_BIAS"
    if [ -n "$ROPE_SCALING_FACTOR" ]; then
        echo "  RoPE Scaling Factor: $ROPE_SCALING_FACTOR"
    fi
}

# Function to use original vocab size as padded vocab size
use_original_vocab_size() {
    local vocab_size=$1
    
    # Use original vocab_size directly without padding
    echo $vocab_size
}

# Check if jq is available
check_jq

# Read model configuration from JSON
if [ "$MG2HF" = true ]; then
    read_model_config "$HF_DIR"
else
    read_model_config "$LOAD_DIR"
fi

# Set up model parallel configuration
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE
)

PADDED_VOCAB_SIZE=$(use_original_vocab_size $VOCAB_SIZE)

# Build OTHER_ARGS based on conversion direction
OTHER_ARGS=()
if [ ${MG2HF} = true ]; then
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${HF_DIR}
        --hf-dir ${HF_DIR}
        --mcore2hf
    )
    mkdir -p ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
else
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${LOAD_DIR}
    )
    mkdir -p ${SAVE_DIR}
    # find -L ${LOAD_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    # find -L ${LOAD_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
fi

# Add GPU usage flag
if [ ${USE_CUDA} = true ]; then
    OTHER_ARGS+=(
        --use-gpu
    )
fi

# Add precision flags
if [ ${PR} = fp16 ]; then
    OTHER_ARGS+=(
        --fp16
    )
elif [ ${PR} = bf16 ]; then
    OTHER_ARGS+=(
        --bf16
    )
fi

# Validate distributed training environment
if [ -z ${NUM_NODES} ]; then
    echo "Please Provide WORLD_SIZE"
    exit 1
fi

if [ -z ${NODE_RANK} ]; then
    echo "Please Provide RANK"
    exit 1
fi

if [ -z ${MASTER_ADDR} ]; then
    echo "Please Provide MASTER_ADDR"
    exit 1
fi

if [ -z ${MASTER_PORT} ]; then
    echo "Please Provide MASTER_PORT"
    exit 1
fi

# Distributed training arguments
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# GPT model arguments - dynamically built from config.json for Qwen3
GPT_MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --num-query-groups $NUM_KEY_VALUE_HEADS
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --padded-vocab-size $PADDED_VOCAB_SIZE
    --norm-epsilon $RMS_NORM_EPS
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --seq-length 1
    --attention-backend auto
    --position-embedding-type rope
    --group-query-attention
    --kv-channels $HEAD_DIM
    --qk-layernorm
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --rotary-base $(printf "%.0f" "$ROPE_THETA")
)

# Add attention bias flag if enabled in config
if [ "$ATTENTION_BIAS" = "true" ]; then
    GPT_MODEL_ARGS+=(--add-qkv-bias)
fi

# Add untie embeddings flag if needed
if [ "$TIE_WORD_EMBEDDINGS" = "false" ]; then
    GPT_MODEL_ARGS+=(--untie-embeddings-and-output-weights)
fi

# # Add attention dropout if specified
# if [ -n "$ATTENTION_DROPOUT" ]; then
#     GPT_MODEL_ARGS+=(--attention-dropout $ATTENTION_DROPOUT)
# fi

# Add RoPE scaling if present
if [ -n "$ROPE_SCALING_FACTOR" ]; then
    GPT_MODEL_ARGS+=(
        --use-rope-scaling
        --rope-scaling-factor $ROPE_SCALING_FACTOR
    )
fi

# Training arguments
# TRAINING_ARGS=(
#     --micro-batch-size 1 
#     --global-batch-size 1024
#     --train-iters 500000 
#     --weight-decay 0.1 
#     --adam-beta1 0.9 
#     --adam-beta2 0.95 
#     --init-method-std 0.006 
#     --clip-grad 1.0 
#     --bf16
#     --lr 6.0e-5 
#     --lr-decay-style cosine 
#     --min-lr 6.0e-6
#     --lr-warmup-fraction .001 
#     --lr-decay-iters 430000 
# )
TRAINING_ARGS=(
    --micro-batch-size 1 
    --train-iters 1
    --bf16
)

# Evaluation and logging arguments
# EVAL_AND_LOGGING_ARGS=(
#     --log-interval 100
#     --save-interval 10000 
#     --eval-interval 1000 
#     --eval-iters 10
# )

# Conversion arguments
CONVERT_ARGS=(
    --model-type GPT 
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}
    --no-load-optim
    --no-load-rng
    --logging-level 1
)

# Build and execute the command
cmd="torchrun ${DISTRIBUTED_ARGS[@]} tools/distributed_checkpoints_convertor/impl/convert.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${OTHER_ARGS[@]}"

echo "Executing conversion command:"
echo $cmd
echo ""
eval $cmd
