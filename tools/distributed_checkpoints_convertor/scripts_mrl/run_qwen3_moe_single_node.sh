#!/bin/bash
# set -x
set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CONVERTOR_DIR}))
# export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250707:${CONVERTOR_DIR}/impl:$PYTHONPATH
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
EXPERT_PARALLEL_SIZE=$9
MOE_GROUPED_GEMM=${10}

# Validate required arguments
if [ -z "$LOAD_DIR" ] || [ -z "$SAVE_DIR" ] || [ -z "$MG2HF" ] || [ -z "$USE_CUDA" ] || [ -z "$PR" ] || [ -z "$TENSOR_PARALLEL_SIZE" ] || [ -z "$PIPELINE_PARALLEL_SIZE" ]; then
    echo "Usage: $0 <LOAD_DIR> <SAVE_DIR> <MG2HF> <USE_CUDA> <PR> <TP> <PP> [HF_DIR] [EP] [MOE_GROUPED_GEMM]"
    echo "  LOAD_DIR: Path to load checkpoints from"
    echo "  SAVE_DIR: Path to save converted checkpoints"
    echo "  MG2HF: true/false - Convert from Megatron to HuggingFace format"
    echo "  USE_CUDA: true/false - Use GPU for conversion"
    echo "  PR: fp16/bf16 - Precision"
    echo "  TP: Tensor model parallel size (e.g., 1, 2, 4, 8)"
    echo "  PP: Pipeline model parallel size (e.g., 1, 2, 4, 8)"
    echo "  HF_DIR: HuggingFace model directory (optional, defaults to LOAD_DIR)"
    echo "  EP: Expert model parallel size (optional, defaults to TP)"
    echo "  MOE_GROUPED_GEMM: true/false - Enable MoE grouped GEMM (optional, defaults to false)"
    exit 1
fi

# Set HF_DIR default if not provided
if [ -z "$HF_DIR" ]; then
    HF_DIR=$LOAD_DIR
fi

# Set MOE_GROUPED_GEMM default if not provided
if [ -z "$MOE_GROUPED_GEMM" ]; then
    MOE_GROUPED_GEMM=false
fi

# Validate MOE_GROUPED_GEMM parameter
if [ "$MOE_GROUPED_GEMM" != "true" ] && [ "$MOE_GROUPED_GEMM" != "false" ]; then
    echo "Error: MOE_GROUPED_GEMM must be 'true' or 'false'"
    exit 1
fi

# Set EXPERT_PARALLEL_SIZE default if not provided
if [ -z "$EXPERT_PARALLEL_SIZE" ]; then
    EXPERT_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
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

if ! [[ "$EXPERT_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || [ "$EXPERT_PARALLEL_SIZE" -lt 1 ]; then
    echo "Error: Expert parallel size must be a positive integer"
    exit 1
fi

echo "Parallel Configuration:"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE"
echo "  Expert Parallel Size: $EXPERT_PARALLEL_SIZE"
echo "  Total Model Parallel Size: $((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))"
echo "  MoE Grouped GEMM: $MOE_GROUPED_GEMM"

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

# Function to read model config from HuggingFace config.json for Qwen3 MoE
read_model_config() {
    local config_dir=$1
    local config_file="${config_dir}/config.json"
    
    if [ ! -f "$config_file" ]; then
        echo "Error: config.json not found in $config_dir"
        exit 1
    fi
    
    echo "Reading Qwen3 MoE model configuration from: $config_file"
    
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
    
    # MoE specific parameters
    NUM_EXPERTS=$(jq -r '.num_experts // empty' "$config_file")
    NUM_EXPERTS_PER_TOK=$(jq -r '.num_experts_per_tok // empty' "$config_file")
    MOE_INTERMEDIATE_SIZE=$(jq -r '.moe_intermediate_size // empty' "$config_file")
    DECODER_SPARSE_STEP=$(jq -r '.decoder_sparse_step // empty' "$config_file")
    NORM_TOPK_PROB=$(jq -r '.norm_topk_prob // empty' "$config_file")
    OUTPUT_ROUTER_LOGITS=$(jq -r '.output_router_logits // empty' "$config_file")
    ROUTER_AUX_LOSS_COEF=$(jq -r '.router_aux_loss_coef // empty' "$config_file")
    
    # MLP only layers (array) - handle as JSON string to preserve array structure
    MLP_ONLY_LAYERS=$(jq -c '.mlp_only_layers // []' "$config_file")
    
    # Attention dropout (sliding window attention is not used in Qwen3)
    ATTENTION_DROPOUT=$(jq -r '.attention_dropout // empty' "$config_file")
    
    # Handle rope_scaling if present
    ROPE_SCALING_FACTOR=$(jq -r '.rope_scaling.factor // empty' "$config_file")
    ROPE_SCALING_TYPE=$(jq -r '.rope_scaling.rope_type // empty' "$config_file")
    
    # Validate required parameters
    if [ -z "$NUM_LAYERS" ] || [ -z "$HIDDEN_SIZE" ] || [ -z "$FFN_HIDDEN_SIZE" ] || [ -z "$NUM_ATTENTION_HEADS" ]; then
        echo "Error: Missing required model parameters in config.json"
        echo "Required: num_hidden_layers, hidden_size, intermediate_size, num_attention_heads"
        exit 1
    fi
    
    # Validate MoE specific parameters
    if [ -z "$NUM_EXPERTS" ] || [ -z "$NUM_EXPERTS_PER_TOK" ]; then
        echo "Error: Missing required MoE parameters in config.json"
        echo "Required for MoE: num_experts, num_experts_per_tok"
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
    
    # Set defaults for MoE parameters
    MOE_INTERMEDIATE_SIZE=${MOE_INTERMEDIATE_SIZE:-768}
    DECODER_SPARSE_STEP=${DECODER_SPARSE_STEP:-1}
    NORM_TOPK_PROB=${NORM_TOPK_PROB:-false}
    OUTPUT_ROUTER_LOGITS=${OUTPUT_ROUTER_LOGITS:-false}
    ROUTER_AUX_LOSS_COEF=${ROUTER_AUX_LOSS_COEF:-0.001}
    
    # Set defaults for attention dropout (sliding window attention not used in Qwen3)
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
    echo ""
    echo "MoE Configuration:"
    echo "  Number of Experts: $NUM_EXPERTS"
    echo "  Experts per Token: $NUM_EXPERTS_PER_TOK"
    echo "  MoE Intermediate Size: $MOE_INTERMEDIATE_SIZE"
    echo "  Decoder Sparse Step: $DECODER_SPARSE_STEP"
    echo "  Normalize TopK Probability: $NORM_TOPK_PROB"
    echo "  Output Router Logits: $OUTPUT_ROUTER_LOGITS"
    echo "  Router Aux Loss Coefficient: $ROUTER_AUX_LOSS_COEF"
    if [ "$MLP_ONLY_LAYERS" != "null" ] && [ -n "$MLP_ONLY_LAYERS" ]; then
        echo "  MLP Only Layers: $MLP_ONLY_LAYERS"
    fi
    echo ""
    echo "Attention Configuration:"
    echo "  Attention Dropout: $ATTENTION_DROPOUT"
    if [ -n "$ROPE_SCALING_FACTOR" ]; then
        echo "  RoPE Scaling Factor: $ROPE_SCALING_FACTOR"
        if [ -n "$ROPE_SCALING_TYPE" ]; then
            echo "  RoPE Scaling Type: $ROPE_SCALING_TYPE"
        fi
    fi
}

# Function to validate MoE layer configuration and construct moe-layer-freq
validate_and_construct_moe_layers() {
    local num_layers=$1
    local mlp_only_layers=$2
    local decoder_sparse_step=$3
    
    echo "🔍 Validating MoE layer configuration..."
    
    # Initialize moe_layer_freq array with all layers as MoE (1)
    local moe_layer_freq=()
    for ((i=0; i<num_layers; i++)); do
        moe_layer_freq[i]=1
    done
    
    # Process mlp_only_layers if provided
    if [ "$mlp_only_layers" != "null" ] && [ -n "$mlp_only_layers" ] && [ "$mlp_only_layers" != "[]" ]; then
        echo "  Processing MLP-only layers: $mlp_only_layers"
        
        # Parse JSON array and set corresponding layers to 0 (MLP-only)
        local mlp_layers_list=$(echo "$mlp_only_layers" | jq -r '.[]' 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$mlp_layers_list" ]; then
            while IFS= read -r layer_idx; do
                if [[ "$layer_idx" =~ ^[0-9]+$ ]] && [ "$layer_idx" -ge 0 ] && [ "$layer_idx" -lt "$num_layers" ]; then
                    moe_layer_freq[$layer_idx]=0
                    echo "    Layer $layer_idx: MLP-only"
                else
                    echo "    Warning: Invalid layer index $layer_idx (should be 0-$((num_layers-1)))"
                fi
            done <<< "$mlp_layers_list"
        else
            echo "    Warning: Failed to parse mlp_only_layers JSON array"
        fi
    fi
    
    # Apply decoder_sparse_step pattern if > 1
    if [ "$decoder_sparse_step" -gt 1 ]; then
        echo "  Applying decoder sparse step pattern: $decoder_sparse_step"
        for ((i=0; i<num_layers; i++)); do
            # Only modify layers that are not already set as MLP-only
            if [ "${moe_layer_freq[i]}" -eq 1 ]; then
                if [ $((i % decoder_sparse_step)) -ne 0 ]; then
                    moe_layer_freq[i]=0
                    echo "    Layer $i: Changed to MLP-only due to sparse step"
                fi
            fi
        done
    fi
    
    # Count MoE vs MLP layers
    local moe_count=0
    local mlp_count=0
    for freq in "${moe_layer_freq[@]}"; do
        # echo $freq
        if [ "$freq" -eq 1 ]; then
            moe_count=$((moe_count + 1))
        else
            mlp_count=$((mlp_count + 1))
        fi
    done
    
    echo "  📊 Layer distribution:"
    echo "    MoE layers: $moe_count"
    echo "    MLP-only layers: $mlp_count"
    echo "    Total layers: $num_layers"
    
    # Construct comma-separated frequency string
    local freq_string="["
    for ((i=0; i<num_layers; i++)); do
        if [ $i -eq 0 ]; then
            freq_string="$freq_string${moe_layer_freq[i]}"
        else
            freq_string="$freq_string,${moe_layer_freq[i]}"
        fi
    done
    freq_string="$freq_string]"
    
    echo "  🔧 Generated moe-layer-freq: $freq_string"
    
    # Sanity checks
    if [ "$moe_count" -eq 0 ]; then
        echo "  ⚠️  Warning: No MoE layers found. This might not be a valid MoE configuration."
    fi
    
    if [ "$decoder_sparse_step" -gt 1 ] && [ "$moe_count" -ne $(( (num_layers + decoder_sparse_step - 1) / decoder_sparse_step )) ]; then
        echo "  ⚠️  Warning: MoE layer count doesn't match expected pattern for decoder_sparse_step=$decoder_sparse_step"
    fi
    
    # Export the frequency string for use in MOE_ARGS
    export MOE_LAYER_FREQUENCY="$freq_string"
    echo "✅ MoE layer validation completed"
    echo ""
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

# Validate MoE layer configuration and construct moe-layer-freq
validate_and_construct_moe_layers "$NUM_LAYERS" "$MLP_ONLY_LAYERS" "$DECODER_SPARSE_STEP"

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

# GPT model arguments - dynamically built from config.json for Qwen3 MoE
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

# Add MoE specific arguments
MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --expert-model-parallel-size $EXPERT_PARALLEL_SIZE
    --moe-router-topk $NUM_EXPERTS_PER_TOK
    --moe-ffn-hidden-size $MOE_INTERMEDIATE_SIZE
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff $ROUTER_AUX_LOSS_COEF
)
# Add MoE grouped GEMM flag if enabled
if [ "$MOE_GROUPED_GEMM" = "true" ]; then
    MOE_ARGS+=(--moe-grouped-gemm)
    echo "  🚀 MoE Grouped GEMM: ENABLED"
    echo "    Note: Requires bf16 precision and GPU compute capability >= 8.0"
    
    # Validate precision requirement for grouped GEMM
    if [ "$PR" != "bf16" ]; then
        echo "  ⚠️  Warning: MoE Grouped GEMM requires bf16 precision, but $PR is specified"
        echo "    Consider using bf16 for optimal performance with grouped GEMM"
    fi
else
    echo "  📝 MoE Grouped GEMM: DISABLED (using SequentialMLP)"
fi

# Add layer frequency pattern based on mlp_only_layers and decoder_sparse_step
if [ -n "$MOE_LAYER_FREQUENCY" ]; then
    MOE_ARGS+=(--moe-layer-freq $MOE_LAYER_FREQUENCY)
fi

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
    if [ -n "$ROPE_SCALING_TYPE" ] && [ "$ROPE_SCALING_TYPE" != "null" ]; then
        GPT_MODEL_ARGS+=(--rope-scaling-type $ROPE_SCALING_TYPE)
    fi
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
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${OTHER_ARGS[@]}"

echo "Executing Qwen3 MoE conversion command:"
echo $cmd
echo ""
eval $cmd
