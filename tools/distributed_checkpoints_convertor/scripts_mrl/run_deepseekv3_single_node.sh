#!/bin/bash
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
NODE_RANK=0
GPUS_PER_NODE=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12560

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

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE
    --expert-model-parallel-size $EXPERT_PARALLEL_SIZE
)

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

read_model_config() {
    local config_dir=$1
    local config_file="${config_dir}/config.json"

    if [ ! -f "$config_file" ]; then
        echo "Error: config.json not found in $config_dir"
        exit 1
    fi

    echo "Reading model configuration from: $config_file"

    # basic model parameters
    NUM_LAYERS=$(jq -r '.num_hidden_layers // empty' "$config_file")
    HIDDEN_SIZE=$(jq -r '.hidden_size // empty' "$config_file")
    FFN_HIDDEN_SIZE=$(jq -r '.intermediate_size // empty' "$config_file")
    NUM_ATTENTION_HEADS=$(jq -r '.num_attention_heads // empty' "$config_file")
    NUM_KEY_VALUE_HEADS=$(jq -r '.num_key_value_heads // empty' "$config_file")
    MAX_POSITION_EMBEDDINGS=$(jq -r '.max_position_embeddings // empty' "$config_file")
    VOCAB_SIZE=$(jq -r '.vocab_size // empty' "$config_file")
    RMS_NORM_EPS=$(jq -r '.rms_norm_eps // empty' "$config_file")
    ROPE_THETA=$(jq -r '.rope_theta // empty' "$config_file")
    TIE_WORD_EMBEDDINGS=$(jq -r '.tie_word_embeddings // "false"' "$config_file")

    # moe parameters
    MOE_INTERMEDIATE_SIZE=$(jq -r '.moe_intermediate_size // empty' "$config_file")
    N_ROUTED_EXPERTS=$(jq -r '.n_routed_experts // empty' "$config_file")
    N_SHARED_EXPERTS=$(jq -r '.n_shared_experts // empty' "$config_file")
    NUM_EXPERTS_PER_TOK=$(jq -r '.num_experts_per_tok // empty' "$config_file")
    TOPK_GROUP=$(jq -r '.topk_group // empty' "$config_file")
    N_GROUP=$(jq -r '.n_group // empty' "$config_file")
    FIRST_K_DENSE=$(jq -r '.first_k_dense_replace // empty' "$config_file")
    ROUTED_SCALING_FACTOR=$(jq -r '.routed_scaling_factor // empty' "$config_file")

    # mla parameters
    Q_LORA_RANK=$(jq -r '.q_lora_rank // empty' "$config_file")
    KV_LORA_RANK=$(jq -r '.kv_lora_rank // empty' "$config_file")
    V_HEAD_DIM=$(jq -r '.v_head_dim // empty' "$config_file")
    QK_ROPE_HEAD_DIM=$(jq -r '.qk_rope_head_dim // empty' "$config_file")
    QK_NOPE_HEAD_DIM=$(jq -r '.qk_nope_head_dim // empty' "$config_file")

    # mtp parameters
    NUM_MTP_LAYERS=$(jq -r '.num_nextn_predict_layers // empty' "$config_file")

    local required_vars=(
        NUM_LAYERS
        HIDDEN_SIZE
        FFN_HIDDEN_SIZE
        NUM_ATTENTION_HEADS
        NUM_KEY_VALUE_HEADS
        MAX_POSITION_EMBEDDINGS
        VOCAB_SIZE
        RMS_NORM_EPS
        ROPE_THETA
        TIE_WORD_EMBEDDINGS
        MOE_INTERMEDIATE_SIZE
        N_ROUTED_EXPERTS
        N_SHARED_EXPERTS
        NUM_EXPERTS_PER_TOK
        TOPK_GROUP
        N_GROUP
        FIRST_K_DENSE
        ROUTED_SCALING_FACTOR
        Q_LORA_RANK
        KV_LORA_RANK
        V_HEAD_DIM
        QK_ROPE_HEAD_DIM
        QK_NOPE_HEAD_DIM
        NUM_MTP_LAYERS
    )
    local missing_fields=()
    for field in "${required_vars[@]}"; do
        if [ -z "${!field}" ]; then
            missing_fields+=("$field")
        fi
    done
    if [ "${#missing_fields[@]}" -ne 0 ]; then
        echo "Error: Missing required fields in config.json: ${missing_fields[*]}"
        exit 1
    fi

    ATTENTION_BIAS=$(jq -r '.attention_bias // "false"' "$config_file")
    if [ "$ATTENTION_BIAS" = "true" ]; then
        echo "attention_bias is true, but only false is supported currently."
        exit 1
    fi

    # rope scaling    
    ROPE_SCALING_JSON=$(jq -c '.rope_scaling // empty' "$config_file")

    ROPE_SCALING_TYPE=
    ROPE_SCALING_FACTOR=
    ROPE_MSCALE=
    ROPE_MSCALE_ALL_DIM=
    BETA_FAST=
    BETA_SLOW=
    if [ -n "$ROPE_SCALING_JSON" ] && [ "$ROPE_SCALING_JSON" != "null" ]; then
        ROPE_SCALING_TYPE=$(echo "$ROPE_SCALING_JSON" | jq -r '.rope_type // .type // empty')
        ROPE_SCALING_FACTOR=$(echo "$ROPE_SCALING_JSON" | jq -r '.factor // empty')
        ROPE_MSCALE=$(echo "$ROPE_SCALING_JSON" | jq -r '.mscale // empty')
        ROPE_MSCALE_ALL_DIM=$(echo "$ROPE_SCALING_JSON" | jq -r '.mscale_all_dim // empty')
        BETA_FAST=$(echo "$ROPE_SCALING_JSON" | jq -r '.beta_fast // empty')
        BETA_SLOW=$(echo "$ROPE_SCALING_JSON" | jq -r '.beta_slow // empty')
    else
        ROPE_SCALING_TYPE="rope"
        ROPE_SCALING_FACTOR="1"
        ROPE_MSCALE="1"
        ROPE_MSCALE_ALL_DIM="1"
        BETA_FAST="32"
        BETA_SLOW="1"
    fi
}

check_jq

if [ "$MG2HF" = true ]; then
    read_model_config "$HF_DIR"
else
    read_model_config "$LOAD_DIR"
fi

if [ "$FIRST_K_DENSE" -gt "$NUM_LAYERS" ]; then
    echo "Error: first_k_dense_replace ($FIRST_K_DENSE) cannot be greater than num_hidden_layers ($NUM_LAYERS)"
    exit 1
fi
MOE_LAYERS=$(( NUM_LAYERS - FIRST_K_DENSE ))
MOE_LAYER_FREQ="\"[0]*${FIRST_K_DENSE}+[1]*${MOE_LAYERS}\""

echo "Model Configuration Summary:"
cat <<EOF
    num_hidden_layers: $NUM_LAYERS
    hidden_size: $HIDDEN_SIZE
    ffn_hidden_size: $FFN_HIDDEN_SIZE
    moe_ffn_hidden_size: $MOE_INTERMEDIATE_SIZE
    num_attention_heads: $NUM_ATTENTION_HEADS
    num_query_groups: $NUM_KEY_VALUE_HEADS
    max_position_embeddings: $MAX_POSITION_EMBEDDINGS
    vocab_size: $VOCAB_SIZE
    rms_norm_eps: $RMS_NORM_EPS
    rope_theta: $ROPE_THETA
    rope_scaling_type: ${ROPE_SCALING_TYPE}
    rope_scaling_factor: ${ROPE_SCALING_FACTOR}
    rope_mscale: ${ROPE_MSCALE}
    rope_mscale_all_dim: ${ROPE_MSCALE_ALL_DIM}
    beta_fast: ${BETA_FAST}
    beta_slow: ${BETA_SLOW}
    tie_word_embeddings: $TIE_WORD_EMBEDDINGS
    attention_bias: $ATTENTION_BIAS
    num_routed_experts: $N_ROUTED_EXPERTS
    num_shared_experts: $N_SHARED_EXPERTS
    num_experts_per_tok: $NUM_EXPERTS_PER_TOK
    topk_group: $TOPK_GROUP
    n_group: $N_GROUP
    first_k_dense_replace: $FIRST_K_DENSE
    routed_scaling_factor: $ROUTED_SCALING_FACTOR
    q_lora_rank: $Q_LORA_RANK
    kv_lora_rank: $KV_LORA_RANK
    v_head_dim: $V_HEAD_DIM
    qk_rope_head_dim: $QK_ROPE_HEAD_DIM
    qk_nope_head_dim: $QK_NOPE_HEAD_DIM
    moe_layers: $MOE_LAYERS
    moe_layer_freq: $MOE_LAYER_FREQ
EOF

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
fi

if [ ${USE_CUDA} = true ]; then
    OTHER_ARGS+=(--use-gpu)
fi

if [ ${PR} = fp16 ]; then
    OTHER_ARGS+=(--fp16)
elif [ ${PR} = bf16 ]; then
    OTHER_ARGS+=(--bf16)
fi

if [ -z ${NUM_NODES} ]; then
    echo "Please provide WORLD_SIZE"
    exit 1
fi
if [ -z ${NODE_RANK} ]; then
    echo "Please provide RANK"
    exit 1
fi
if [ -z ${MASTER_ADDR} ]; then
    echo "Please provide MASTER_ADDR"
    exit 1
fi
if [ -z ${MASTER_PORT} ]; then
    echo "Please provide MASTER_PORT"
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --model-type GPT
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --num-query-groups $NUM_KEY_VALUE_HEADS
    --seq-length 1
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --padded-vocab-size $VOCAB_SIZE
    --norm-epsilon $RMS_NORM_EPS
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --attention-backend auto
    --qk-layernorm
    --no-rope-fusion
    --mtp-num-layers 1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --mtp-num-layers $NUM_MTP_LAYERS
)

if [ "$TIE_WORD_EMBEDDINGS" = false ]; then
    GPT_MODEL_ARGS+=(--untie-embeddings-and-output-weights)
fi

MOE_ARGS=(
    --num-experts $N_ROUTED_EXPERTS
    --moe-ffn-hidden-size $MOE_INTERMEDIATE_SIZE
    --moe-shared-expert-intermediate-size $(( MOE_INTERMEDIATE_SIZE * N_SHARED_EXPERTS ))
    --moe-router-topk-scaling-factor $ROUTED_SCALING_FACTOR
    --moe-router-score-function sigmoid
    --moe-token-dispatcher-type alltoall
    --moe-router-topk $NUM_EXPERTS_PER_TOK
    --moe-router-group-topk $TOPK_GROUP
    --moe-router-num-groups $N_GROUP
    --moe-router-enable-expert-bias
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-load-balancing-type seq_aux_loss
)
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

MLA_ARGS=(
    --multi-latent-attention
    --q-lora-rank $Q_LORA_RANK
    --kv-lora-rank $KV_LORA_RANK
    --v-head-dim $V_HEAD_DIM
)

POSITION_EMBED_ARGS=(
    --rotary-base $(printf "%.0f" "$ROPE_THETA")
    --position-embedding-type ${ROPE_SCALING_TYPE}
    --rotary-scaling-factor ${ROPE_SCALING_FACTOR}
    --mscale ${ROPE_MSCALE}
    --mscale-all-dim ${ROPE_MSCALE_ALL_DIM}
    --beta-fast ${BETA_FAST}
    --beta-slow ${BETA_SLOW}
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --train-iters 1
    --bf16
)

CONVERT_ARGS=(
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}
    --no-load-optim
    --no-load-rng
    --synchronizer deepseek_v3
    --logging-level 1
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} tools/distributed_checkpoints_convertor/impl/convert.py \
    ${GPT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${POSITION_EMBED_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${OTHER_ARGS[@]}"

echo "$cmd"
eval $cmd
