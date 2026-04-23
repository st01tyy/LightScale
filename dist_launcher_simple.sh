#! /bin/bash

# 默认值
NPROC_PER_NODE=8

# 打印错误并退出
function error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# 解析参数
while [[ $# -gt 0 ]]; do
    if [[ "$1" != --* ]]; then
        break  # 遇到第一个不以“--”开头的参数，停止解析
    fi
    case "$1" in
        --MASTER_ADDR) MASTER_ADDR="$2"; shift 2 ;;
        --MASTER_PORT) MASTER_PORT="$2"; shift 2 ;;
        --WORLD_SIZE) WORLD_SIZE="$2"; shift 2 ;;
        --NPROC_PER_NODE) NPROC_PER_NODE="$2"; shift 2 ;;
        --NODE_RANK) NODE_RANK="$2"; shift 2 ;;
        --LOG_DIR) LOG_DIR="$2"; shift 2 ;;
        *) error_exit "Unknown parameter: $1" ;;
    esac
done

# 检查必传参数
[[ -z "$MASTER_ADDR" ]] && error_exit "MASTER_ADDR is required"
[[ -z "$MASTER_PORT" ]] && error_exit "MASTER_PORT is required"
[[ -z "$WORLD_SIZE" ]] && error_exit "WORLD_SIZE is required"
[[ -z "$NODE_RANK" ]] && error_exit "NODE_RANK is required"
[[ -z "$LOG_DIR" ]] && error_exit "LOG_DIR is required"

# 打印解析结果
cat <<EOF
Parsed parameters:
  MASTER_ADDR: $MASTER_ADDR
  MASTER_PORT: $MASTER_PORT
  WORLD_SIZE: $WORLD_SIZE
  NPROC_PER_NODE: $NPROC_PER_NODE
  NODE_RANK: $NODE_RANK
  LOG_DIR: $LOG_DIR
EOF

# 解析后续的执行命令及其参数
if [[ $# -eq 0 ]]; then
    error_exit "No execution command provided"
fi
EXEC_CMD="$1"
shift
EXEC_ARGS="$@"

# 打印即将执行的命令
cat <<EOF
Executing command:
  $EXEC_CMD $EXEC_ARGS
EOF

export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE="$WORLD_SIZE"

# 获取当前时间戳的函数
timestamp() {
    echo "$(date +"%Y-%m-%d %H:%M:%S")"
}

mkdir -p "${LOG_DIR}"

# 启动所有进程
PIDS=()
for (( LOCAL_RANK=0; LOCAL_RANK<$NPROC_PER_NODE; LOCAL_RANK++ )); do
    GLOBAL_RANK=$(( $NODE_RANK * $NPROC_PER_NODE + $LOCAL_RANK ))

    # 启动训练脚本
    echo "$(timestamp) 启动进程: RANK=$GLOBAL_RANK, LOCAL_RANK=$LOCAL_RANK"
    RANK="$GLOBAL_RANK" LOCAL_RANK="$LOCAL_RANK" $EXEC_CMD $EXEC_ARGS 2>&1 &
    PIDS+=($!)
done

declare -A processed=()  # 记录进程是否已处理
has_error=0

# 获取当前时间戳的函数
timestamp() {
    echo "$(date +"%Y-%m-%d %H:%M:%S")"
}

# 信号处理函数：终止所有子进程并退出
terminate_children() {
    echo "$(timestamp) 捕获终止信号，正在清理子进程..."
    has_error=1  # 标记异常退出
    # 向所有子进程发送终止信号
    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            echo "$(timestamp) 终止进程 $PID"
            kill "$PID" 2>/dev/null
        fi
    done
    # 等待所有子进程退出
    for PID in "${PIDS[@]}"; do
        wait "$PID" 2>/dev/null
        echo "$(timestamp) 进程 $PID 已退出"
    done
    echo "$(timestamp) 退出脚本，状态码：$has_error"
    exit "$has_error"
}

# 注册信号捕获（SIGTERM/SIGINT/SIGHUP）
trap terminate_children SIGTERM SIGINT SIGHUP

echo "$(timestamp) 开始监控子进程..."

while true; do
    all_processed=1  # 标记是否所有进程都已处理
    for PID in "${PIDS[@]}"; do
        if [[ -z "${processed[$PID]}" ]]; then
            all_processed=0  # 存在未处理的进程
            if ! kill -0 "$PID" 2>/dev/null; then
                # 进程已结束，获取退出状态
                wait "$PID"
                exit_status=$?
                processed["$PID"]=1
                echo "$(timestamp) 进程 $PID 结束，退出状态 $exit_status"
                if (( exit_status != 0 )); then
                    has_error=1
                    echo "$(timestamp) 发现异常退出的进程 $PID"
                fi
            fi
        fi
    done

    # 退出条件1: 所有进程处理完成
    if (( all_processed )); then
        echo "$(timestamp) 所有进程已处理完毕，退出脚本"
        break
    fi

    # 退出条件2: 发现异常后处理完毕
    if (( has_error )); then
        echo "$(timestamp) 发现异常，开始清理剩余进程..."
        for PID in "${PIDS[@]}"; do
            if [[ -z "${processed[$PID]}" ]]; then
                echo "$(timestamp) 终止进程 $PID"
                kill "$PID" 2>/dev/null
            fi
        done
        for PID in "${PIDS[@]}"; do
            if [[ -z "${processed[$PID]}" ]]; then
                wait "$PID" 2>/dev/null
                echo "$(timestamp) 进程 $PID 已退出"
            fi
        done
        echo "$(timestamp) 异常处理完成，退出脚本"
        break
    fi

    sleep 5
done

echo "$(timestamp) 脚本执行结束，状态码：$has_error"
exit "$has_error"