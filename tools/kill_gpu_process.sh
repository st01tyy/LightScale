#!/bin/bash

# 获取所有GPU上的进程ID
process_ids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

# 检查是否有进程信息
if [ -z "$process_ids" ]; then
	  echo "没有找到GPU上的进程。"
	    exit 0
    fi

    # 将进程ID转换为数组
    process_ids_array=($process_ids)

    # 杀死所有找到的进程
    for pid in "${process_ids_array[@]}"; do
	      echo "正在杀死进程 ID: $pid"
	        kill -9 $pid
	done

	echo "所有GPU上的进程已成功杀死。"
