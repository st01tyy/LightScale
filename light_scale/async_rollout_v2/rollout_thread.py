"""Thread entrypoint for async rollout v2."""

import asyncio
import logging
from typing import Dict, Any

from light_scale.logger_utils import setup_logger_v2_sub_process

from light_scale.async_rollout_v2.config_loader import load_rollout_config, get_async_rollout_config
from light_scale.async_rollout_v2.rollout_loop import run_rollout_loop


class RolloutInitializationError(Exception):
	"""Raised when rollout initialization fails."""

	pass


def rollout_thread_main(
	rollout_cfg_path: str,
	passed_iters: int,
	rollout_batch_size: int,
	input_queue,
	output_queue,
	stop_event,
	logging_queue,
	start_event,
	failed_event,
	log_level: int,
):
	"""rollout 线程主函数，负责初始化上下文并启动 asyncio 主循环。"""
	logger = setup_logger_v2_sub_process(
		name="rollout_main",
		setup_distributed=False,
		level=log_level,
		log_queue=logging_queue,
	)
	logger.warning(f"log level for async rollout: {log_level}")

	try:
		config = load_rollout_config(rollout_cfg_path)
		async_cfg = get_async_rollout_config(config)
		_ = async_cfg.get("services")
		_ = async_cfg.get("workers")
		_ = async_cfg.get("data")
	except RolloutInitializationError as err:
		logger.error("Rollout 初始化失败: %s", err)
		stop_event.set()
		failed_event.set()
		return
	except Exception as err:
		logger.exception("Rollout 初始化出现未预期异常: %s", err)
		stop_event.set()
		failed_event.set()
		return

	logger.info("Rollout 配置解析完成，进入 asyncio 主循环初始化。")

	try:
		asyncio.run(
			run_rollout_loop(
				rollout_batch_size=rollout_batch_size,
				passed_iters=passed_iters,
				async_cfg=async_cfg,
				input_queue=input_queue,
				output_queue=output_queue,
				stop_event=stop_event,
				start_event=start_event,
				failed_event=failed_event,
				logger=logger,
				log_level=log_level,
			)
		)
	except Exception as err:
		logger.exception("Rollout 主循环异常: %s", err)
		stop_event.set()
		failed_event.set()
