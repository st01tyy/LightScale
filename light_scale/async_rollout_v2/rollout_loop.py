"""Asyncio main loop for rollout v2."""

import asyncio
import logging
from typing import Any, Dict, List, Tuple, Type, Optional

import queue
import datasets
from datasets import load_from_disk

from light_scale.data import MultiResponseSample, Resource
from light_scale.async_rollout_v2.executors import get_process_pool, close_all, get_thread_pool
from light_scale.async_rollout_v2.registries import SERVICE_CLASS_REGISTRY, WORKER_CLASS_REGISTRY
from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.workers.base_worker import AsyncBaseWorker


class RolloutInitializationError(Exception):
	"""Raised when rollout initialization fails."""

	pass


async def run_rollout_loop(
	rollout_batch_size: int,
	passed_iters: int,
	async_cfg: Dict[str, Any],
	input_queue,
	output_queue,
	stop_event,
	start_event,
	failed_event,
	logger: logging.Logger,
	log_level: int,
) -> None:
	"""asyncio 主循环：按批次并发执行 worker，结果写回输出队列。"""
	try:
		(
			name_to_services,
			type_to_worker_params,
			data_type_to_worker_type,
			samples,
		) = await _initialize_rollout_context(async_cfg, stop_event, logger, log_level)
		services_to_stop = list(name_to_services.values())
		start_event.set()
		logger.info("Rollout v2 初始化完成，启动主循环。")
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

	get_process_pool(32)
	logger.info("进程池已初始化")

	get_thread_pool(32)
	logger.info("线程池已初始化")

	try:
		await _run_rollout_loop(
			rollout_batch_size=rollout_batch_size,
			passed_iters=passed_iters,
			samples=samples,
			name_to_services=name_to_services,
			type_to_worker_params=type_to_worker_params,
			data_type_to_worker_type=data_type_to_worker_type,
			input_queue=input_queue,
			output_queue=output_queue,
			stop_event=stop_event,
			logger=logger,
			log_level=log_level,
		)
	except asyncio.CancelledError:
		logger.warning(
			"Rollout v2 asyncio loop received cancellation",
		)
		raise
	except Exception as err:
		logger.exception("Rollout v2 asyncio loop crashed unexpectedly")
		raise
	finally:
		await _stop_services(services_to_stop, logger)
		close_all()
		if stop_event.is_set():
			logger.warning("Rollout v2 asyncio loop stopping")
		else:
			logger.warning(
				"Rollout v2 asyncio loop exited without stop signal, likely via cancellation/exception unwind"
			)


async def _initialize_rollout_context(
	async_cfg: Dict[str, Any],
	stop_event,
	logger: logging.Logger,
	log_level: int,
) -> Tuple[
	Dict[str, AsyncBaseService],
	Dict[Type[AsyncBaseWorker], Dict[str, Any]],
	Dict[str, Type[AsyncBaseWorker]],
	datasets.Dataset,
]:
	services_cfg = async_cfg.get("services") or []
	workers_cfg = async_cfg.get("workers") or []
	for worker_cfg in workers_cfg:
		worker_type = worker_cfg.get("type", None)
		if worker_type is None:
			raise RolloutInitializationError("每个 worker 必须包含 type 字段")
		if worker_type not in WORKER_CLASS_REGISTRY:
			raise RolloutInitializationError(f"未知的 worker 类型: {worker_type}")
	for service_cfg in services_cfg:
		service_type = service_cfg.get("type", None)
		if service_type is None:
			raise RolloutInitializationError("每个 service 必须包含 type 字段")
		if service_type not in SERVICE_CLASS_REGISTRY:
			raise RolloutInitializationError(f"未知的 service 类型: {service_type}")

	name_to_services = await _initialize_services(services_cfg, stop_event, logger, log_level)
	(
		type_to_worker_params,
		data_type_to_worker_type,
		) = _prepare_worker_params(workers_cfg, logger, async_cfg.get("llm_judge") or {})
	
	samples = await _load_dataset(async_cfg.get("data"), logger)
	samples = samples.shuffle(seed=42)
	_ensure_worker_service_dependencies(type_to_worker_params, name_to_services)
	_ensure_data_worker_dependencies(data_type_to_worker_type, samples)
	return name_to_services, type_to_worker_params, data_type_to_worker_type, samples


async def _run_rollout_loop(
	rollout_batch_size: int,
	passed_iters: int,
	samples: datasets.Dataset,
	name_to_services: Dict[str, AsyncBaseService],
	type_to_worker_params: Dict[Type[AsyncBaseWorker], Dict[str, Any]],
	data_type_to_worker_type: Dict[str, Type[AsyncBaseWorker]],
	input_queue,
	output_queue,
	stop_event,
	logger: logging.Logger,
	log_level: int,
) -> None:
	thread_pool_size = rollout_batch_size
	total_samples = len(samples)
	if total_samples == 0:
		raise RolloutInitializationError("数据集为空，无法执行 rollout")
	cursor = (rollout_batch_size * passed_iters) % total_samples
	logger.info(
		"Rollout v2 主循环已就绪: total_samples=%s, batch_size=%s, cursor=%s",
		total_samples,
		rollout_batch_size,
		cursor,
	)

	while not stop_event.is_set():
		passed_iters_value = await _queue_get(input_queue)
		if passed_iters_value is None:
			await asyncio.sleep(0.1)
			continue

		batch_ids: List[int] = []
		while len(batch_ids) < rollout_batch_size:
			batch_ids.append(cursor)
			cursor = (cursor + 1) % total_samples
		batch_samples = samples.select(batch_ids)

		logger.info("Rollout step %s started", passed_iters_value)
		await _execute_batch(
			batch_samples=batch_samples,
			data_type_to_worker_type=data_type_to_worker_type,
			type_to_worker_params=type_to_worker_params,
			name_to_services=name_to_services,
			stop_event=stop_event,
			output_queue=output_queue,
			logger=logger,
			log_level=log_level,
		)


async def _execute_batch(
	batch_samples: datasets.Dataset,
	data_type_to_worker_type: Dict[str, Type[AsyncBaseWorker]],
	type_to_worker_params: Dict[Type[AsyncBaseWorker], Dict[str, Any]],
	name_to_services: Dict[str, AsyncBaseService],
	stop_event,
	output_queue,
	logger: logging.Logger,
	log_level: int,
) -> None:
	if stop_event.is_set():
		return

	tasks = []
	for sample in batch_samples:
		if stop_event.is_set():
			break
		worker_cls = data_type_to_worker_type[sample["dataset_type"]]
		worker_params = type_to_worker_params.get(worker_cls, dict())
		try:
			worker = worker_cls(
				input_data=sample,
				service_dict=name_to_services,
				stop_event=stop_event,
				log_level=log_level,
				**worker_params,
			)
			logger.debug(
				"样本(dataset_type=%s) 分配 worker=%s",
				sample["dataset_type"],
				worker_cls,
			)
		except Exception as e:
			logger.error(
				"worker %s 初始化失败 (dataset_type=%s): %s",
				worker_cls,
				sample["dataset_type"],
				str(e)
			)
			stop_event.set()
			break
		try:
			tasks.append(asyncio.create_task(worker.run()))
		except Exception:
			logger.error(
				"worker %s 提交执行失败 (dataset_type=%s)",
				worker_cls,
				sample["dataset_type"],
			)
			stop_event.set()
			break

	if not tasks:
		return
	completed_tasks = 0
	total_tasks = len(tasks)

	for task in asyncio.as_completed(tasks):
		if stop_event.is_set():
			logger.warning("检测到 stop_event 已设置，停止等待剩余任务完成")
			break
		try:
			result_sample: MultiResponseSample = await task
			logger.debug("样本完成，写入输出队列")
			await _queue_put(output_queue, result_sample)
			completed_tasks += 1
			logger.info("样本已入队: %d/%d", completed_tasks, total_tasks)
		except Exception as e:
			logger.error(f"worker 处理样本失败: {str(e)}")
			stop_event.set()
			break

	if stop_event.is_set():
		logger.warning("检测到 stop_event 已设置，取消所有任务")
		for task in tasks:
			task.cancel()


async def _initialize_services(
	service_cfgs: List[Dict[str, Any]],
	stop_event,
	logger: logging.Logger,
	log_level: int,
) -> Dict[str, AsyncBaseService]:
	name_to_services: Dict[str, AsyncBaseService] = {}
	for service_cfg in service_cfgs:
		service_name = service_cfg.get("name", None)
		service_type = service_cfg.get("type", None)
		if not service_name or not service_type:
			raise RolloutInitializationError("每个 service 必须包含 type 与 name")
		if service_name in name_to_services:
			raise RolloutInitializationError(f"检测到重复的 service name: {service_name}")

		resources_cfg = service_cfg.get("resources", None)
		if not isinstance(resources_cfg, list) or not resources_cfg:
			raise RolloutInitializationError(
				f"service {service_name} 未提供 resources 或格式不正确"
			)
		resources = _build_resources(resources_cfg)
		params = service_cfg.get("params") or {}
		service_cls = SERVICE_CLASS_REGISTRY.get(service_type, None)
		if service_cls is None:
			raise RolloutInitializationError(
				f"service {service_name} 指定了未知类型 {service_type}"
			)
		service = service_cls(
			name=service_name,
			resources=resources,
			log_level=log_level,
			**params,
		)
		name_to_services[service_name] = service

	try:
		for service in name_to_services.values():
			await service.start()
			logger.info("service %s 启动成功", service.name)
		return name_to_services
	except Exception:
		if stop_event is not None:
			stop_event.set()
		await _stop_services(list(name_to_services.values()), logger)
		raise


def _build_resources(resources_cfg: List[Dict[str, Any]]) -> List[Resource]:
	resources: List[Resource] = []
	for resource_params in resources_cfg:
		resource = Resource(**resource_params)
		resources.append(resource)
	if len(resources) == 0:
		raise RolloutInitializationError("service 的资源列表为空")
	return resources


def _prepare_worker_params(
	worker_cfgs: List[Dict[str, Any]],
	logger: logging.Logger,
	shared_llm_judge_cfg: Dict[str, Any],
) -> Tuple[Dict[type, Dict[str, Any]], Dict[str, type]]:
	type_to_worker_params: Dict[type, Dict[str, Any]] = dict()
	standalone_params_dict: Dict[type, Dict[str, Any]] = dict()
	data_type_to_worker_type: Dict[str, type] = dict()
	parent_worker_cfgs = [worker_cfg for worker_cfg in worker_cfgs if worker_cfg.get("handle_data_types", None) is None]
	actual_worker_cfgs = [worker_cfg for worker_cfg in worker_cfgs if worker_cfg.get("handle_data_types", None) is not None]
	for worker_cfg in parent_worker_cfgs:
		worker_type = worker_cfg["type"]
		cfg_params = worker_cfg.get("params", None)
		if cfg_params is None:
			raise RolloutInitializationError(f"parent worker {worker_type} 未提供 params")
		standalone_params_dict[WORKER_CLASS_REGISTRY[worker_type]] = cfg_params

	for worker_cfg in actual_worker_cfgs:
		worker_type = worker_cfg["type"]
		if not worker_type:
			raise RolloutInitializationError("worker 配置缺少 type")
		cfg_params = worker_cfg.get("params", dict())
		cfg_params = dict() if cfg_params is None else cfg_params
		data_types = worker_cfg.get("handle_data_types", None)
		if data_types is None:
			logger.warning("worker %s 未指定 handle_data_types", worker_type)
			continue
		if not isinstance(data_types, list):
			raise RolloutInitializationError(
				f"worker {worker_type} 的 handle_data_types 应为列表"
			)
		# assert cfg_params is not None, f"{worker_type} worker params is None"
		standalone_params_dict[WORKER_CLASS_REGISTRY[worker_type]] = cfg_params
		for data_type in data_types:
			if data_type in data_type_to_worker_type:
				existing_worker = data_type_to_worker_type[data_type]
				raise RolloutInitializationError(
					f"数据类型 {data_type} 同时被 worker {existing_worker} 和 {worker_type} 处理"
				)
			data_type_to_worker_type[data_type] = WORKER_CLASS_REGISTRY[worker_type]

	def _collect_ancestor_parent_params_and_merge(worker_cls: type) -> Dict[str, Any]:
		merged_parents_params: Dict[str, Any] = {}
		ancestors = [c for c in worker_cls.__mro__[1:] if c is not object]
		ancestors = list(reversed(ancestors))
		for parent_cls in ancestors:
			parent_param = standalone_params_dict.get(parent_cls, dict())
			merged_parents_params.update(parent_param)
		return merged_parents_params

	for worker_cls, standalone_params in standalone_params_dict.items():
		merged_parents_params = _collect_ancestor_parent_params_and_merge(worker_cls)
		worker_params = merged_parents_params
		if getattr(worker_cls, "USES_LLM_JUDGE", False):
			worker_params.update(shared_llm_judge_cfg)
		worker_params.update(standalone_params)
		type_to_worker_params[worker_cls] = worker_params

	return type_to_worker_params, data_type_to_worker_type


def _ensure_worker_service_dependencies(
	type_to_worker_params: Dict[Type[AsyncBaseWorker], Dict[str, Any]],
	name_to_services: Dict[str, AsyncBaseService],
) -> None:
	for worker_cls in type_to_worker_params.keys():
		required_service_names = worker_cls.REQUIRED_SERVICE_NAMES
		for service_name in required_service_names:
			service_instance = name_to_services.get(service_name, None)
			if service_instance is None:
				raise RolloutInitializationError(
					f"worker {worker_cls.__name__} 依赖的 service {service_name} 未成功启动"
				)


def _ensure_data_worker_dependencies(
	data_type_to_worker_type: Dict[str, Type[AsyncBaseWorker]],
	samples: datasets.Dataset,
):
	data_types = samples.unique("dataset_type")
	for data_type in data_types:
		worker_type = data_type_to_worker_type.get(data_type, None)
		if worker_type is None:
			raise RolloutInitializationError(
				f"样本数据类型 {data_type} 未找到可处理的 worker"
			)


async def _load_dataset(data_path_value: Optional[str], logger: logging.Logger) -> datasets.Dataset:
	logger.info("开始加载数据集，路径=%s", data_path_value)
	return await asyncio.to_thread(load_from_disk, data_path_value)


async def _stop_services(services: List[AsyncBaseService], logger: logging.Logger) -> None:
	if not services:
		return
	logger.warning("开始停止所有 service，数量=%s", len(services))
	for service in services:
		try:
			await service.stop()
			logger.info("service %s 停止成功", service.name)
		except Exception:
			logger.error("service %s 停止失败", service.name)


async def _queue_get(input_queue):
	def _get():
		try:
			return input_queue.get(timeout=0.5)
		except queue.Empty:
			return None
		except (EOFError, OSError):
			return None
	return await asyncio.to_thread(_get)


async def _queue_put(output_queue, item) -> None:
	await asyncio.to_thread(output_queue.put, item)
