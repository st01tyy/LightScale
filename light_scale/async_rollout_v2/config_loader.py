"""Config loader for async rollout v2."""

from pathlib import Path
from typing import Any, Dict, List

import yaml


class RolloutInitializationError(Exception):
	"""Raised when rollout initialization fails."""

	pass


def _validate_data_config(data_cfg: Any) -> None:
	if isinstance(data_cfg, str):
		if not data_cfg:
			raise RolloutInitializationError("async_rollout.data 不能为空字符串")
		return
	if not isinstance(data_cfg, list) or not data_cfg:
		raise RolloutInitializationError("async_rollout.data 必须是非空字符串或非空列表")
	seen_paths = set()
	for index, item in enumerate(data_cfg):
		if not isinstance(item, dict):
			raise RolloutInitializationError("async_rollout.data 列表中的每个元素都必须是字典")
		n_value = item.get("n")
		data_path = item.get("data_path")
		if set(item.keys()) != {"n", "data_path"}:
			raise RolloutInitializationError(
				"async_rollout.data 列表元素只能包含 n 和 data_path 两个字段"
			)
		if not isinstance(n_value, int) or n_value <= 0:
			raise RolloutInitializationError(
				f"async_rollout.data[{index}].n 必须是正整数"
			)
		if not isinstance(data_path, str) or not data_path:
			raise RolloutInitializationError(
				f"async_rollout.data[{index}].data_path 必须是非空字符串"
			)
		if data_path in seen_paths:
			raise RolloutInitializationError(
				f"async_rollout.data 中存在重复 data_path: {data_path}"
			)
		seen_paths.add(data_path)


def _validate_validation_data_paths(validation_data_paths: Any) -> None:
	if validation_data_paths is None:
		return
	if not isinstance(validation_data_paths, list):
		raise RolloutInitializationError("async_rollout.validation_data_paths 必须为列表或 null")
	seen_paths = set()
	for index, path_value in enumerate(validation_data_paths):
		if not isinstance(path_value, str) or not path_value:
			raise RolloutInitializationError(
				f"async_rollout.validation_data_paths[{index}] 必须是非空字符串"
			)
		if path_value in seen_paths:
			raise RolloutInitializationError(
				f"async_rollout.validation_data_paths 中存在重复路径: {path_value}"
			)
		seen_paths.add(path_value)


def load_rollout_config(rollout_cfg_path: str) -> Dict[str, Any]:
	if not rollout_cfg_path:
		raise RolloutInitializationError("未提供 rollout 配置路径")
	config_path = Path(rollout_cfg_path).expanduser().resolve()
	if not config_path.exists():
		raise RolloutInitializationError(f"找不到配置文件: {config_path}")
	with config_path.open("r", encoding="utf-8") as cfg_file:
		try:
			return yaml.safe_load(cfg_file) or {}
		except yaml.YAMLError as err:
			raise RolloutInitializationError(f"解析配置文件失败: {err}") from err


def get_async_rollout_config(config: Dict[str, Any]) -> Dict[str, Any]:
	async_cfg = config
	if not isinstance(async_cfg, dict):
		raise RolloutInitializationError("配置文件缺少 async_rollout 节点或类型不正确")
	if async_cfg.get("data") is None:
		raise RolloutInitializationError("async_rollout.data 配置缺失")
	_validate_data_config(async_cfg.get("data"))
	services_cfg = async_cfg.get("services") or []
	if not isinstance(services_cfg, list):
		raise RolloutInitializationError("async_rollout.services 必须是列表")
	workers_cfg = async_cfg.get("workers") or []
	if not isinstance(workers_cfg, list):
		raise RolloutInitializationError("async_rollout.workers 必须是列表")
	validation_data_paths = async_cfg.get("validation_data_paths")
	_validate_validation_data_paths(validation_data_paths)
	teacher_models_registry = async_cfg.get("teacher_models_registry")
	if teacher_models_registry is not None and not isinstance(teacher_models_registry, list):
		raise RolloutInitializationError("async_rollout.teacher_models_registry 必须为列表或 null")
	student_service_name = async_cfg.get("student_service_name")
	if student_service_name is not None and (not isinstance(student_service_name, str) or not student_service_name):
		raise RolloutInitializationError("async_rollout.student_service_name 必须为非空字符串或 null")
	for service_cfg in services_cfg:
		if service_cfg.get("name") is None or service_cfg.get("type") is None:
			raise RolloutInitializationError("每个 service 必须包含 type 与 name")
		resources_cfg = service_cfg.get("resources", None)
		if not isinstance(resources_cfg, list) or not resources_cfg:
			raise RolloutInitializationError(
				f"service {service_cfg.get('name')} 未提供 resources 或格式不正确"
			)
	for worker_cfg in workers_cfg:
		if worker_cfg.get("type") is None:
			raise RolloutInitializationError("每个 worker 必须包含 type 字段")
	if teacher_models_registry is not None:
		teacher_data_types = set()
		for registry_entry in teacher_models_registry:
			if not isinstance(registry_entry, dict):
				raise RolloutInitializationError("teacher_models_registry 的每个条目必须是字典")
			if registry_entry.get("service_name") is None:
				raise RolloutInitializationError("teacher_models_registry 的每个条目必须包含 service_name")
			data_types = registry_entry.get("data_type")
			if not isinstance(data_types, list):
				raise RolloutInitializationError("teacher_models_registry.data_type 必须是列表")
			for data_type in data_types:
				if isinstance(data_type, str) and data_type:
					teacher_data_types.add(data_type)

		shared_llm_judge_cfg = async_cfg.get("llm_judge") or {}
		for worker_cfg in workers_cfg:
			if worker_cfg.get("type") != "llm_judge":
				continue
			handle_data_types = worker_cfg.get("handle_data_types")
			if not isinstance(handle_data_types, list) or not handle_data_types:
				continue
			if not any(data_type in teacher_data_types for data_type in handle_data_types):
				continue
			worker_params = worker_cfg.get("params") or {}
			use_ref_answers = worker_params.get(
				"use_ref_answers",
				shared_llm_judge_cfg.get("use_ref_answers", False),
			)
			if use_ref_answers:
				raise RolloutInitializationError(
					"teacher mode 当前要求 rollout response 保持原样，不能与 llm_judge.use_ref_answers 同时开启"
				)
	return async_cfg
