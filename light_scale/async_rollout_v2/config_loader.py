"""Config loader for async rollout v2."""

from pathlib import Path
from typing import Any, Dict

import yaml


class RolloutInitializationError(Exception):
	"""Raised when rollout initialization fails."""

	pass


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
	services_cfg = async_cfg.get("services") or []
	if not isinstance(services_cfg, list):
		raise RolloutInitializationError("async_rollout.services 必须是列表")
	workers_cfg = async_cfg.get("workers") or []
	if not isinstance(workers_cfg, list):
		raise RolloutInitializationError("async_rollout.workers 必须是列表")
	teacher_models_registry = async_cfg.get("teacher_models_registry")
	if teacher_models_registry is not None and not isinstance(teacher_models_registry, list):
		raise RolloutInitializationError("async_rollout.teacher_models_registry 必须为列表或 null")
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
		for registry_entry in teacher_models_registry:
			if not isinstance(registry_entry, dict):
				raise RolloutInitializationError("teacher_models_registry 的每个条目必须是字典")
			if registry_entry.get("service_name") is None:
				raise RolloutInitializationError("teacher_models_registry 的每个条目必须包含 service_name")
			data_types = registry_entry.get("data_type")
			if not isinstance(data_types, list):
				raise RolloutInitializationError("teacher_models_registry.data_type 必须是列表")
	return async_cfg
