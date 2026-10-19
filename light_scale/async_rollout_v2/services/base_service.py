"""AsyncBaseService for async rollout v2."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from light_scale.data import Resource
from light_scale.logger_utils import get_logging_queue, setup_logger_v2_sub_process


class AsyncBaseService(ABC):
	"""异步服务基类：负责生命周期管理与健康检查。"""

	def __init__(
		self,
		name: str,
		resources: Optional[List[Resource]] = None,
		startup_timeout: float = 60.0,
		log_level: int = logging.INFO,
	):
		self.name = name
		self.resources = resources or []
		self.startup_timeout = startup_timeout
		self._ready = False
		self.logger = setup_logger_v2_sub_process(
			name=self.name,
			level=log_level,
			log_queue=get_logging_queue(),
		)

	@property
	def is_ready(self) -> bool:
		return self._ready

	async def start(self) -> None:
		"""启动 service 并执行健康检查。"""
		await self._initialize()
		await self._health_check()
		self._ready = True

	async def stop(self) -> None:
		"""停止 service 并清理资源。"""
		await self._finalize()
		self._ready = False

	def submit(self, task):
		"""保持旧接口语义：返回 awaitable。"""
		return self.request(task)

	@abstractmethod
	async def request(self, task):
		"""子类实现实际请求逻辑。"""
		raise NotImplementedError

	async def _initialize(self) -> None:
		"""子类可选覆盖：初始化连接池等资源。"""
		return None

	async def _finalize(self) -> None:
		"""子类可选覆盖：关闭连接池等资源。"""
		return None

	@abstractmethod
	async def _health_check(self) -> None:
		"""子类实现健康检查，异常直接抛出。"""
		raise NotImplementedError
