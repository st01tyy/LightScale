"""Async worker implementations for async rollout v2."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np

from light_scale.data import Message, MultiResponseSample
from light_scale.async_rollout_v2.reward_utils import compute_normed_rewards
from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.services.sglang_native_service import (
	AsyncSGLangNativeService,
	LogprobTuple,
	SGLangNativeGenerateTask,
	build_prefill_logprob_task,
)
from light_scale.async_rollout_v2.services.sglang_service import (
	AsyncSGLangService,
	SGLangResult,
	SGLangTask,
)
from light_scale.async_rollout_v2.utils.chat_template_utils import get_cached_template_artifacts
from light_scale.logger_utils import get_logging_queue, setup_logger_v2_sub_process
import traceback


class AsyncBaseWorker(ABC):
	"""异步 Worker 基类，负责定义单条数据 rollout 阶段处理逻辑。"""

	REQUIRED_SERVICE_NAMES: List[str] = []
	USES_LLM_JUDGE: bool = False

	def __init__(
		self,
		input_data: dict,
		service_dict: Dict[str, AsyncBaseService],
		stop_event,
		log_level: int,
		teacher_service_name: Optional[str] = None,
		**worker_cfg,
	):
		super().__init__()
		self.input_data = input_data
		self.service_dict = service_dict
		self.teacher_service_name = teacher_service_name
		if stop_event is None:
			raise ValueError("AsyncBaseWorker 初始化需要 stop_event")
		self.stop_event = stop_event
		self.logger = setup_logger_v2_sub_process(
			name=type(self).__name__,
			level=log_level,
			log_queue=get_logging_queue(),
		)

	@abstractmethod
	async def run(self) -> MultiResponseSample:
		raise NotImplementedError


@dataclass
class AsyncSingleTurnWorkerConfig:
	"""单轮 worker 通用配置，统一描述推理与奖励阶段所需参数。"""

	n_samples: int
	max_tokens: int
	advantage_estimator: str
	temperature: float = 1.0
	top_p: float = 1.0
	top_k: int = -1
	presence_penalty: float = 0.0
	tokenizer_path: Optional[str] = None
	stop_token: Optional[str] = None
	add_stop_token: bool = False
	request_timeout: Optional[float] = None
	force_thinking: bool = False
	begin_of_thinking: Optional[str] = None
	end_of_thinking: Optional[str] = None
	apply_token_budget: bool = True


class AsyncSingleTurnWorker(AsyncBaseWorker, ABC):
	"""封装单轮 SGLang 推理-打分-归一化流程的异步基类。"""

	CONFIG_CLS = AsyncSingleTurnWorkerConfig
	COMPLETION_SERVICE_NAME = "actor_model"
	REQUIRED_SERVICE_NAMES = AsyncBaseWorker.REQUIRED_SERVICE_NAMES + [COMPLETION_SERVICE_NAME]

	def __init__(
		self,
		input_data: dict,
		service_dict: Dict[str, AsyncBaseService],
		stop_event,
		log_level: int,
		teacher_service_name: Optional[str] = None,
		**worker_cfg,
	):
		super().__init__(
			input_data=input_data,
			service_dict=service_dict,
			stop_event=stop_event,
			log_level=log_level,
			teacher_service_name=teacher_service_name,
		)
		if "advantage_estimator" not in worker_cfg:
			raise ValueError("AsyncSingleTurnWorker 需要提供 advantage_estimator 配置")
		self._config = self.CONFIG_CLS(**worker_cfg)
		self._sglang_service = self._require_completion_service()
		self._teacher_service: Optional[AsyncSGLangNativeService] = None
		self._tokenizer: Optional[Any] = None
		if self.teacher_service_name is not None:
			self._teacher_service = self._require_teacher_service()
			if not self._config.tokenizer_path:
				raise ValueError("teacher_service_name 已配置时，tokenizer_path 不能为空")
			self._tokenizer, _ = get_cached_template_artifacts(
				tokenizer_path=self._config.tokenizer_path,
				trust_remote_code=True,
				chat_template_path=None,
				owner_name=type(self).__name__,
			)

	async def run(self) -> MultiResponseSample:
		"""执行单条样本的完整流程：构建 prompt、调用服务、打分以及奖励归一化。"""
		logger = self.logger
		raw_sample = self.input_data
		sample = MultiResponseSample(
			prompt=raw_sample["prompt"],
			dataset_type=raw_sample["dataset_type"],
			ground_truth=raw_sample["ground_truth"],
			problem=raw_sample.get("problem", None),
			sample_id=raw_sample.get("sample_id", None)
		)
		if self.stop_event.is_set():
			self._clear_sample(sample)
			logger.debug("stop_event 已置位，直接清空样本输出")
			return sample
		self._maybe_apply_force_thinking_prompt(sample)
		task = self._build_task_from_sample(sample, raw_sample)
		result = await self._request_sglang(task)
		if result is None:
			self._clear_sample(sample)
			logger.warning("result返回为空，提前终止worker")
			return sample
		self._handle_completion_result(sample, result)
		if self._teacher_service is None:
			await self._score_responses(sample, raw_sample)
		else:
			task_results = await asyncio.gather(
				self._score_responses(sample, raw_sample),
				self._prepare_teacher_distillation_targets(sample),
				return_exceptions=True,
			)
			score_error = task_results[0] if isinstance(task_results[0], Exception) else None
			teacher_error = task_results[1] if isinstance(task_results[1], Exception) else None
			if teacher_error is not None:
				self.logger.error("teacher distillation 目标构造失败：%s", teacher_error)
				self.logger.error(
					"".join(
						traceback.format_exception(
							type(teacher_error),
							teacher_error,
							teacher_error.__traceback__,
						)
					)
				)
				self.stop_event.set()
				self._clear_sample(sample)
				return sample
			if score_error is not None:
				raise score_error
		self._postprocess_sample(sample)
		logger.debug("AsyncSingleTurnWorker 处理完成: dataset_type=%s", sample.dataset_type)
		return sample

	def _clear_sample(self, sample: MultiResponseSample) -> None:
		sample.responses = []
		sample.group_messages = []
		sample.rewards = []
		sample.reward_metrics_list = []
		sample.normed_rewards = []
		sample.completion_tokens = 0
		sample.total_tokens = 0
		sample.group_content_ids = []
		sample.group_loss_mask = []
		sample.group_teacher_log_probs = []

	def _maybe_apply_force_thinking_prompt(self, sample: MultiResponseSample) -> None:
		if not self._config.force_thinking:
			return
		prefix = self._config.begin_of_thinking
		prompt = sample.prompt
		if not prefix:
			raise ValueError("force_thinking 已启用，但未提供 begin_of_thinking")
		if prompt.startswith(prefix):
			return
		sample.prompt = prompt + prefix

	def _require_completion_service(self) -> AsyncSGLangService:
		service_name = self.COMPLETION_SERVICE_NAME
		service = self.service_dict.get(service_name)
		if service is None:
			raise RuntimeError(
				f"AsyncSingleTurnWorker 初始化失败：缺少 {service_name} 服务实例"
			)
		if not isinstance(service, AsyncSGLangService):
			raise TypeError(
				f"AsyncSingleTurnWorker 需要 AsyncSGLangService 类型的服务，当前为 {type(service).__name__}"
			)
		return service

	def _require_teacher_service(self) -> AsyncSGLangNativeService:
		service_name = self.teacher_service_name
		service = self.service_dict.get(service_name)
		if service is None:
			raise RuntimeError(
				f"AsyncSingleTurnWorker 初始化失败：缺少 teacher service {service_name}"
			)
		if not isinstance(service, AsyncSGLangNativeService):
			raise TypeError(
				f"AsyncSingleTurnWorker 需要 AsyncSGLangNativeService 类型的 teacher service，当前为 {type(service).__name__}"
			)
		return service

	def _build_task_from_sample(self, sample: MultiResponseSample, raw_sample: dict) -> SGLangTask:
		cfg = self._config
		if not sample.prompt:
			raise ValueError("AsyncSingleTurnWorker 收到的样本缺少 prompt")
		max_tokens = raw_sample["token_budget"] if "token_budget" in raw_sample and cfg.apply_token_budget else cfg.max_tokens
		max_tokens = min(max_tokens, cfg.max_tokens)
		self.logger.debug(f"actual max tokens: {max_tokens}")
		return SGLangTask(
			prompt=sample.prompt,
			n_samples=cfg.n_samples,
			max_tokens=max_tokens,
			temperature=cfg.temperature,
			top_p=cfg.top_p,
			top_k=cfg.top_k,
			presence_penalty=cfg.presence_penalty,
			stop=cfg.stop_token,
			add_stop=cfg.add_stop_token,
			timeout=cfg.request_timeout,
		)

	async def _request_sglang(self, task: SGLangTask) -> Optional[SGLangResult]:
		try:
			return await self._sglang_service.submit(task)
		except Exception as err:
			self.logger.error("%s 服务异常：%s", self._sglang_service.name, err)
			self.logger.error(traceback.format_exc())
			self.stop_event.set()
			return None

	def _build_messages(self, prompt: str, response: str) -> List[Message]:
		return [
			Message(content=prompt, is_masked=True),
			Message(content=response, is_masked=False),
		]

	def _sync_group_messages_from_responses(self, sample: MultiResponseSample) -> None:
		responses = sample.responses or []
		sample.group_messages = [
			self._build_messages(sample.prompt, response)
			for response in responses
		]

	def _handle_completion_result(self, sample: MultiResponseSample, result: Optional[SGLangResult]):
		if result is None:
			sample.responses = []
			sample.group_messages = []
			sample.completion_tokens = 0
			sample.total_tokens = 0
			sample.group_content_ids = []
			sample.group_loss_mask = []
			sample.group_teacher_log_probs = []
			return
		responses = [text for text, _ in result.responses]
		expected = self._config.n_samples
		if len(responses) != expected:
			raise RuntimeError(
				"SGLang 返回的响应数量与配置不一致："
				f"expected={expected}, actual={len(responses)}"
			)
		sample.responses = responses
		self._sync_group_messages_from_responses(sample)
		sample.completion_tokens = result.completion_tokens
		sample.total_tokens = result.total_tokens
		sample.group_content_ids = []
		sample.group_loss_mask = []
		sample.group_teacher_log_probs = []

	async def _prepare_teacher_distillation_targets(self, sample: MultiResponseSample) -> None:
		message_groups = sample.group_messages or []
		if not message_groups:
			sample.group_content_ids = []
			sample.group_loss_mask = []
			sample.group_teacher_log_probs = []
			return

		results = await asyncio.gather(
			*(self._build_teacher_branch_targets(messages) for messages in message_groups)
		)
		sample.group_content_ids = [result[0] for result in results]
		sample.group_loss_mask = [result[1] for result in results]
		sample.group_teacher_log_probs = [result[2] for result in results]

	async def _build_teacher_branch_targets(
		self,
		messages: List[Message],
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		content_ids, loss_mask = self._build_content_ids_and_loss_mask(messages)
		raw_logprobs = await self._fetch_teacher_log_probs_for_messages(messages, content_ids)
		teacher_log_probs = self._normalize_teacher_log_probs(content_ids, raw_logprobs)
		return content_ids, loss_mask, teacher_log_probs

	def _build_content_ids_and_loss_mask(
		self,
		messages: List[Message],
	) -> Tuple[np.ndarray, np.ndarray]:
		if self._tokenizer is None:
			raise RuntimeError("tokenizer 未初始化")

		content_ids: List[int] = []
		loss_mask: List[int] = []
		for message in messages:
			token_ids = self._tokenizer.encode(
				message.content,
				add_special_tokens=False,
			)
			content_ids.extend(token_ids)
			loss_mask.extend([0 if message.is_masked else 1] * len(token_ids))
		return (
			np.asarray(content_ids, dtype=np.int32),
			np.asarray(loss_mask, dtype=np.uint8),
		)

	async def _fetch_teacher_log_probs_for_messages(
		self,
		messages: List[Message],
		content_ids: np.ndarray,
	) -> List[LogprobTuple]:
		del messages
		if self._teacher_service is None:
			raise RuntimeError("teacher service 未初始化")
		task = self._build_teacher_prefill_task(content_ids)
		result = await self._teacher_service.request(task)
		return result.meta_info.input_token_logprobs

	def _build_teacher_prefill_task(self, content_ids: np.ndarray) -> SGLangNativeGenerateTask:
		return build_prefill_logprob_task(
			input_ids=content_ids.tolist(),
			top_logprobs_num=0,
			logprob_start_len=0,
			return_text_in_logprobs=False,
			timeout=self._config.request_timeout or 300.0,
		)

	def _normalize_teacher_log_probs(
		self,
		content_ids: np.ndarray,
		raw_logprobs: List[LogprobTuple],
	) -> np.ndarray:
		normalized = np.zeros((len(content_ids),), dtype=np.float32)
		for idx, entry in enumerate(raw_logprobs[: len(content_ids)]):
			logprob = entry[0]
			normalized[idx] = 0.0 if logprob is None else float(logprob)
		return normalized

	@abstractmethod
	async def _score_responses(self, sample: MultiResponseSample, raw_sample: dict) -> None:
		return None

	def _postprocess_sample(self, sample: MultiResponseSample) -> None:
		responses = sample.responses or []
		if not responses:
			sample.normed_rewards = []
			return
		if sample.rewards is None:
			sample.rewards = [None] * len(responses)
		valid_flags = [
			resp is not None and reward is not None
			for resp, reward in zip(responses, sample.rewards)
		]
		sample.normed_rewards = compute_normed_rewards(
			rewards=sample.rewards,
			valid_flags=valid_flags,
			advantage_estimator=self._config.advantage_estimator,
		)
