"""Rock worker for async rollout v2."""

import asyncio
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from light_scale.async_rollout_v2.reward_utils import compute_normed_rewards
from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.services.rock_service import (
	AsyncRockService,
	RockResultTask,
	RockSubmitTask,
)
from light_scale.async_rollout_v2.utils.chat_template_utils import (
	convert_openai_trace_to_messages,
	count_tokens,
	extract_compat_response,
	get_cached_template_artifacts,
	parse_rendered_messages,
	render_chat,
	normalize_openai_messages,
	normalize_tools,
)
from light_scale.async_rollout_v2.workers.base_worker import AsyncBaseWorker
from light_scale.data import Message, MultiResponseSample


@dataclass
class RockWorkerConfig:
	n_samples: int
	advantage_estimator: str
	tokenizer_path: str
	chat_template_path: Optional[str] = None
	poll_interval_seconds: float = 2.0
	poll_timeout_seconds: float = 300.0
	poll_max_attempts: Optional[int] = None
	tokenizer_trust_remote_code: bool = True
	add_generation_prompt: bool = False
	sample_id_key: str = "task_id"
	sample_config_key: str = "sample_config"
	prompt_key: str = "prompt"
	problem_key: Optional[str] = "problem"
	ground_truth_key: str = "ground_truth"
	rock_service_name: str = "rock_service"
	target_service_name: str = "actor_model"
	submit_request_timeout: Optional[float] = None
	result_request_timeout: Optional[float] = None
	submit_retry: Optional[int] = None
	result_retry: Optional[int] = None


@dataclass
class RockBranchResult:
	response: str
	messages: List[Message]
	reward: float
	reward_metrics: Dict[str, Any]
	completion_tokens: int
	total_tokens: int
	valid: bool


class RockWorker(AsyncBaseWorker):
	"""Worker that submits Rock tasks and polls for OpenAI-style traces."""

	def __init__(
		self,
		input_data: dict,
		service_dict: Dict[str, AsyncBaseService],
		stop_event,
		log_level: int,
		**worker_cfg,
	):
		super().__init__(
			input_data=input_data,
			service_dict=service_dict,
			stop_event=stop_event,
			log_level=log_level,
		)
		self._config = RockWorkerConfig(**worker_cfg)
		self._rock_service = self._require_rock_service(self._config.rock_service_name)
		self._target_urls = self._get_target_urls(self._config.target_service_name)
		self._tokenizer, self._chat_template = self._get_cached_template_artifacts(
			tokenizer_path=self._config.tokenizer_path,
			trust_remote_code=self._config.tokenizer_trust_remote_code,
			chat_template_path=self._config.chat_template_path,
		)

	async def run(self) -> MultiResponseSample:
		raw_sample = self.input_data
		sample_id = str(raw_sample[self._config.sample_id_key])
		ground_truth = raw_sample[self._config.ground_truth_key]
		prompt = self._extract_prompt(raw_sample)

		sample = MultiResponseSample(
			prompt=prompt,
			dataset_type=raw_sample["dataset_type"],
			ground_truth=ground_truth,
			problem=raw_sample.get(self._config.problem_key, None) if self._config.problem_key else None,
			sample_id=sample_id,
		)
		if self.stop_event.is_set():
			self._clear_sample(sample)
			return sample

		branch_tasks = [
			asyncio.create_task(self._run_single_branch(raw_sample, sample_id))
			for _ in range(self._config.n_samples)
		]
		branch_results = await asyncio.gather(*branch_tasks, return_exceptions=True)

		outcomes: List[RockBranchResult] = []
		for branch_result in branch_results:
			if isinstance(branch_result, Exception):
				self.logger.warning("Rock branch failed: %s", branch_result)
				outcomes.append(self._build_failed_branch(prompt, str(branch_result)))
			else:
				outcomes.append(branch_result)

		sample.responses = [outcome.response for outcome in outcomes]
		sample.group_messages = [outcome.messages for outcome in outcomes]
		sample.rewards = [outcome.reward for outcome in outcomes]
		sample.reward_metrics_list = [deepcopy(outcome.reward_metrics) for outcome in outcomes]
		sample.avg_reward_metrics = self._build_avg_reward_metrics(sample.reward_metrics_list)
		sample.normed_rewards = compute_normed_rewards(
			rewards=sample.rewards,
			valid_flags=[outcome.valid for outcome in outcomes],
			advantage_estimator=self._config.advantage_estimator,
		)
		sample.completion_tokens = sum(outcome.completion_tokens for outcome in outcomes)
		sample.total_tokens = sum(outcome.total_tokens for outcome in outcomes)
		return sample

	async def _run_single_branch(self, raw_sample: dict, sample_id: str) -> RockBranchResult:
		sample_config = self._extract_sample_config(raw_sample)
		submit_result = await self._rock_service.submit_task(
			RockSubmitTask(
				sample_id=sample_id,
				url_list=self._target_urls,
				sample_config=sample_config,
				timeout=self._config.submit_request_timeout,
				retry=self._config.submit_retry,
			)
		)
		result_payload = await self._poll_result(submit_result.task_id)
		return self._build_branch_result(raw_sample, result_payload)

	async def _poll_result(self, task_id: str) -> Dict[str, Any]:
		attempt = 0
		start_time = asyncio.get_running_loop().time()
		while True:
			attempt += 1
			payload = await self._rock_service.get_task_result(
				RockResultTask(
					task_id=task_id,
					timeout=self._config.result_request_timeout,
					retry=self._config.result_retry,
				)
			)
			task_status = str(payload.get("task_status", "")).lower()
			if task_status in {"completed", "failed"}:
				return payload

			elapsed = asyncio.get_running_loop().time() - start_time
			if elapsed >= self._config.poll_timeout_seconds:
				raise TimeoutError(f"Rock polling timed out for task_id={task_id}")
			if self._config.poll_max_attempts is not None and attempt >= self._config.poll_max_attempts:
				raise TimeoutError(f"Rock polling exceeded max attempts for task_id={task_id}")
			if self.stop_event.is_set():
				raise RuntimeError("stop_event set during Rock polling")
			await asyncio.sleep(self._config.poll_interval_seconds)

	def _build_branch_result(self, raw_sample: dict, payload: Dict[str, Any]) -> RockBranchResult:
		task_status = str(payload.get("task_status", "")).lower()
		raw_messages = payload.get("messages") or []
		raw_tools = payload.get("tools") or []
		reward_metrics = deepcopy(payload.get("reward_metrics") or {})
		reward_metrics["rock_ok"] = 1 if task_status == "completed" else 0

		if task_status != "completed":
			prompt = self._extract_prompt(raw_sample)
			extra_message = str(payload.get("extra_message", payload.get("message", "Rock task failed")))
			return self._build_failed_branch(prompt, extra_message, reward_metrics=reward_metrics)

		normalized_messages = self._normalize_openai_messages(raw_messages)
		rendered_messages = self._convert_openai_trace_to_messages(
			messages=normalized_messages,
			tools=self._normalize_tools(raw_tools),
		)
		response = self._extract_compat_response(normalized_messages, rendered_messages)
		completion_tokens, total_tokens = self._count_tokens(rendered_messages)
		reward = float(payload.get("reward", 0.0))

		return RockBranchResult(
			response=response,
			messages=rendered_messages,
			reward=reward,
			reward_metrics=reward_metrics,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			valid=bool(rendered_messages),
		)

	def _build_failed_branch(
		self,
		prompt: str,
		reason: str,
		reward_metrics: Optional[Dict[str, Any]] = None,
	) -> RockBranchResult:
		masked_content = prompt if prompt else (reason if reason else "Rock task failed")
		messages = [Message(content=masked_content, is_masked=True)]
		reward_metrics = deepcopy(reward_metrics or {})
		reward_metrics.setdefault("rock_ok", 0)
		return RockBranchResult(
			response="",
			messages=messages,
			reward=0.0,
			reward_metrics=reward_metrics,
			completion_tokens=0,
			total_tokens=sum(len(self._tokenizer.encode(msg.content, add_special_tokens=False)) for msg in messages),
			valid=False,
		)

	def _extract_sample_config(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
		sample_config = raw_sample.get(self._config.sample_config_key)
		if not isinstance(sample_config, dict):
			raise KeyError(
				f"RockWorker expects a dict field '{self._config.sample_config_key}' in dataset rows"
			)
		return deepcopy(sample_config)

	def _extract_prompt(self, raw_sample: Dict[str, Any]) -> str:
		prompt = raw_sample.get(self._config.prompt_key)
		if prompt is not None:
			return str(prompt)

		sample_config = raw_sample.get(self._config.sample_config_key)
		if isinstance(sample_config, dict):
			messages = sample_config.get("messages")
			if isinstance(messages, list):
				for message in reversed(messages):
					if str(message.get("role", "")).lower() == "user":
						return str(message.get("content") or "")
		return ""

	def _require_rock_service(self, service_name: str) -> AsyncRockService:
		service = self.service_dict.get(service_name)
		if service is None:
			raise RuntimeError(f"RockWorker missing service '{service_name}'")
		if not isinstance(service, AsyncRockService):
			raise TypeError(
				f"RockWorker requires AsyncRockService for '{service_name}', got {type(service).__name__}"
			)
		return service

	def _get_target_urls(self, service_name: str) -> List[str]:
		service = self.service_dict.get(service_name)
		if service is None:
			raise RuntimeError(f"RockWorker missing target service '{service_name}'")
		url_list = [resource.base_url for resource in getattr(service, "resources", [])]
		if not url_list:
			raise RuntimeError(f"Target service '{service_name}' has no resource base URLs")
		return url_list

	def _normalize_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		return normalize_tools(tools)

	def _normalize_openai_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		return normalize_openai_messages(messages)

	def _convert_openai_trace_to_messages(
		self,
		messages: List[Dict[str, Any]],
		tools: List[Dict[str, Any]],
	) -> List[Message]:
		return convert_openai_trace_to_messages(
			tokenizer=self._tokenizer,
			messages=messages,
			tools=tools,
			chat_template=self._chat_template,
			add_generation_prompt=self._config.add_generation_prompt,
		)

	def _parse_rendered_messages(self, rendered_text: str) -> List[Message]:
		return parse_rendered_messages(rendered_text)

	def _render_chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
		return render_chat(
			tokenizer=self._tokenizer,
			messages=messages,
			tools=tools,
			chat_template=self._chat_template,
			add_generation_prompt=self._config.add_generation_prompt,
		)

	def _extract_compat_response(
		self,
		normalized_messages: List[Dict[str, Any]],
		rendered_messages: List[Message],
	) -> str:
		return extract_compat_response(normalized_messages)

	def _count_tokens(self, messages: List[Message]) -> Tuple[int, int]:
		return count_tokens(self._tokenizer, messages)

	def _build_avg_reward_metrics(self, reward_metrics_list: List[Dict[str, Any]]) -> List[str]:
		avg_reward_metrics: List[str] = []
		for reward_metrics in reward_metrics_list:
			for key in reward_metrics.keys():
				if key not in avg_reward_metrics:
					avg_reward_metrics.append(key)
		return avg_reward_metrics

	def _clear_sample(self, sample: MultiResponseSample) -> None:
		sample.responses = []
		sample.group_messages = []
		sample.rewards = []
		sample.reward_metrics_list = []
		sample.avg_reward_metrics = []
		sample.normed_rewards = []
		sample.completion_tokens = 0
		sample.total_tokens = 0

	@classmethod
	def _get_cached_template_artifacts(
		cls,
		tokenizer_path: str,
		trust_remote_code: bool,
		chat_template_path: Optional[str],
	):
		return get_cached_template_artifacts(
			tokenizer_path=tokenizer_path,
			trust_remote_code=trust_remote_code,
			chat_template_path=chat_template_path,
			owner_name="RockWorker",
		)
