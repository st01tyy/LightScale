"""AsyncSGLangService for async rollout v2."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientConnectionError, ServerDisconnectedError

from light_scale.data import Resource
from light_scale.async_rollout_v2.services.base_service import AsyncBaseService


@dataclass
class SGLangInputTooLongError(RuntimeError):
	message: str
	input_tokens: int
	max_context_tokens: int
	current_max_tokens: Optional[int] = None
	requested_completion_tokens: Optional[int] = None

	def __str__(self) -> str:
		return self.message


@dataclass
class _ContextOverflowDetails:
	message: str
	input_tokens: int
	max_context_tokens: int
	current_max_tokens: Optional[int] = None
	requested_completion_tokens: Optional[int] = None


@dataclass
class SGLangTask:
	"""描述一次 SGLang 推理请求的任务结构。"""

	prompt: str
	n_samples: int
	max_tokens: int
	temperature: float = 1.0
	top_p: float = 1.0
	top_k: int = -1
	presence_penalty: float = 0.0
	stop: Optional[str] = None
	add_stop: bool = False
	retry: int = 10
	timeout: Optional[float] = 900.0


@dataclass
class SGLangChatCompletionTask(SGLangTask):
	"""描述一次 SGLang ChatCompletion 请求任务。"""

	messages: List[Dict[str, Any]] = field(default_factory=list)
	tools: Optional[List[Dict[str, Any]]] = None
	tool_choice: Optional[Any] = None
	extra_body: Optional[Dict[str, Any]] = None
	response_format: Optional[Dict[str, Any]] = None


@dataclass
class SGLangResult:
	responses: List[Tuple[str, Optional[str]]]  # 每条样本对应(文本内容, 原始 finish_reason)
	completion_tokens: int
	total_tokens: int
	messages: Optional[List[Dict[str, Any]]] = None


class AsyncSGLangService(AsyncBaseService):
	"""异步 SGLang 推理服务封装。"""

	def __init__(
		self,
		name: str,
		resources: List[Resource],
		startup_timeout: float = 300.0,
		health_timeout: float = 30.0,
		health_retries: int = 3,
		max_concurrent_per_resource: int = 4,
		limit_per_host: int = 100,
		log_level: int = logging.INFO,
	):
		assert resources and len(resources) > 0, "AsyncSGLangService 初始化时必须提供至少一个 Resource"
		super().__init__(
			name=name,
			resources=resources,
			startup_timeout=startup_timeout,
			log_level=log_level,
		)
		self._health_timeout = health_timeout
		self._health_retries = health_retries
		self._max_concurrent_per_resource = max_concurrent_per_resource
		self._limit_per_host = limit_per_host
		self._session: Optional[aiohttp.ClientSession] = None
		self._base_urls = [res.base_url.rstrip("/") for res in resources]
		self._semaphores = {url: asyncio.Semaphore(max_concurrent_per_resource) for url in self._base_urls}
		self._url_index = 0
		self._url_lock = asyncio.Lock()
		self._session_lock = asyncio.Lock()
		self._session_version = 0

	async def _initialize(self) -> None:
		connector = aiohttp.TCPConnector(
			limit=self._max_concurrent_per_resource * len(self._base_urls), 
			limit_per_host=self._max_concurrent_per_resource, 
			ttl_dns_cache=300, force_close=True
		)
		timeout = aiohttp.ClientTimeout(total=self._health_timeout)
		self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
		self._session_version += 1

	async def _finalize(self) -> None:
		if self._session is not None:
			await self._session.close()
			self._session = None

	async def _recreate_session(self, expected_version: Optional[int] = None) -> bool:
		self.logger.warning("SGLang session disconnected, recreating...")
		async with self._session_lock:
			if expected_version is not None and self._session_version != expected_version:
				self.logger.warning("SGLang session version changed during recreation, skipping recreate")
				return False
			await self._finalize()
			await self._initialize()
			return True

	async def _health_check(self) -> None:
		if self._session is None:
			raise RuntimeError("AsyncSGLangService session 未初始化")
		for base_url in self._base_urls:
			health_url = f"{base_url}/health"
			healthy = False
			for attempt in range(1, self._health_retries + 1):
				try:
					async with self._session.get(health_url) as resp:
						if resp.status == 200:
							healthy = True
							break
				except Exception as err:
					self.logger.warning(
						"SGLang 服务 %s 健康检查失败(%d/%d): %s",
						health_url,
						attempt,
						self._health_retries,
						err,
					)
				if attempt < self._health_retries:
					await asyncio.sleep(1)
			if not healthy:
				raise RuntimeError(f"SGLang 服务 {health_url} 未就绪")

	async def request(self, task: SGLangTask) -> SGLangResult:
		if self._session is None:
			raise RuntimeError("AsyncSGLangService 未启动或 session 未初始化")
		base_url = await self._get_next_url()
		semaphore = self._semaphores[base_url]
		async with semaphore:
			return await self._call_sglang(base_url, task)

	async def _get_next_url(self) -> str:
		async with self._url_lock:
			url = self._base_urls[self._url_index % len(self._base_urls)]
			self._url_index += 1
			return url

	async def _call_sglang(self, url: str, task: SGLangTask) -> SGLangResult:
		if self._session is None:
			raise RuntimeError("AsyncSGLangService session 未初始化")
		api_path, payload = self._build_request_payload(task)
		effective_payload = dict(payload)
		has_retried_with_reduced_max_tokens = False
		timeout = task.timeout
		request_timeout = aiohttp.ClientTimeout(total=timeout)
		max_attempts = max(1, task.retry + 1)
		last_error: Optional[Exception] = None
		for attempt in range(1, max_attempts + 1):
			try:
				expected_version = self._session_version
				async with self._session.post(
					f"{url}{api_path}",
					headers={"Content-Type": "application/json", "Accept": "application/json"},
					data=json.dumps(effective_payload),
					timeout=request_timeout,
				) as resp:
					if resp.status != 200:
						text = await resp.text()
						overflow_details = self._parse_context_overflow_details(
							response_status_code=resp.status,
							response_text=text,
							current_max_tokens=effective_payload.get("max_tokens"),
						)
						adjusted_max_tokens = self._get_adjusted_max_tokens_from_bad_request(overflow_details)
						if adjusted_max_tokens is not None and not has_retried_with_reduced_max_tokens:
							old_max_tokens = effective_payload.get("max_tokens")
							effective_payload["max_tokens"] = adjusted_max_tokens
							has_retried_with_reduced_max_tokens = True
							self.logger.warning(
								"SGLang context overflow on %s: adjust max_tokens from %s to %s and retry once",
								url,
								old_max_tokens,
								adjusted_max_tokens,
							)
							continue
						if overflow_details is not None:
							raise SGLangInputTooLongError(
								message=overflow_details.message,
								input_tokens=overflow_details.input_tokens,
								max_context_tokens=overflow_details.max_context_tokens,
								current_max_tokens=overflow_details.current_max_tokens,
								requested_completion_tokens=overflow_details.requested_completion_tokens,
							)
						raise RuntimeError(f"status code: {resp.status}, response: {text}")
					response_data = await resp.json()
				break
			except SGLangInputTooLongError:
				raise
			except asyncio.TimeoutError as err:
				last_error = err
				self.logger.warning(
					"SGLang request timeout on %s (attempt %d/%d)",
					url,
					attempt,
					max_attempts,
				)
				if attempt < max_attempts:
					await asyncio.sleep(0.2 * attempt)
					continue
				raise
			except ServerDisconnectedError as err:
				last_error = err
				self.logger.warning(
					"SGLang request server disconnected on %s (attempt %d/%d): %s",
					url,
					attempt,
					max_attempts,
					err,
				)
				if attempt < max_attempts:
					await self._recreate_session(expected_version=expected_version)
					continue
				raise
			except ClientConnectionError as err:
				last_error = err
				self.logger.warning(
					"SGLang request connection error on %s (attempt %d/%d): %s",
					url,
					attempt,
					max_attempts,
					err,
				)
				if attempt < max_attempts:
					continue
				raise
			except aiohttp.ClientError as err:
				last_error = err
				self.logger.warning(
					"SGLang client error on %s (attempt %d/%d): %s",
					url,
					attempt,
					max_attempts,
					err,
				)
				if attempt < max_attempts:
					await asyncio.sleep(0.2 * attempt)
					continue
				raise
			except Exception as err:
				last_error = err
				self.logger.warning(
					"SGLang unknown error on %s (attempt %d/%d): %s",
					url,
					attempt,
					max_attempts,
					err,
				)
				self.logger.warning(f"{vars(task)}")
				if attempt < max_attempts:
					await asyncio.sleep(0.2 * attempt)
					continue
				raise
		else:
			raise RuntimeError(f"SGLang request failed after retries: {last_error}")

		responses = self._parse_response_data(task, response_data)
		usage = response_data.get("usage", {})
		completion_tokens = usage.get("completion_tokens", 0)
		total_tokens = usage.get("total_tokens", 0)
		return SGLangResult(
			responses=responses,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			messages=response_data.get("_parsed_messages"),
		)

	def _parse_context_overflow_details(
		self,
		response_status_code: int,
		response_text: str,
		current_max_tokens: Any,
	) -> Optional[_ContextOverflowDetails]:
		if response_status_code != 400:
			return None

		try:
			response_json = json.loads(response_text)
			err_msg = response_json.get("message", "")
		except Exception:
			err_msg = response_text

		max_context_match = re.search(
			r"maximum context length of\s+(\d+)\s+tokens",
			err_msg,
			re.IGNORECASE,
		)
		short_context_match = re.search(
			r"input\s*\((\d+)\s+tokens\)\s+is\s+longer\s+than\s+the\s+model'?s\s+context\s+length\s*\((\d+)\s+tokens\)",
			err_msg,
			re.IGNORECASE,
		)
		input_tokens_match = re.search(
			r"(\d+)\s+tokens from the input (?:messages|prompt)|(?:input messages|input prompt|messages|prompt)\s*[:=]?\s*(\d+)\s*tokens",
			err_msg,
			re.IGNORECASE,
		)
		completion_tokens_match = re.search(
			r"and\s+(\d+)\s+tokens for the completion",
			err_msg,
			re.IGNORECASE,
		)
		if short_context_match is not None:
			input_tokens = int(short_context_match.group(1))
			max_context_len = int(short_context_match.group(2))
			requested_completion_tokens = current_max_tokens if isinstance(current_max_tokens, int) else None
			return _ContextOverflowDetails(
				message=err_msg,
				input_tokens=input_tokens,
				max_context_tokens=max_context_len,
				current_max_tokens=current_max_tokens if isinstance(current_max_tokens, int) else None,
				requested_completion_tokens=requested_completion_tokens,
			)

		if max_context_match is None or input_tokens_match is None:
			return None

		max_context_len = int(max_context_match.group(1))
		if input_tokens_match.group(1) is not None:
			input_tokens = int(input_tokens_match.group(1))
		else:
			input_tokens = int(input_tokens_match.group(2))
		requested_completion_tokens = None
		if completion_tokens_match is not None:
			requested_completion_tokens = int(completion_tokens_match.group(1))

		return _ContextOverflowDetails(
			message=err_msg,
			input_tokens=input_tokens,
			max_context_tokens=max_context_len,
			current_max_tokens=current_max_tokens if isinstance(current_max_tokens, int) else None,
			requested_completion_tokens=requested_completion_tokens,
		)

	def _get_adjusted_max_tokens_from_bad_request(
		self,
		overflow_details: Optional[_ContextOverflowDetails],
	) -> Optional[int]:
		if overflow_details is None:
			return None

		safe_max_tokens = overflow_details.max_context_tokens - overflow_details.input_tokens - 100
		if safe_max_tokens <= 0:
			return None

		effective_current_max_tokens = overflow_details.current_max_tokens
		if not isinstance(effective_current_max_tokens, int):
			effective_current_max_tokens = overflow_details.requested_completion_tokens
		if not isinstance(effective_current_max_tokens, int):
			return None
		if safe_max_tokens >= effective_current_max_tokens:
			return None
		return safe_max_tokens

	def _build_request_payload(self, task: SGLangTask) -> Tuple[str, Dict[str, Any]]:
		if isinstance(task, SGLangChatCompletionTask):
			payload: Dict[str, Any] = {
				"messages": task.messages,
				"stream": False,
				"model": self.name,
				"n": task.n_samples,
				"max_tokens": task.max_tokens,
				"temperature": task.temperature,
				"top_p": task.top_p,
				"top_k": task.top_k,
			}
			if task.tools is not None:
				payload["tools"] = task.tools
			if task.tool_choice is not None:
				payload["tool_choice"] = task.tool_choice
			if task.extra_body is not None:
				payload["extra_body"] = task.extra_body
			if task.response_format is not None:
				payload["response_format"] = task.response_format
			if task.stop is not None:
				payload["stop"] = task.stop
			return "/v1/chat/completions", payload

		payload = {
			"prompt": task.prompt,
			"stream": False,
			"model": self.name,
			"n": task.n_samples,
			"max_tokens": task.max_tokens,
			"temperature": task.temperature,
			"top_p": task.top_p,
			"top_k": task.top_k,
		}
		if task.stop is not None:
			payload["stop"] = task.stop
		return "/v1/completions", payload

	def _parse_response_data(
		self,
		task: SGLangTask,
		response_data: Dict[str, Any],
	) -> List[Tuple[str, Optional[str]]]:
		choices = response_data.get("choices", [])
		if len(choices) != task.n_samples:
			raise RuntimeError(f"Expected {task.n_samples} choices, but got {len(choices)}")

		if isinstance(task, SGLangChatCompletionTask):
			responses: List[Tuple[str, Optional[str]]] = []
			parsed_messages: List[Dict[str, Any]] = []
			for choice in choices:
				message = choice.get("message", {})
				text = message.get("content", "") if isinstance(message, dict) else ""
				finish_reason = choice.get("finish_reason")
				responses.append((text, finish_reason))
				if isinstance(message, dict):
					parsed_message: Dict[str, Any] = {
						"role": str(message.get("role", "assistant") or "assistant").lower(),
						"content": "" if message.get("content") is None else str(message.get("content")),
						"finish_reason": finish_reason,
					}
					if message.get("tool_calls") is not None:
						parsed_message["tool_calls"] = message.get("tool_calls")
					if message.get("reasoning_content") is not None:
						parsed_message["reasoning_content"] = message.get("reasoning_content")
					if message.get("name") is not None:
						parsed_message["name"] = str(message.get("name"))
					parsed_messages.append(parsed_message)
				else:
					parsed_messages.append({"role": "assistant", "content": text})
			response_data["_parsed_messages"] = parsed_messages
			return responses

		responses = []
		for choice in choices:
			text = choice.get("text", "")
			finish_reason = choice.get("finish_reason")
			if finish_reason == "stop" and task.stop and task.add_stop and not text.endswith(task.stop):
				text += task.stop
			responses.append((text, finish_reason))
		return responses

