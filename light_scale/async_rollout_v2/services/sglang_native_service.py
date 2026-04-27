"""Async native SGLang service for async rollout v2."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientConnectionError, ServerDisconnectedError

from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.data import Resource


LogprobTuple = Tuple[float, int, Optional[str]]
TopLogprobList = List[LogprobTuple]


@dataclass
class SGLangNativeGenerateTask:
	"""Describe one native /generate request."""

	text: Optional[str] = None
	input_ids: Optional[List[int]] = None
	sampling_params: Dict[str, Any] = field(default_factory=dict)
	return_logprob: bool = False
	logprob_start_len: int = -1
	top_logprobs_num: int = 0
	token_ids_logprob: Optional[List[int]] = None
	return_text_in_logprobs: bool = False
	stream: bool = False
	retry: int = 3
	timeout: Optional[float] = 300.0


@dataclass
class SGLangNativeMetaInfo:
	request_id: str
	finish_reason: Any
	prompt_tokens: int
	completion_tokens: int
	cached_tokens: int
	weight_version: Optional[int] = None
	total_retractions: Optional[int] = None
	input_token_logprobs: List[LogprobTuple] = field(default_factory=list)
	output_token_logprobs: List[LogprobTuple] = field(default_factory=list)
	input_top_logprobs: List[Optional[TopLogprobList]] = field(default_factory=list)
	output_top_logprobs: List[Optional[TopLogprobList]] = field(default_factory=list)
	input_token_ids_logprobs: Optional[List[TopLogprobList]] = None
	output_token_ids_logprobs: Optional[List[TopLogprobList]] = None


@dataclass
class SGLangNativeResult:
	text: str
	output_ids: List[int]
	meta_info: SGLangNativeMetaInfo
	raw_response: Optional[Dict[str, Any]] = None


def build_prefill_logprob_task(
	text: Optional[str] = None,
	input_ids: Optional[List[int]] = None,
	top_logprobs_num: int = 5,
	logprob_start_len: int = 0,
	return_text_in_logprobs: bool = True,
	timeout: float = 300.0,
	retry: int = 3,
	token_ids_logprob: Optional[List[int]] = None,
) -> SGLangNativeGenerateTask:
	"""Build a pure-prefill task for prompt-side logprob extraction."""

	return SGLangNativeGenerateTask(
		text=text,
		input_ids=input_ids,
		sampling_params={"temperature": 0, "max_new_tokens": 0},
		return_logprob=True,
		logprob_start_len=logprob_start_len,
		top_logprobs_num=top_logprobs_num,
		token_ids_logprob=token_ids_logprob,
		return_text_in_logprobs=return_text_in_logprobs,
		stream=False,
		retry=retry,
		timeout=timeout,
	)


class AsyncSGLangNativeService(AsyncBaseService):
	"""Async wrapper over SGLang native /generate."""

	def __init__(
		self,
		name: str,
		resources: List[Resource],
		startup_timeout: float = 300.0,
		health_timeout: float = 30.0,
		health_retries: int = 3,
		request_timeout: float = 300.0,
		retry: int = 3,
		max_concurrent_per_resource: int = 8,
		limit_per_host: int = 100,
		generate_path: str = "/generate",
		health_path: str = "/health",
		default_return_text_in_logprobs: bool = False,
		log_level: int = logging.INFO,
	):
		assert resources and len(resources) > 0, "AsyncSGLangNativeService 初始化时必须提供至少一个 Resource"
		super().__init__(
			name=name,
			resources=resources,
			startup_timeout=startup_timeout,
			log_level=log_level,
		)
		self._health_timeout = health_timeout
		self._health_retries = health_retries
		self._request_timeout = request_timeout
		self._retry = retry
		self._max_concurrent_per_resource = max_concurrent_per_resource
		self._limit_per_host = limit_per_host
		self._generate_path = self._normalize_path(generate_path)
		self._health_path = self._normalize_path(health_path)
		self._default_return_text_in_logprobs = default_return_text_in_logprobs
		self._session: Optional[aiohttp.ClientSession] = None
		self._base_urls = [res.base_url.rstrip("/") for res in resources]
		self._semaphores = {
			url: asyncio.Semaphore(max_concurrent_per_resource) for url in self._base_urls
		}
		self._url_index = 0
		self._url_lock = asyncio.Lock()
		self._session_lock = asyncio.Lock()
		self._session_version = 0

	async def _initialize(self) -> None:
		connector = aiohttp.TCPConnector(
			limit=self._max_concurrent_per_resource * len(self._base_urls),
			limit_per_host=self._limit_per_host,
			ttl_dns_cache=300,
			force_close=True,
		)
		timeout = aiohttp.ClientTimeout(total=self._health_timeout)
		self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
		self._session_version += 1

	async def _finalize(self) -> None:
		if self._session is not None:
			await self._session.close()
			self._session = None

	async def _health_check(self) -> None:
		if self._session is None:
			raise RuntimeError("AsyncSGLangNativeService session 未初始化")

		for base_url in self._base_urls:
			health_url = f"{base_url}{self._health_path}"
			healthy = False
			for attempt in range(1, self._health_retries + 1):
				try:
					async with self._session.get(health_url) as resp:
						if resp.status == 200:
							healthy = True
							break
				except Exception as err:
					self.logger.warning(
						"SGLang native 服务 %s 健康检查失败(%d/%d): %s",
						health_url,
						attempt,
						self._health_retries,
						err,
					)
				if attempt < self._health_retries:
					await asyncio.sleep(1)
			if not healthy:
				raise RuntimeError(f"SGLang native 服务 {health_url} 未就绪")

	async def request(self, task: SGLangNativeGenerateTask) -> SGLangNativeResult:
		self._validate_task(task)
		if self._session is None:
			raise RuntimeError("AsyncSGLangNativeService 未启动或 session 未初始化")

		base_url = await self._get_next_url()
		semaphore = self._semaphores[base_url]
		async with semaphore:
			return await self._call_generate(base_url, task)

	async def _get_next_url(self) -> str:
		async with self._url_lock:
			url = self._base_urls[self._url_index % len(self._base_urls)]
			self._url_index += 1
			return url

	async def _recreate_session(self, expected_version: Optional[int] = None) -> bool:
		self.logger.warning("SGLang native session disconnected, recreating...")
		async with self._session_lock:
			if expected_version is not None and self._session_version != expected_version:
				self.logger.warning("SGLang native session version changed during recreation, skipping recreate")
				return False
			await self._finalize()
			await self._initialize()
			return True

	async def _call_generate(self, url: str, task: SGLangNativeGenerateTask) -> SGLangNativeResult:
		if self._session is None:
			raise RuntimeError("AsyncSGLangNativeService session 未初始化")

		payload = self._build_request_payload(task)
		request_timeout = aiohttp.ClientTimeout(total=task.timeout or self._request_timeout)
		max_attempts = max(1, (task.retry if task.retry is not None else self._retry) + 1)
		last_error: Optional[Exception] = None

		for attempt in range(1, max_attempts + 1):
			expected_version = self._session_version
			try:
				async with self._session.post(
					f"{url}{self._generate_path}",
					headers={"Content-Type": "application/json", "Accept": "application/json"},
					data=json.dumps(payload),
					timeout=request_timeout,
				) as resp:
					text = await resp.text()
					if resp.status != 200:
						raise RuntimeError(f"status code: {resp.status}, response: {text}")
					try:
						response_data = json.loads(text)
					except json.JSONDecodeError as err:
						raise RuntimeError(f"SGLang native response is not valid JSON: {text}") from err
				if not isinstance(response_data, dict):
					raise RuntimeError(f"SGLang native response must be a JSON object: {response_data}")
				return self._parse_generate_response(response_data)
			except asyncio.TimeoutError as err:
				last_error = err
				self.logger.warning(
					"SGLang native request timeout on %s (attempt %d/%d)",
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
					"SGLang native request server disconnected on %s (attempt %d/%d): %s",
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
					"SGLang native request connection error on %s (attempt %d/%d): %s",
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
					"SGLang native client error on %s (attempt %d/%d): %s",
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
					"SGLang native unknown error on %s (attempt %d/%d): %s",
					url,
					attempt,
					max_attempts,
					err,
				)
				self.logger.warning("task=%s", vars(task))
				if attempt < max_attempts:
					await asyncio.sleep(0.2 * attempt)
					continue
				raise

		raise RuntimeError(f"SGLang native request failed after retries: {last_error}")

	def _build_request_payload(self, task: SGLangNativeGenerateTask) -> Dict[str, Any]:
		payload: Dict[str, Any] = {
			"sampling_params": dict(task.sampling_params),
			"return_logprob": task.return_logprob,
			"logprob_start_len": task.logprob_start_len,
			"top_logprobs_num": task.top_logprobs_num,
			"return_text_in_logprobs": (
				task.return_text_in_logprobs or self._default_return_text_in_logprobs
			),
			"stream": False,
		}
		if task.text is not None:
			payload["text"] = task.text
		if task.input_ids is not None:
			payload["input_ids"] = task.input_ids
		if task.token_ids_logprob is not None:
			payload["token_ids_logprob"] = task.token_ids_logprob
		return payload

	def _parse_generate_response(self, response_data: Dict[str, Any]) -> SGLangNativeResult:
		meta_info_data = response_data.get("meta_info")
		if not isinstance(meta_info_data, dict):
			raise RuntimeError(f"SGLang native response missing meta_info dict: {response_data}")

		text = response_data.get("text")
		if text is None:
			text = ""
		if not isinstance(text, str):
			text = str(text)

		output_ids = response_data.get("output_ids") or []
		if not isinstance(output_ids, list) or any(not isinstance(token_id, int) for token_id in output_ids):
			raise RuntimeError(f"SGLang native response output_ids is invalid: {output_ids}")

		return SGLangNativeResult(
			text=text,
			output_ids=output_ids,
			meta_info=self._parse_meta_info(meta_info_data),
		)

	def _parse_meta_info(self, meta_info_data: Dict[str, Any]) -> SGLangNativeMetaInfo:
		prompt_tokens = meta_info_data.get("prompt_tokens")
		if not isinstance(prompt_tokens, int):
			raise RuntimeError(f"SGLang native meta_info.prompt_tokens is invalid: {prompt_tokens}")

		return SGLangNativeMetaInfo(
			request_id=str(meta_info_data.get("id", "")),
			finish_reason=meta_info_data.get("finish_reason"),
			prompt_tokens=prompt_tokens,
			completion_tokens=self._optional_int(meta_info_data.get("completion_tokens"), default=0),
			cached_tokens=self._optional_int(meta_info_data.get("cached_tokens"), default=0),
			weight_version=self._optional_int(meta_info_data.get("weight_version")),
			total_retractions=self._optional_int(meta_info_data.get("total_retractions")),
			input_token_logprobs=self._parse_logprob_tuple_list(meta_info_data.get("input_token_logprobs")),
			output_token_logprobs=self._parse_logprob_tuple_list(meta_info_data.get("output_token_logprobs")),
			input_top_logprobs=self._parse_top_logprobs(meta_info_data.get("input_top_logprobs")),
			output_top_logprobs=self._parse_top_logprobs(meta_info_data.get("output_top_logprobs")),
			input_token_ids_logprobs=self._parse_optional_token_id_logprobs(
				meta_info_data.get("input_token_ids_logprobs")
			),
			output_token_ids_logprobs=self._parse_optional_token_id_logprobs(
				meta_info_data.get("output_token_ids_logprobs")
			),
		)

	def _parse_logprob_tuple_list(self, value: Any) -> List[LogprobTuple]:
		if value is None:
			return []
		if not isinstance(value, list):
			raise RuntimeError(f"SGLang native logprob list is invalid: {value}")
		return [self._parse_logprob_tuple(item) for item in value]

	def _parse_top_logprobs(self, value: Any) -> List[Optional[TopLogprobList]]:
		if value is None:
			return []
		if not isinstance(value, list):
			raise RuntimeError(f"SGLang native top logprobs is invalid: {value}")

		parsed: List[Optional[TopLogprobList]] = []
		for item in value:
			if item is None:
				parsed.append(None)
				continue
			if not isinstance(item, list):
				raise RuntimeError(f"SGLang native top logprobs entry is invalid: {item}")
			parsed.append([self._parse_logprob_tuple(entry) for entry in item])
		return parsed

	def _parse_optional_token_id_logprobs(self, value: Any) -> Optional[List[TopLogprobList]]:
		if value is None:
			return None
		parsed = self._parse_top_logprobs(value)
		return [item or [] for item in parsed]

	def _parse_logprob_tuple(self, value: Any) -> LogprobTuple:
		if not isinstance(value, (list, tuple)) or len(value) != 3:
			raise RuntimeError(f"SGLang native logprob tuple is invalid: {value}")

		logprob = value[0]
		token_id = value[1]
		token_text = value[2]
		if not isinstance(logprob, (int, float)):
			raise RuntimeError(f"SGLang native logprob value is invalid: {value}")
		if not isinstance(token_id, int):
			raise RuntimeError(f"SGLang native token id is invalid: {value}")
		if token_text is not None and not isinstance(token_text, str):
			token_text = str(token_text)
		return (float(logprob), token_id, token_text)

	def _validate_task(self, task: SGLangNativeGenerateTask) -> None:
		has_text = task.text is not None
		has_input_ids = task.input_ids is not None
		if has_text == has_input_ids:
			raise ValueError("SGLangNativeGenerateTask 必须且只能提供 text 或 input_ids 其中之一")
		if task.stream:
			raise ValueError("AsyncSGLangNativeService 当前仅支持非流式请求")
		if task.top_logprobs_num < 0:
			raise ValueError("top_logprobs_num 不能小于 0")
		if not task.return_logprob and task.top_logprobs_num > 0:
			raise ValueError("top_logprobs_num > 0 时必须启用 return_logprob")
		if not isinstance(task.sampling_params, dict):
			raise ValueError("sampling_params 必须是 dict")
		if task.input_ids is not None:
			if not isinstance(task.input_ids, list) or any(not isinstance(token_id, int) for token_id in task.input_ids):
				raise ValueError("input_ids 必须是 int 列表")
		if task.token_ids_logprob is not None:
			if not isinstance(task.token_ids_logprob, list) or any(
				not isinstance(token_id, int) for token_id in task.token_ids_logprob
			):
				raise ValueError("token_ids_logprob 必须是 int 列表")

	@staticmethod
	def _optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
		if value is None:
			return default
		if not isinstance(value, int):
			raise RuntimeError(f"Expected int-compatible field, got: {value}")
		return value

	@staticmethod
	def _normalize_path(path: str) -> str:
		path = str(path or "").strip()
		if not path:
			return "/"
		return path if path.startswith("/") else f"/{path}"