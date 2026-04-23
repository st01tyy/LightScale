"""Async Rock service for async rollout v2."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientConnectionError, ServerDisconnectedError

from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.data import Resource


@dataclass
class RockSubmitTask:
    sample_id: str
    url_list: List[str]
    sample_config: Dict[str, Any]
    timeout: Optional[float] = None
    retry: Optional[int] = None


@dataclass
class RockResultTask:
    task_id: str
    timeout: Optional[float] = None
    retry: Optional[int] = None


@dataclass
class RockSubmitResult:
    status: str
    message: str
    task_id: str


class AsyncRockService(AsyncBaseService):
    """Async Rock client for submit and result polling APIs."""

    def __init__(
        self,
        name: str,
        resources: List[Resource],
        startup_timeout: float = 300.0,
        health_timeout: float = 30.0,
        health_retries: int = 3,
        request_timeout: float = 60.0,
        retry: int = 3,
        max_concurrent_per_resource: int = 16,
        limit_per_host: int = 100,
        submit_path: str = "/submit",
        result_path: str = "/result",
        health_path: str = "/health",
        log_level: int = logging.INFO,
    ):
        assert resources and len(resources) > 0, "AsyncRockService requires at least one Resource"
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
        self._submit_path = self._normalize_path(submit_path)
        self._result_path = self._normalize_path(result_path)
        self._health_path = self._normalize_path(health_path)
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
            raise RuntimeError("AsyncRockService session is not initialized")

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
                        "Rock service %s health check failed (%d/%d): %s",
                        health_url,
                        attempt,
                        self._health_retries,
                        err,
                    )
                if attempt < self._health_retries:
                    await asyncio.sleep(1)
            if not healthy:
                raise RuntimeError(f"Rock service {health_url} is not ready")

    async def request(self, task):
        if isinstance(task, RockSubmitTask):
            return await self.submit_task(task)
        if isinstance(task, RockResultTask):
            return await self.get_task_result(task)
        raise TypeError(f"Unsupported Rock task type: {type(task).__name__}")

    async def submit_task(self, task: RockSubmitTask) -> RockSubmitResult:
        response = await self._request_json(
            path=self._submit_path,
            payload={
                "sample_id": task.sample_id,
                "url_list": task.url_list,
                "sample_config": task.sample_config,
            },
            timeout=task.timeout,
            retry=task.retry,
        )
        status = str(response.get("status", ""))
        if status != "success":
            raise RuntimeError(
                f"Rock submit failed: status={status}, message={response.get('message', '')}"
            )
        task_id = response.get("task_id")
        if not task_id:
            raise RuntimeError("Rock submit succeeded but task_id is missing")
        return RockSubmitResult(
            status=status,
            message=str(response.get("message", "")),
            task_id=str(task_id),
        )

    async def get_task_result(self, task: RockResultTask) -> Dict[str, Any]:
        response = await self._request_json(
            path=self._result_path,
            payload={"task_id": task.task_id},
            timeout=task.timeout,
            retry=task.retry,
        )
        task_status = response.get("task_status")
        if task_status is None:
            raise RuntimeError("Rock result response missing task_status")
        return response

    async def _request_json(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout: Optional[float],
        retry: Optional[int],
    ) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("AsyncRockService is not started")

        base_url = await self._get_next_url()
        semaphore = self._semaphores[base_url]
        async with semaphore:
            return await self._post_json(
                url=f"{base_url}{path}",
                payload=payload,
                timeout=timeout if timeout is not None else self._request_timeout,
                retry=retry if retry is not None else self._retry,
            )

    async def _get_next_url(self) -> str:
        async with self._url_lock:
            url = self._base_urls[self._url_index % len(self._base_urls)]
            self._url_index += 1
            return url

    async def _recreate_session(self, expected_version: Optional[int] = None) -> bool:
        self.logger.warning("Rock session disconnected, recreating...")
        async with self._session_lock:
            if expected_version is not None and self._session_version != expected_version:
                return False
            await self._finalize()
            await self._initialize()
            return True

    async def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float,
        retry: int,
    ) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("AsyncRockService session is not initialized")

        request_timeout = aiohttp.ClientTimeout(total=timeout)
        max_attempts = max(1, retry + 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            expected_version = self._session_version
            try:
                async with self._session.post(
                    url,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    data=json.dumps(payload),
                    timeout=request_timeout,
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        raise RuntimeError(f"status code: {resp.status}, response: {text}")
                    try:
                        response = json.loads(text)
                    except json.JSONDecodeError as err:
                        raise RuntimeError(f"Rock response is not valid JSON: {text}") from err
                    if not isinstance(response, dict):
                        raise RuntimeError(f"Rock response must be a JSON object: {response}")
                    return response
            except asyncio.TimeoutError as err:
                last_error = err
                self.logger.warning(
                    "Rock request timeout on %s (attempt %d/%d)",
                    url,
                    attempt,
                    max_attempts,
                )
            except ServerDisconnectedError as err:
                last_error = err
                self.logger.warning(
                    "Rock request server disconnected on %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    max_attempts,
                    err,
                )
                if attempt < max_attempts:
                    await self._recreate_session(expected_version=expected_version)
            except ClientConnectionError as err:
                last_error = err
                self.logger.warning(
                    "Rock request connection error on %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    max_attempts,
                    err,
                )
            except aiohttp.ClientError as err:
                last_error = err
                self.logger.warning(
                    "Rock client error on %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    max_attempts,
                    err,
                )
            except Exception as err:
                last_error = err
                self.logger.warning(
                    "Rock request error on %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    max_attempts,
                    err,
                )

            if attempt < max_attempts:
                await asyncio.sleep(0.2 * attempt)

        raise RuntimeError(f"Rock request failed after retries: {last_error}")

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            raise ValueError("Rock service path must not be empty")
        return path if path.startswith("/") else f"/{path}"