"""Executor helpers for async rollout v2."""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional

_THREAD_POOL: Optional[ThreadPoolExecutor] = None
_PROCESS_POOL: Optional[ProcessPoolExecutor] = None


def get_thread_pool(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
	global _THREAD_POOL
	if _THREAD_POOL is None:
		_THREAD_POOL = ThreadPoolExecutor(max_workers=max_workers)
	return _THREAD_POOL


def get_process_pool(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
	global _PROCESS_POOL
	if _PROCESS_POOL is None:
		_PROCESS_POOL = ProcessPoolExecutor(max_workers=max_workers)
	return _PROCESS_POOL


def close_all() -> None:
	global _THREAD_POOL, _PROCESS_POOL
	if _THREAD_POOL is not None:
		_THREAD_POOL.shutdown(wait=True, cancel_futures=True)
		_THREAD_POOL = None
	if _PROCESS_POOL is not None:
		_PROCESS_POOL.shutdown(wait=True, cancel_futures=True)
		_PROCESS_POOL = None
