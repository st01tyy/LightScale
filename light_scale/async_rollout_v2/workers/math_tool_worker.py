"""Async math tool worker for async rollout v2."""

import asyncio
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.executors import get_thread_pool
from light_scale.async_rollout_v2.utils.llm_judge import (
    LLMJudgeConfig,
    judge_single_response,
)
from light_scale.async_rollout_v2.workers.function_call_worker import (
    AsyncFunctionCallWorker,
    AsyncFunctionCallWorkerConfig,
    FunctionCallBranchResult,
)
from light_scale.data import MultiResponseSample


DEFAULT_MATH_TASK_PROMPT = (
    "You are a professional mathematics expert.\n"
    "Please solve the provided problem step-by-step.\n"
    "You MUST use the Python interpreter tool for any complex calculations or to verify your logical steps to ensure 100% accuracy.\n\n"
    "The Python tool does NOT preserve any code or variables between calls.\n"
    "Every tool call must contain a complete, standalone, directly executable script.\n"
    "Do NOT submit incremental code lines, partial patches, or references to previously defined variables/functions.\n\n"
    "Safety requirements for any generated code:\n"
    "1) Code must be able to finish within 5 seconds in local execution.\n"
    "2) Internet/network access is strictly forbidden.\n"
    "3) `os` library usage is strictly forbidden.\n"
    "4) Creating threads/processes (or any concurrency primitives) is strictly forbidden.\n"
    "5) If a requested operation violates these rules, do not generate that code; use safe mathematical alternatives instead."
)


@lru_cache(maxsize=None)
def _load_task_prompt(task_prompt_path: Optional[str]) -> str:
    if task_prompt_path is None:
        return DEFAULT_MATH_TASK_PROMPT
    return Path(task_prompt_path).expanduser().read_text(encoding="utf-8").strip()


def validate_generated_code_safety(code: str) -> Tuple[bool, str]:
    if code is None:
        return False, "empty code"

    clean_code = code.replace("```python", "").replace("```", "").strip()
    lower_code = clean_code.lower()

    forbidden_substrings = [
        "import os",
        "from os import",
        "__import__('os'",
        '__import__("os"',
        "import socket",
        "from socket import",
        "socket.",
        "import requests",
        "from requests import",
        "requests.",
        "import urllib",
        "from urllib",
        "urllib.",
        "import http",
        "from http",
        "http.",
        "import httpx",
        "from httpx",
        "httpx.",
        "import aiohttp",
        "from aiohttp",
        "aiohttp.",
        "import ftplib",
        "from ftplib",
        "ftplib.",
        "import telnetlib",
        "from telnetlib",
        "telnetlib.",
        "import websocket",
        "from websocket",
        "websocket.",
        "import threading",
        "from threading import",
        "threading.",
        "import multiprocessing",
        "from multiprocessing",
        "multiprocessing.",
        "import concurrent",
        "from concurrent",
        "concurrent.futures",
        "threadpoolexecutor",
        "processpoolexecutor",
        "import ctypes",
        "from ctypes",
        "ctypes.",
        "import signal",
        "from signal import",
        "signal.raise_signal",
        "import faulthandler",
        "from faulthandler",
        "faulthandler.",
        "import resource",
        "from resource",
        "resource.",
        "os.kill",
        "sys.exit",
        "raise systemexit",
        "subprocess.",
        "import subprocess",
        "from subprocess",
        "pip install",
        "pip3 install",
        "python -m pip",
        "python3 -m pip",
        "conda install",
        "mamba install",
        "poetry add",
        "uv pip install",
        "easy_install",
        "apt-get install",
        "apt install",
        "yum install",
        "dnf install",
        "brew install",
        "pacman -s",
        "apk add",
    ]

    for token in forbidden_substrings:
        if token in lower_code:
            return False, f"forbidden usage detected: {token}"

    forbidden_patterns = [
        r"\bimport\s+os\b",
        r"\bfrom\s+os\s+import\b",
        r"\b__import__\s*\(\s*['\"]os['\"]\s*\)",
        r"\bimport\s+(threading|multiprocessing|concurrent|subprocess|socket|requests|urllib|http|httpx|aiohttp|ftplib|telnetlib|websocket|ctypes|signal|faulthandler|resource)\b",
        r"\bfrom\s+(threading|multiprocessing|concurrent|subprocess|socket|requests|urllib|http|httpx|aiohttp|ftplib|telnetlib|websocket|ctypes|signal|faulthandler|resource)\s+import\b",
        r"\bos\.kill\s*\(",
        r"\bsys\.exit\s*\(",
        r"\braise\s+SystemExit\b",
        r"\bsignal\.raise_signal\s*\(",
        r"\b(pip|pip3)\s+install\b",
        r"\bpython\d*\s*-m\s+pip\s+install\b",
        r"\b(conda|mamba|poetry|uv)\s+(install|add|pip\s+install)\b",
        r"\b(apt-get|apt|yum|dnf|brew|pacman|apk)\s+(install|add|-S)\b",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, clean_code, flags=re.IGNORECASE | re.MULTILINE):
            return False, f"forbidden pattern detected: {pattern}"

    return True, "ok"


_POLL_INTERVAL_SECONDS = 0.5
_TERMINATE_GRACE_SECONDS = 1.0


def _build_python_subprocess_script(code: str) -> str:
    clean_code = code.replace("```python", "").replace("```", "").strip()
    prelude = "import math\nimport numpy as np\n\n"
    return f"{prelude}{clean_code}\n"


def _read_output_preview(output_path: str, max_output_bytes: int) -> Tuple[str, bool]:
    if not os.path.exists(output_path):
        return "", False

    output_size = os.path.getsize(output_path)
    output_truncated = output_size > max_output_bytes
    with open(output_path, "rb") as output_file:
        raw_bytes = output_file.read(max_output_bytes)
    return raw_bytes.decode("utf-8", errors="ignore").strip(), output_truncated


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _execute_python_code_in_subprocess(code: str, timeout_seconds: int, max_output_bytes: int) -> Dict[str, Any]:
    work_dir = tempfile.mkdtemp(prefix=f"light_scale_math_tool_{uuid.uuid4().hex}_", dir="/tmp")
    code_path = os.path.join(work_dir, "main.py")
    output_path = os.path.join(work_dir, "output.txt")
    process: Optional[subprocess.Popen] = None
    status = "ok"

    try:
        Path(code_path).write_text(_build_python_subprocess_script(code), encoding="utf-8")

        with open(output_path, "wb") as output_file:
            process = subprocess.Popen(
                [sys.executable, code_path],
                stdout=output_file,
                stderr=subprocess.STDOUT,
                cwd=work_dir,
            )

        start_time = time.monotonic()
        while True:
            if process.poll() is not None:
                if process.returncode != 0:
                    status = "error"
                break

            if os.path.exists(output_path) and os.path.getsize(output_path) > max_output_bytes:
                status = "output_too_long"
                _terminate_process(process)
                break

            if time.monotonic() - start_time > timeout_seconds:
                status = "timeout"
                _terminate_process(process)
                break

            time.sleep(_POLL_INTERVAL_SECONDS)

        stdout_text, output_truncated = _read_output_preview(output_path, max_output_bytes)
        if output_truncated and status == "ok":
            status = "output_too_long"

        return {
            "status": status,
            "stdout": stdout_text,
            "stderr": "",
            "output_truncated": output_truncated or status == "output_too_long",
        }
    except Exception as err:
        if process is not None:
            try:
                _terminate_process(process)
            except Exception:
                pass
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Python runner error: {err}",
            "output_truncated": False,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@dataclass
class AsyncMathToolWorkerConfig(AsyncFunctionCallWorkerConfig):
    judge_service_name: str = "judge_model"
    python_timeout_seconds: int = 10
    enforce_code_safety: bool = True
    max_tool_output_bytes: int = 65536
    tool_output_preview_chars: int = 100
    judge_timeout: float = 600.0
    judge_retry: int = 2
    judge_json_retries: int = 2
    judge_temperature: float = 0.2
    judge_parallelism: int = 0
    judge_max_tokens: int = 1024
    enable_thinking: bool = False
    separate_reasoning: bool = False
    problem_key: str = "problem"
    ground_truth_key: str = "ground_truth"
    sample_id_key: str = "sample_id"
    dataset_type_key: str = "dataset_type"
    task_prompt_path: Optional[str] = None


class AsyncMathToolWorker(AsyncFunctionCallWorker):
    REQUIRED_SERVICE_NAMES = ["actor_model", "judge_model"]
    CONFIG_CLS = AsyncMathToolWorkerConfig
    USES_LLM_JUDGE = True

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
            **worker_cfg,
        )
        self._validate_unsupported_reasoning_config()
        self._judge_service = self._require_sglang_service(self._config.judge_service_name)

    def _build_initial_messages(self, sample: MultiResponseSample) -> List[Dict[str, Any]]:
        task_prompt = _load_task_prompt(self._config.task_prompt_path)
        return [
            {
                "role": "user",
                "content": (
                    f"{task_prompt}\n\n"
                    f"Question: {sample.problem}\n"
                    "Please provide your detailed derivation and the final answer."
                ),
            },
        ]

    def _build_initial_reward_metrics(self) -> Dict[str, Any]:
        return {
            "tool_ok": 1,
            "executor_ok": 1,
            "tool_round_limit_exceeded": 0,
            "tool_limit_final_turn_called_tool": 0,
        }

    def _build_failure_reward_metrics(self) -> Dict[str, Any]:
        return {
            "tool_ok": 0,
            "executor_ok": 0,
            "tool_round_limit_exceeded": 0,
            "tool_limit_final_turn_called_tool": 0,
            "tool_output_truncated": 0,
            "judge_ok": 0,
            "format": 0,
            "verdict": 0,
        }

    def _build_tool_feedback(
        self,
        execution_result: Dict[str, Any],
        remaining_context_tokens: Optional[int],
    ) -> Tuple[str, Dict[str, Any]]:
        stdout_text = str(execution_result.get("stdout") or "").strip()
        stderr_text = str(execution_result.get("stderr") or "").strip()
        status = str(execution_result.get("status") or "ok")
        output_truncated = bool(execution_result.get("output_truncated"))

        parts: List[str] = []
        if stdout_text:
            parts.append(stdout_text)
        if stderr_text:
            if parts:
                parts.append("[stderr]")
            parts.append(stderr_text)

        content = "\n".join(parts).strip() or "Success (No output)"
        metric_updates: Dict[str, Any] = {"tool_ok": 1, "executor_ok": 1, "tool_output_truncated": 0}

        if status == "timeout":
            metric_updates["executor_ok"] = 0
        elif status == "error":
            metric_updates["executor_ok"] = 0
        elif status == "output_too_long":
            metric_updates["tool_ok"] = 0
            metric_updates["executor_ok"] = 0
            metric_updates["tool_output_truncated"] = 1

        content_token_count = len(self._tokenizer.encode(content, add_special_tokens=False))
        should_summarize = output_truncated or (
            remaining_context_tokens is not None and content_token_count > remaining_context_tokens
        )
        if should_summarize:
            preview = content[: self._config.tool_output_preview_chars]
            if len(content) > self._config.tool_output_preview_chars:
                preview += "..."
            warning = "Output exceeded the local capture budget and execution was stopped." if output_truncated else "Output was truncated locally to preserve remaining context budget."
            if remaining_context_tokens is not None:
                warning += f" Estimated remaining context budget before the next actor turn: {remaining_context_tokens} tokens."
            content = f"{preview}\n\n[tool warning] {warning}"
            metric_updates["tool_output_truncated"] = 1

        return content, metric_updates

    async def _handle_tool_call(
        self,
        *,
        tool_call: Dict[str, Any],
        branch_tag: str,
        round_idx: int,
        tool_pos: int,
        sample: MultiResponseSample,
        trace_messages: List[Dict[str, Any]],
        remaining_context_tokens: Optional[int],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        function = tool_call.get("function") or {}
        function_name = function.get("name")
        tool_call_id = tool_call.get("id") or f"call_{branch_tag}_{round_idx}_{tool_pos}"

        if function_name != "execute_python":
            return (
                {
                    "role": "tool",
                    "content": f"Unsupported tool: {function_name}",
                    "tool_call_id": tool_call_id,
                },
                {"tool_ok": 0, "executor_ok": 0},
            )

        raw_arguments = function.get("arguments") or "{}"
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as err:
                return (
                    {
                        "role": "tool",
                        "content": f"Tool argument parse error: {err}",
                        "tool_call_id": tool_call_id,
                    },
                    {"tool_ok": 0, "executor_ok": 0, "tool_output_truncated": 0},
                )

        code = arguments.get("code", "")
        if self._config.enforce_code_safety:
            is_safe, reason = validate_generated_code_safety(code)
            if not is_safe:
                return (
                    {
                        "role": "tool",
                        "content": f"Python Safety Error: {reason}",
                        "tool_call_id": tool_call_id,
                    },
                    {"tool_ok": 0, "executor_ok": 0, "tool_output_truncated": 0},
                )

        loop = asyncio.get_running_loop()
        pool = get_thread_pool()
        execution_result = await loop.run_in_executor(
            pool,
            _execute_python_code_in_subprocess,
            code,
            self._config.python_timeout_seconds,
            self._config.max_tool_output_bytes,
        )
        tool_content, metric_updates = self._build_tool_feedback(
            execution_result=execution_result,
            remaining_context_tokens=remaining_context_tokens,
        )
        return (
            {
                "role": "tool",
                "content": tool_content,
                "tool_call_id": tool_call_id,
            },
            metric_updates,
        )

    async def _finalize_branch_result(
        self,
        sample: MultiResponseSample,
        branch_result: FunctionCallBranchResult,
    ) -> FunctionCallBranchResult:
        if self._has_length_limit_issue(branch_result.reward_metrics):
            branch_result.reward = 0.0
            branch_result.reward_metrics.setdefault("judge_ok", 0)
            branch_result.reward_metrics.setdefault("format", 0)
            branch_result.reward_metrics.setdefault("verdict", 0)
            branch_result.valid_for_advantage = branch_result.valid
            return self._apply_length_limit_policy(branch_result)

        reward, judge_metrics = await judge_single_response(
            judge_service=self._judge_service,
            sample=sample,
            response=branch_result.response,
            judge_config=LLMJudgeConfig.from_object(
                self._config,
                fallback_max_tokens=self._config.max_tokens,
            ),
            logger=self.logger,
        )
        branch_result.reward = reward
        branch_result.reward_metrics.update(judge_metrics)
        branch_result.valid_for_advantage = (
            branch_result.valid and int(branch_result.reward_metrics.get("judge_ok", 0)) == 1
        )
        return self._apply_length_limit_policy(branch_result)

    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code for precise mathematical calculations, algebraic simplification, or simulations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to run",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _estimate_local_message_token_delta(
        self,
        *,
        message: Dict[str, Any],
        trace_messages: List[Dict[str, Any]],
    ) -> int:
        content = "" if message.get("content") is None else str(message.get("content"))
        role = str(message.get("role", "")).lower()
        if role != "tool":
            return len(self._tokenizer.encode(content, add_special_tokens=False))

        previous_role = ""
        if len(trace_messages) >= 2 and isinstance(trace_messages[-2], dict):
            previous_role = str(trace_messages[-2].get("role", "")).lower()

        rendered_parts: List[str] = []
        if previous_role != "tool":
            rendered_parts.append("<|im_start|>observation")
        rendered_parts.append("\n")
        if "<tool_response>" not in content:
            rendered_parts.append("<tool_response>\n")
        rendered_parts.append(content)
        if "</tool_response>" not in content:
            rendered_parts.append("\n</tool_response>")
        if previous_role != "tool":
            rendered_parts.append("<|im_end|>\n")

        rendered_delta = "".join(rendered_parts)
        return len(self._tokenizer.encode(rendered_delta, add_special_tokens=False))

    def _build_actor_extra_body(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("math_tool_worker does not support actor extra_body or reasoning controls yet")

    def _on_tool_round_limit_exceeded(self, reward_metrics: Dict[str, Any]) -> None:
        super()._on_tool_round_limit_exceeded(reward_metrics)
        reward_metrics["tool_ok"] = 0

    def _validate_unsupported_reasoning_config(self) -> None:
        if self._config.enable_thinking:
            raise NotImplementedError("math_tool_worker does not support enable_thinking yet")
        if self._config.separate_reasoning:
            raise NotImplementedError("math_tool_worker does not support separate_reasoning yet")
        if self._config.actor_extra_body is not None:
            raise NotImplementedError("math_tool_worker does not support actor_extra_body yet")