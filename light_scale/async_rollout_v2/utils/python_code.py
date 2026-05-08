"""Structured local Python execution utilities for async rollout v2."""

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Optional, Tuple


class PythonCodeExecutionStatus(str, Enum):
    OK = "ok"
    SAFETY_VIOLATION = "safety_violation"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    OUTPUT_TOO_LONG = "output_too_long"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class PythonCodeExecutionRequest:
    code: str
    timeout_seconds: int
    max_output_bytes: int
    stdin_text: Optional[str] = None
    prelude: str = "import math\nimport numpy as np\n"
    tmp_dir_root: str = "/tmp"
    filename: str = "main.py"
    input_filename: str = "input.txt"
    output_filename: str = "output.txt"


@dataclass(frozen=True)
class PythonCodeExecutionResult:
    status: PythonCodeExecutionStatus
    stdout: str
    output_truncated: bool
    exit_code: Optional[int] = None
    failure_reason: str = ""


_POLL_INTERVAL_SECONDS = 0.5
_TERMINATE_GRACE_SECONDS = 1.0


def validate_generated_code_safety(code: str) -> Tuple[bool, str]:
    if code is None:
        return False, "empty code"

    clean_code = _normalize_code(code)
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


def execute_python_code(request: PythonCodeExecutionRequest) -> PythonCodeExecutionResult:
    clean_code = _normalize_code(request.code)
    is_safe, safety_reason = validate_generated_code_safety(clean_code)
    if not is_safe:
        return PythonCodeExecutionResult(
            status=PythonCodeExecutionStatus.SAFETY_VIOLATION,
            stdout=f"Python Safety Error: {safety_reason}",
            output_truncated=False,
            failure_reason=safety_reason,
        )

    work_dir = tempfile.mkdtemp(
        prefix=f"light_scale_python_code_{uuid.uuid4().hex}_",
        dir=request.tmp_dir_root,
    )
    code_path = os.path.join(work_dir, request.filename)
    input_path = os.path.join(work_dir, request.input_filename)
    output_path = os.path.join(work_dir, request.output_filename)
    process: Optional[subprocess.Popen] = None

    try:
        script_text = _build_python_subprocess_script(clean_code, request.prelude)
        Path(code_path).write_text(script_text, encoding="utf-8")
        if request.stdin_text is not None:
            Path(input_path).write_text(request.stdin_text, encoding="utf-8")

        syntax_result = _run_python_syntax_check(code_path)
        if syntax_result is not None:
            return syntax_result

        with open(output_path, "wb") as output_file:
            if request.stdin_text is None:
                process = subprocess.Popen(
                    [sys.executable, code_path],
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    cwd=work_dir,
                )
            else:
                with open(input_path, "rb") as input_file:
                    process = subprocess.Popen(
                        [sys.executable, code_path],
                        stdin=input_file,
                        stdout=output_file,
                        stderr=subprocess.STDOUT,
                        cwd=work_dir,
                    )

                    start_time = time.monotonic()
                    status = PythonCodeExecutionStatus.OK
                    while True:
                        if process.poll() is not None:
                            if process.returncode != 0:
                                status = PythonCodeExecutionStatus.RUNTIME_ERROR
                            break

                        if os.path.exists(output_path) and os.path.getsize(output_path) > request.max_output_bytes:
                            status = PythonCodeExecutionStatus.OUTPUT_TOO_LONG
                            _terminate_process(process)
                            break

                        if time.monotonic() - start_time > request.timeout_seconds:
                            status = PythonCodeExecutionStatus.TIMEOUT
                            _terminate_process(process)
                            break

                        time.sleep(_POLL_INTERVAL_SECONDS)

            if request.stdin_text is None:
                start_time = time.monotonic()
                status = PythonCodeExecutionStatus.OK
                while True:
                    if process.poll() is not None:
                        if process.returncode != 0:
                            status = PythonCodeExecutionStatus.RUNTIME_ERROR
                        break

                    if os.path.exists(output_path) and os.path.getsize(output_path) > request.max_output_bytes:
                        status = PythonCodeExecutionStatus.OUTPUT_TOO_LONG
                        _terminate_process(process)
                        break

                    if time.monotonic() - start_time > request.timeout_seconds:
                        status = PythonCodeExecutionStatus.TIMEOUT
                        _terminate_process(process)
                        break

                    time.sleep(_POLL_INTERVAL_SECONDS)

        stdout_text, output_truncated = _read_output_preview(output_path, request.max_output_bytes)
        if output_truncated and status == PythonCodeExecutionStatus.OK:
            status = PythonCodeExecutionStatus.OUTPUT_TOO_LONG

        return PythonCodeExecutionResult(
            status=status,
            stdout=stdout_text,
            output_truncated=output_truncated or status == PythonCodeExecutionStatus.OUTPUT_TOO_LONG,
            exit_code=process.returncode,
        )
    except Exception as err:
        if process is not None:
            try:
                _terminate_process(process)
            except Exception:
                pass
        return PythonCodeExecutionResult(
            status=PythonCodeExecutionStatus.INTERNAL_ERROR,
            stdout="",
            output_truncated=False,
            failure_reason=f"Python runner error: {err}",
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _normalize_code(code: str) -> str:
    return code.replace("```python", "").replace("```", "").strip()


def _build_python_subprocess_script(code: str, prelude: str) -> str:
    normalized_prelude = prelude if prelude.endswith("\n") else f"{prelude}\n"
    return f"{normalized_prelude}\n{code}\n"


def _run_python_syntax_check(code_path: str) -> Optional[PythonCodeExecutionResult]:
    try:
        syntax_process = subprocess.run(
            [sys.executable, "-m", "py_compile", code_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except Exception as err:
        return PythonCodeExecutionResult(
            status=PythonCodeExecutionStatus.INTERNAL_ERROR,
            stdout="",
            output_truncated=False,
            failure_reason=f"Python syntax check error: {err}",
        )

    if syntax_process.returncode == 0:
        return None

    stdout_text = syntax_process.stdout.decode("utf-8", errors="ignore").strip()
    return PythonCodeExecutionResult(
        status=PythonCodeExecutionStatus.SYNTAX_ERROR,
        stdout=stdout_text,
        output_truncated=False,
        exit_code=syntax_process.returncode,
        failure_reason="Python syntax check failed",
    )


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