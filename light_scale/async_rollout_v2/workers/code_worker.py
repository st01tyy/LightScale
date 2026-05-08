"""Async code worker for async rollout v2."""

import asyncio
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from light_scale.async_rollout_v2.executors import get_thread_pool
from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.utils.python_code import (
    PythonCodeExecutionRequest,
    PythonCodeExecutionStatus,
    execute_python_code,
)
from light_scale.async_rollout_v2.workers.base_worker import (
    AsyncSingleTurnWorker,
    AsyncSingleTurnWorkerConfig,
)
from light_scale.data import MultiResponseSample


_PYTHON_CODE_BLOCK_PATTERN = re.compile(r"```python\s*\n(.*?)```", flags=re.IGNORECASE | re.DOTALL)


@dataclass
class AsyncCodeWorkerConfig(AsyncSingleTurnWorkerConfig):
    """Code worker configuration."""

    python_timeout_seconds: int = 10
    max_code_output_bytes: int = 65536


class AsyncCodeWorker(AsyncSingleTurnWorker):
    """Score code responses by executing extracted Python against local test cases."""

    CONFIG_CLS = AsyncCodeWorkerConfig

    async def _score_responses(self, sample: MultiResponseSample, raw_sample: dict) -> None:
        sample.ground_truth = ""
        responses = sample.responses or []
        if not responses:
            sample.rewards = []
            sample.reward_metrics_list = []
            sample.avg_reward_metrics = []
            return

        judge_info, judge_error = self._load_judge_info(raw_sample)
        if judge_error is not None:
            metrics = [self._build_invalid_judge_metrics(judge_error) for _ in responses]
            sample.rewards = [0.0 for _ in responses]
            sample.reward_metrics_list = metrics
            sample.avg_reward_metrics = ["format", "correctness", "executor_ok", "pass_rate"]
            return

        test_cases = judge_info["test_cases"]
        results = await asyncio.gather(
            *(self._score_single_response(response, test_cases) for response in responses)
        )
        sample.rewards = [reward for reward, _ in results]
        sample.reward_metrics_list = [metrics for _, metrics in results]
        sample.avg_reward_metrics = ["format", "correctness", "executor_ok", "pass_rate"]

    def _load_judge_info(self, raw_sample: dict) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        raw_ground_truth = raw_sample.get("ground_truth", "")
        try:
            judge_info = json.loads(raw_ground_truth)
        except Exception as err:
            return None, f"invalid_judge_info_json: {err}"

        if not isinstance(judge_info, dict):
            return None, "invalid_judge_info_type"
        test_cases = judge_info.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            return None, "invalid_test_cases"
        for idx, test_case in enumerate(test_cases):
            if not isinstance(test_case, dict):
                return None, f"invalid_test_case_{idx}_type"
            if "input" not in test_case or "output" not in test_case:
                return None, f"invalid_test_case_{idx}_schema"
        return judge_info, None

    async def _score_single_response(
        self,
        response: Optional[str],
        test_cases: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        code = extract_first_python_code_block(response)
        if code is None:
            return 0.0, {
                "format": 0,
                "correctness": 0,
                "executor_ok": 0,
                "pass_rate": 0.0,
                "execution_status": "missing_python_code_block",
            }

        passed_cases = 0
        total_cases = len(test_cases)
        last_status = PythonCodeExecutionStatus.OK.value
        executor_ok = 1
        for test_case in test_cases:
            execution_result = await self._execute_code_with_test_case(
                code=code,
                stdin_text="" if test_case.get("input") is None else str(test_case.get("input")),
            )
            last_status = execution_result.status.value
            if execution_result.status != PythonCodeExecutionStatus.OK:
                executor_ok = 0
                return 0.0, {
                    "format": 1,
                    "correctness": 0,
                    "executor_ok": executor_ok,
                    "pass_rate": passed_cases / total_cases,
                    "execution_status": last_status,
                }

            expected_output = "" if test_case.get("output") is None else str(test_case.get("output"))
            if execution_result.stdout.rstrip() != expected_output.rstrip():
                return 0.0, {
                    "format": 1,
                    "correctness": 0,
                    "executor_ok": executor_ok,
                    "pass_rate": passed_cases / total_cases,
                    "execution_status": "wrong_answer",
                }

            passed_cases += 1

        return 1.0, {
            "format": 1,
            "correctness": 1,
            "executor_ok": executor_ok,
            "pass_rate": 1.0,
            "execution_status": last_status,
        }

    async def _execute_code_with_test_case(self, *, code: str, stdin_text: str):
        loop = asyncio.get_running_loop()
        pool = get_thread_pool()
        return await loop.run_in_executor(
            pool,
            execute_python_code,
            PythonCodeExecutionRequest(
                code=code,
                stdin_text=stdin_text,
                prelude="",
                timeout_seconds=self._config.python_timeout_seconds,
                max_output_bytes=self._config.max_code_output_bytes,
            ),
        )

    def _build_invalid_judge_metrics(self, reason: str) -> Dict[str, Any]:
        return {
            "format": 0,
            "correctness": 0,
            "executor_ok": 0,
            "pass_rate": 0.0,
            "execution_status": reason,
        }


def extract_first_python_code_block(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None
    match = _PYTHON_CODE_BLOCK_PATTERN.search(response)
    if match is None:
        return None
    code = match.group(1).strip()
    return code or None