"""Shared LLM judge helpers for async rollout workers."""

import asyncio
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from light_scale.async_rollout_v2.services.sglang_service import (
    AsyncSGLangService,
    SGLangChatCompletionTask,
)
from light_scale.data import MultiResponseSample


@dataclass
class LLMJudgeConfig:
    judge_timeout: float = 600.0
    judge_retry: int = 2
    judge_json_retries: int = 2
    judge_temperature: float = 0.2
    judge_parallelism: int = 0
    judge_max_tokens: int = 1024

    @classmethod
    def from_object(cls, source: Any, fallback_max_tokens: Optional[int] = None) -> "LLMJudgeConfig":
        default = cls()
        judge_max_tokens = getattr(source, "judge_max_tokens", None)
        if judge_max_tokens is None:
            judge_max_tokens = fallback_max_tokens if fallback_max_tokens is not None else default.judge_max_tokens
        return cls(
            judge_timeout=float(getattr(source, "judge_timeout", default.judge_timeout)),
            judge_retry=int(getattr(source, "judge_retry", default.judge_retry)),
            judge_json_retries=int(getattr(source, "judge_json_retries", default.judge_json_retries)),
            judge_temperature=float(getattr(source, "judge_temperature", default.judge_temperature)),
            judge_parallelism=int(getattr(source, "judge_parallelism", default.judge_parallelism)),
            judge_max_tokens=int(judge_max_tokens),
        )


def maybe_extract_cot_response(
    response: Optional[str],
    *,
    use_cot_reward: bool,
    end_of_thinking: Optional[str],
) -> str:
    if response is None:
        return ""
    if not use_cot_reward:
        return response

    text = response.strip()
    if not text:
        return "no response"

    if end_of_thinking and end_of_thinking in text:
        extracted = text.split(end_of_thinking, 1)[1].strip()
        return extracted if extracted else "no response"
    if "</think>" in text:
        extracted = text.split("</think>", 1)[1].strip()
        return extracted if extracted else "no response"
    return "no response"


def build_judge_messages(sample: MultiResponseSample, response: str) -> List[Dict[str, str]]:
    question = sample.problem if sample.problem is not None else sample.prompt
    system_prompt = """
You are a rigorous and objective Evaluation Agent. Your sole task is to determine if a Candidate Answer matches the Ground Truth for a given Question.

# Task Requirements
1. Direct Comparison: Compare the Candidate Answer against the Ground Truth. Focus on core facts, numerical accuracy, and logical equivalence.
2. Ignore Noise: Disregard tone, wordiness, or formatting differences. If the core meaning is the same (e.g., 0.5 vs 1/2), mark it as true.
3. No Self-Solving: Do not attempt to solve the question yourself. Use only the provided Ground Truth as the absolute reference.
4. Constraint on Hint: If the answer is incorrect, provide a hint in the second person (You). Point out the nature of the error but NEVER reveal the correct answer.

# Output Format
You must output a valid JSON object with the following keys:
- reasoning: A brief internal explanation of why the answers match or differ.
- verdict: boolean (true/false).
- hint: A concise, second-person message to the candidate explaining their mistake without revealing the correct answer.
"""
    user_prompt = f"""
Please evaluate the following:

Question:
{question}

Ground Truth:
{sample.ground_truth}

Candidate Answer:
{response}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_json(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("judge response is empty")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        payload = json.loads(fence_match.group(1).strip())
        if isinstance(payload, dict):
            return payload

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        payload = json.loads(text[start : end + 1])
        if isinstance(payload, dict):
            return payload
    raise ValueError("judge response is not a valid JSON object")


def to_verdict(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


async def judge_single_response(
    *,
    judge_service: AsyncSGLangService,
    sample: MultiResponseSample,
    response: str,
    judge_config: LLMJudgeConfig,
    logger,
) -> Tuple[float, Dict[str, int]]:
    text = (response or "").strip()
    if not text or text == "no response":
        return 0.0, {"verdict": 0, "judge_ok": 1, "format": 0}

    messages = build_judge_messages(sample, response)
    retry_count = 0
    max_json_retries = max(0, judge_config.judge_json_retries)

    while retry_count <= max_json_retries:
        task = SGLangChatCompletionTask(
            prompt="",
            n_samples=1,
            max_tokens=judge_config.judge_max_tokens,
            temperature=judge_config.judge_temperature,
            retry=judge_config.judge_retry,
            timeout=judge_config.judge_timeout,
            messages=messages,
            response_format={"type": "json_object"},
        )
        try:
            result = await judge_service.submit(task)
        except Exception as err:
            logger.warning("judge service failed: %s", err)
            return 0.0, {"verdict": 0, "judge_ok": 0, "format": 1}

        raw_text = result.responses[0][0] if result.responses else ""
        try:
            payload = parse_judge_json(raw_text)
        except Exception as err:
            retry_count += 1
            if retry_count <= max_json_retries:
                logger.warning("judge JSON parse failed, retry %s/%s: %s", retry_count, max_json_retries, err)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not parseable JSON. "
                            "Return ONLY one valid JSON object with keys: verdict, reasoning, hint. "
                            "No markdown fences and no extra text."
                        ),
                    }
                )
                continue
            return 0.0, {"verdict": 0, "judge_ok": 0, "format": 1}

        verdict = to_verdict(payload.get("verdict"))
        if verdict is None:
            retry_count += 1
            if retry_count <= max_json_retries:
                logger.warning("judge verdict invalid, retry %s/%s: payload=%s", retry_count, max_json_retries, payload)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON has invalid verdict. "
                            "The verdict field must be a boolean true/false. "
                            "Return ONLY one valid JSON object with keys: verdict, reasoning, hint. "
                            "No markdown fences and no extra text."
                        ),
                    }
                )
                continue
            return 0.0, {"verdict": 0, "judge_ok": 0, "format": 1}

        return (1.0 if verdict else 0.0), {"verdict": int(verdict), "judge_ok": 1, "format": 1}

    return 0.0, {"verdict": 0, "judge_ok": 0, "format": 1}


async def judge_responses(
    *,
    judge_service: AsyncSGLangService,
    sample: MultiResponseSample,
    responses: Sequence[str],
    judge_config: LLMJudgeConfig,
    logger,
    base_metrics_list: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    if not responses:
        return [], []

    parallelism = judge_config.judge_parallelism
    max_concurrency = len(responses) if parallelism <= 0 else min(parallelism, len(responses))
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_with_semaphore(response: str) -> Tuple[float, Dict[str, int]]:
        async with semaphore:
            return await judge_single_response(
                judge_service=judge_service,
                sample=sample,
                response=response,
                judge_config=judge_config,
                logger=logger,
            )

    results = await asyncio.gather(
        *[_run_with_semaphore(response) for response in responses],
        return_exceptions=True,
    )

    rewards: List[float] = []
    metrics_list: List[Dict[str, Any]] = []
    base_metrics_list = base_metrics_list or [dict() for _ in responses]
    for idx, result in enumerate(results):
        base_metrics = deepcopy(base_metrics_list[idx]) if idx < len(base_metrics_list) else {}
        if isinstance(result, Exception):
            logger.warning("judge failed: %s", result)
            base_metrics.update({"judge_ok": 0, "format": 0, "verdict": 0})
            rewards.append(0.0)
            metrics_list.append(base_metrics)
            continue

        reward, judge_metrics = result
        base_metrics.update(judge_metrics)
        rewards.append(reward)
        metrics_list.append(base_metrics)
    return rewards, metrics_list
