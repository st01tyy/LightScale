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
    judge_mode: str = "ground_truth"
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
            judge_mode=str(getattr(source, "judge_mode", default.judge_mode)),
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


def build_ground_truth_judge_messages(sample: MultiResponseSample, response: str) -> List[Dict[str, str]]:
    question = sample.problem if sample.problem is not None else sample.prompt
    system_prompt = """
You are a rigorous and objective Evaluation Agent. Your sole task is to determine if a Candidate Answer matches the Ground Truth for a given Question.

# Task Requirements
1. Direct Comparison: Compare the Candidate Answer against the Ground Truth. Focus on core facts, numerical accuracy, and logical equivalence.
2. Ignore Noise: Disregard tone, wordiness, or formatting differences. If the core meaning is the same (e.g., 0.5 vs 1/2), mark it as true.
3. No Self-Solving: Do not attempt to solve the question yourself. Use only the provided Ground Truth as the absolute reference.
4. Constraint on Hint: If the answer is incorrect, provide a hint in the second person (You). Point out the nature of the error but NEVER reveal the correct answer.

# Output Format
You must output a valid JSON object. The JSON must strictly comply with the following structural rules:
1. **Strict Key Order and Presence**: The JSON must contain exactly three keys, and they must appear in this precise order: `reasoning`, `verdict`, `hint`. No additional keys are allowed.
2. **Flat Structure**: All three keys must reside at the top-level (the first hierarchy) of the JSON object. No nested objects or arrays are permitted.
3. **Raw Output**: Return only the valid JSON object. Do not wrap it in conversational text.

## Field Specifications:
- `reasoning` (string): A step-by-step checklist and explanation of which constraints were met and which were violated.
- `verdict` (boolean): Set to `true` if and only if all constraints are fully satisfied. Otherwise, set to `false`.
- `hint` (string): If `verdict` is `false`, provide a concise, second-person message explaining the mistake (e.g., "You did not write the response in a table format"). If `verdict` is `true`, this field must be an empty string `""`.

## Example Output Template
```json
{
  "reasoning": "reasoning content here",
  "verdict": true or false,
  "hint": "hint content here"
}
```
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


def build_direct_judge_messages(sample: MultiResponseSample, response: str) -> List[Dict[str, str]]:
    question = sample.problem if sample.problem is not None else sample.prompt
    system_prompt = """
You are a rigorous and objective Evaluation Agent. Your task is to judge whether a Candidate Answer correctly answers a given Question, using the Question itself as the basis for evaluation.

# Task Requirements
1. Solve-as-needed: You may reason about the Question to determine whether the Candidate Answer is correct.
2. Correctness First: Focus on factual, mathematical, and logical correctness. Ignore stylistic issues unless they make the answer incorrect or non-responsive.
3. Instruction Following Matters: Treat explicit requirements in the Question as part of correctness. This includes requested answer format, output schema, units, rounding rules, response length limits, whether explanation is allowed or forbidden, whether only a final answer should be returned, and any other response constraints.
4. Format Violations Count as Incorrect: If the Candidate Answer is factually right but violates a required output format or instruction-following constraint in the Question, mark it as false.
5. Conservative Judgement: If the Candidate Answer is ambiguous, incomplete, self-contradictory, or you cannot reliably verify it from the Question, mark it as false.
6. Constraint on Hint: If the answer is incorrect, provide a hint in the second person (You). Point out the nature of the error but NEVER reveal the final correct answer directly.

# Output Format
You must output a valid JSON object. The JSON must strictly comply with the following structural rules:
1. **Strict Key Order and Presence**: The JSON must contain exactly three keys, and they must appear in this precise order: `reasoning`, `verdict`, `hint`. No additional keys are allowed.
2. **Flat Structure**: All three keys must reside at the top-level (the first hierarchy) of the JSON object. No nested objects or arrays are permitted.
3. **Raw Output**: Return only the valid JSON object. Do not wrap it in conversational text.

## Field Specifications:
- `reasoning` (string): A step-by-step checklist and explanation of which constraints were met and which were violated.
- `verdict` (boolean): Set to `true` if and only if all constraints are fully satisfied. Otherwise, set to `false`.
- `hint` (string): If `verdict` is `false`, provide a concise, second-person message explaining the mistake (e.g., "You did not write the response in a table format"). If `verdict` is `true`, this field must be an empty string `""`.

## Example Output Template
```json
{
  "reasoning": "reasoning content here",
  "verdict": true or false,
  "hint": "hint content here"
}
```
"""
    user_prompt = f"""
Please evaluate the following:

Question:
{question}

Candidate Answer:
{response}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def build_instruction_following_judge_messages(sample: MultiResponseSample, response: str) -> List[Dict[str, str]]:
    question = sample.problem if sample.problem is not None else sample.prompt
    system_prompt = """
You are a rigorous and objective Evaluation Agent. Your task is to judge whether a Candidate Answer strictly satisfies all instructions, constraints, and formatting requirements specified in a given Question, without evaluating the factual accuracy or truthfulness of the content itself.

# Task Requirements
1. **Focus on Constraints, Ignore Correctness**: Do NOT evaluate whether the content of the Candidate Answer is factually correct, scientifically true, or logically sound. Your sole focus is on whether the Candidate Answer follows the rules set by the Question. (e.g., If the user asks for an article about "the moon being made of green cheese" containing the word "apple", you only verify if it is an article and if it contains "apple"; do not judge the scientific accuracy of the moon's composition.)
2. **Comprehensive Instruction Checking**: Identify all explicit and implicit requirements in the Question. This includes:
   - Formatting & Output schema (e.g., JSON, Markdown table, raw text).
   - Word/Length limits (e.g., "under 100 words", "exactly 3 sentences").
   - Required elements (e.g., specific keywords, certain structure, specific units).
   - Negative constraints (e.g., "do not use the word X", "do not explain", "no introductory text").
3. **Zero Tolerance for Violations**: The Candidate Answer must satisfy 100% of the constraints to receive a "true" verdict. If even a single minor constraint is violated (e.g., missing one keyword, exceeding the word limit by one word, or failing to use the requested format), mark the verdict as false.
4. **Conservative Judgement**: If you cannot reliably verify whether a constraint is met, or if the Candidate Answer is ambiguous regarding a constraint, mark the verdict as false.
5. **Constraint on Hint**: If the verdict is false, provide a concise hint in the second person (You). Point out the specific constraint(s) that were violated (e.g., "You did not include the keyword 'X'"), but DO NOT generate or suggest the corrected content.

# Output Format
You must output a valid JSON object. The JSON must strictly comply with the following structural rules:
1. **Strict Key Order and Presence**: The JSON must contain exactly three keys, and they must appear in this precise order: `reasoning`, `verdict`, `hint`. No additional keys are allowed.
2. **Flat Structure**: All three keys must reside at the top-level (the first hierarchy) of the JSON object. No nested objects or arrays are permitted.
3. **Raw Output**: Return only the valid JSON object. Do not wrap it in conversational text.

## Field Specifications:
- `reasoning` (string): A step-by-step checklist and explanation of which constraints were met and which were violated.
- `verdict` (boolean): Set to `true` if and only if all constraints are fully satisfied. Otherwise, set to `false`.
- `hint` (string): If `verdict` is `false`, provide a concise, second-person message explaining the mistake (e.g., "You did not write the response in a table format"). If `verdict` is `true`, this field must be an empty string `""`.

## Example Output Template
```json
{
  "reasoning": "reasoning content here",
  "verdict": true or false,
  "hint": "hint content here"
}
```
"""
    user_prompt = f"""
Please evaluate the following:

Question:
{question}

Candidate Answer:
{response}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_judge_messages(
    sample: MultiResponseSample,
    response: str,
    judge_mode: str = "ground_truth",
) -> List[Dict[str, str]]:
    mode = (judge_mode or "ground_truth").strip().lower()
    if mode == "ground_truth":
        return build_ground_truth_judge_messages(sample, response)
    if mode == "direct":
        return build_direct_judge_messages(sample, response)
    if mode == "instruction_following":
        return build_instruction_following_judge_messages(sample, response)
    raise ValueError(f"unknown judge_mode: {judge_mode}")


def parse_judge_json(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("judge response is empty")
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        payload = json.loads(fence_match.group(1).strip())
        if isinstance(payload, dict):
            return payload
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

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

    messages = build_judge_messages(sample, response, judge_config.judge_mode)
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
