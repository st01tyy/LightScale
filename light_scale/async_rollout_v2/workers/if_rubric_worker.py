"""Async IF rubric worker for async rollout v2."""

import asyncio
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from light_scale.data import MultiResponseSample
from light_scale.async_rollout_v2.services.sglang_service import SGLangChatCompletionTask
from light_scale.async_rollout_v2.utils.llm_judge import LLMJudgeConfig
from light_scale.async_rollout_v2.workers.llm_judge_worker import (
    AsyncLLMJudgeWorker,
    AsyncLLMJudgeWorkerConfig,
)


_RUBRIC_RESULT_PATTERN = re.compile(r"因此\s*(满足|不满足)")


@dataclass
class AsyncIFRubricWorkerConfig(AsyncLLMJudgeWorkerConfig):
    """IF rubric worker configuration."""


class AsyncIFRubricWorker(AsyncLLMJudgeWorker):
    """Judge instruction-following responses with rubric prompts via the judge model."""

    CONFIG_CLS = AsyncIFRubricWorkerConfig

    async def _score_responses(self, sample: MultiResponseSample, raw_sample: dict) -> None:
        responses = sample.responses or []
        if not responses:
            sample.rewards = []
            sample.reward_metrics_list = []
            sample.avg_reward_metrics = []
            return

        rubric = str(sample.ground_truth or "")
        processed_responses = [self._maybe_extract_response(response) for response in responses]
        rewards, reward_metrics_list = await self._judge_rubric_responses(
            sample=sample,
            rubric=rubric,
            responses=processed_responses,
        )

        if self._config.use_ref_answers and rewards and all(reward == 0.0 for reward in rewards):
            ref_answers = raw_sample.get("ref_answers", [])
            num_ref_answers = min(len(ref_answers), self._config.max_num_ref_answers)
            num_ref_answers = min(num_ref_answers, self._config.n_samples - 1)
            num_ref_answers = min(num_ref_answers, len(responses))
            if num_ref_answers > 0:
                responses[:num_ref_answers] = ref_answers[:num_ref_answers]
                sample.responses = responses
                self._sync_group_messages_from_responses(sample)
                for idx in range(num_ref_answers):
                    rewards[idx] = 1.0
                    reward_metrics_list[idx] = {"verdict": 1, "judge_ok": 1, "format": 1}
                self.logger.debug("using %s ref answers", num_ref_answers)

        sample.rewards = rewards
        sample.reward_metrics_list = reward_metrics_list
        sample.avg_reward_metrics = ["verdict", "judge_ok", "format"]

    async def _judge_rubric_responses(
        self,
        *,
        sample: MultiResponseSample,
        rubric: str,
        responses: List[str],
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        if not responses:
            return [], []

        judge_config = LLMJudgeConfig.from_object(self._config, fallback_max_tokens=self._config.max_tokens)
        parallelism = judge_config.judge_parallelism
        max_concurrency = len(responses) if parallelism <= 0 else min(parallelism, len(responses))
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _run_with_semaphore(response: str) -> Tuple[float, Dict[str, int]]:
            async with semaphore:
                return await self._judge_single_rubric_response(
                    sample=sample,
                    response=response,
                    rubric=rubric,
                    judge_config=judge_config,
                )

        results = await asyncio.gather(
            *[_run_with_semaphore(response) for response in responses],
            return_exceptions=True,
        )

        rewards: List[float] = []
        metrics_list: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning("if_rubric judge failed: %s", result)
                rewards.append(0.0)
                metrics_list.append({"judge_ok": 0, "format": 0, "verdict": 0})
                continue

            reward, metrics = result
            rewards.append(reward)
            metrics_list.append(metrics)
        return rewards, metrics_list

    async def _judge_single_rubric_response(
        self,
        *,
        sample: MultiResponseSample,
        response: str,
        rubric: str,
        judge_config: LLMJudgeConfig,
    ) -> Tuple[float, Dict[str, int]]:
        text = (response or "").strip()
        if not text or text == "no response":
            return 0.0, {"verdict": 0, "judge_ok": 1, "format": 0}

        try:
            grader_prompt = rubric.format(text)
        except Exception as err:
            self.logger.warning("if_rubric prompt format failed: %s", err)
            return 0.0, {"verdict": 0, "judge_ok": 0, "format": 0}

        task = SGLangChatCompletionTask(
            prompt="",
            n_samples=1,
            max_tokens=max(4096, judge_config.judge_max_tokens),
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            retry=judge_config.judge_retry,
            timeout=judge_config.judge_timeout,
            messages=[{"role": "user", "content": grader_prompt}],
        )
        try:
            result = await self._judge_service.submit(task)
        except Exception as err:
            self.logger.warning("if_rubric judge service failed: %s", err)
            return 0.0, {"verdict": 0, "judge_ok": 0, "format": 0}

        raw_text = result.responses[0][0] if result.responses else ""
        verdict = self._postprocess_rubric_judgment(raw_text)
        if verdict is None:
            return 0.0, {"verdict": 0, "judge_ok": 0, "format": 0}
        return (1.0 if verdict else 0.0), {"verdict": int(verdict), "judge_ok": 1, "format": 1}

    def _postprocess_rubric_judgment(self, text: str) -> Optional[bool]:
        results = _RUBRIC_RESULT_PATTERN.findall(text or "")
        if not results:
            return None
        return all(result == "满足" for result in results)