"""Async mixed worker combining rule-based scoring and LLM judge scoring."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from light_scale.async_rollout_v2.utils.llm_judge import LLMJudgeConfig, judge_responses
from light_scale.async_rollout_v2.workers.llm_judge_worker import (
	AsyncLLMJudgeWorker,
	AsyncLLMJudgeWorkerConfig,
)
from light_scale.async_rollout_v2.workers.rule_based_reward_helper import compute_rule_based_reward
from light_scale.data import MultiResponseSample


@dataclass
class AsyncRuleLLMMixedWorkerConfig(AsyncLLMJudgeWorkerConfig):
	"""Rule + LLM mixed worker configuration."""


class AsyncRuleLLMMixedWorker(AsyncLLMJudgeWorker):
	"""Use both the rule-based verifier and LLM judge, then AND their verdicts."""

	CONFIG_CLS = AsyncRuleLLMMixedWorkerConfig

	async def _score_responses(self, sample: MultiResponseSample, raw_sample: dict) -> None:
		responses = sample.responses or []
		if not responses:
			sample.rewards = []
			sample.reward_metrics_list = []
			sample.avg_reward_metrics = []
			return

		processed_responses = [self._maybe_extract_response(response) for response in responses]
		rule_task = self._score_rule_responses(sample=sample, responses=responses)
		judge_task = judge_responses(
			judge_service=self._judge_service,
			sample=sample,
			responses=processed_responses,
			judge_config=LLMJudgeConfig.from_object(self._config, fallback_max_tokens=self._config.max_tokens),
			logger=self.logger,
		)
		(rule_rewards, rule_metrics_list), (judge_rewards, judge_metrics_list) = await asyncio.gather(
			rule_task,
			judge_task,
		)

		final_rewards: List[float] = []
		reward_metrics_list: List[Dict[str, Any]] = []
		for rule_reward, rule_metrics, judge_reward, judge_metrics in zip(
			rule_rewards,
			rule_metrics_list,
			judge_rewards,
			judge_metrics_list,
		):
			rule_metrics = rule_metrics or {}
			judge_metrics = judge_metrics or {}
			judge_ok = int(judge_metrics.get("judge_ok", 0))
			rule_pass = float(rule_reward or 0.0) > 0.5
			judge_pass = float(judge_reward or 0.0) > 0.5
			mixed_pass = judge_ok == 1 and rule_pass and judge_pass
			final_rewards.append(1.0 if mixed_pass else 0.0)
			reward_metrics_list.append(
				{
					"rule_reward": float(rule_reward or 0.0),
					"rule_format": float(rule_metrics.get("format", 0.0)),
					"rule_correctness": float(rule_metrics.get("correctness", 0.0)),
					"rule_language": float(rule_metrics.get("language", 0.0)),
					"judge_reward": float(judge_reward or 0.0),
					"judge_verdict": int(judge_metrics.get("verdict", 0)),
					"judge_ok": judge_ok,
					"judge_format": int(judge_metrics.get("format", 0)),
					"mixed_verdict": int(mixed_pass),
				}
			)

		sample.rewards = final_rewards
		sample.reward_metrics_list = reward_metrics_list
		sample.avg_reward_metrics = [
			"mixed_verdict",
			"rule_reward",
			"rule_format",
			"rule_correctness",
			"rule_language",
			"judge_reward",
			"judge_verdict",
			"judge_ok",
			"judge_format",
		]

	async def _score_rule_responses(
		self,
		*,
		sample: MultiResponseSample,
		responses: List[str],
	) -> Tuple[List[float], List[Dict[str, float]]]:
		results = await asyncio.gather(
			*[
				compute_rule_based_reward(
					dataset_type=sample.dataset_type,
					response=response,
					ground_truth=sample.ground_truth,
					prompt=sample.prompt,
					force_thinking=self._config.force_thinking,
					begin_of_thinking=self._config.begin_of_thinking,
					use_cot_reward=self._config.use_cot_reward,
				)
				for response in responses
			]
		)
		rule_rewards: List[float] = []
		rule_metrics_list: List[Dict[str, float]] = []
		for reward, metrics in results:
			rule_rewards.append(reward)
			rule_metrics_list.append(metrics or {})
		return rule_rewards, rule_metrics_list