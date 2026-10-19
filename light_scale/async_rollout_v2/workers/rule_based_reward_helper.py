"""Shared rule-based reward helpers for async rollout workers."""

import asyncio
from typing import Dict, Tuple

from light_scale.async_rollout_v2.executors import get_process_pool
from verifier.rule_based_rm import compute_score as rule_based_score
from verifier.rule_based_rm_cot import compute_score as rule_based_score_cot


async def compute_rule_based_reward(
	*,
	dataset_type: str,
	response: str,
	ground_truth: str,
	prompt: str,
	force_thinking: bool,
	begin_of_thinking: str,
	use_cot_reward: bool,
) -> Tuple[float, Dict[str, float]]:
	"""Run the existing rule-based verifier in the shared process pool."""
	pool = get_process_pool()
	loop = asyncio.get_running_loop()
	actual_response = response
	if force_thinking:
		actual_response = begin_of_thinking + actual_response
	if not use_cot_reward:
		return await loop.run_in_executor(
			pool,
			rule_based_score,
			dataset_type,
			actual_response,
			ground_truth,
			prompt,
		)
	return await loop.run_in_executor(
		pool,
		rule_based_score_cot,
		dataset_type,
		actual_response,
		ground_truth,
		prompt,
	)