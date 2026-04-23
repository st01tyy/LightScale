"""Async math worker for async rollout v2."""

import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple

from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.executors import get_process_pool
from light_scale.async_rollout_v2.workers.base_worker import (
	AsyncSingleTurnWorker,
	AsyncSingleTurnWorkerConfig,
)
from light_scale.data import MultiResponseSample
from verifier.rule_based_rm import compute_score as rule_based_score
from verifier.rule_based_rm_cot import compute_score as rule_based_score_cot
from copy import deepcopy
import re

async def reward_fn(completion, ground_truth):
    """
    极简 0/1 奖励函数：只有结构完全正确且答案一致才给 1.0
    """
    try:
        # 1. 校验有且仅有一对 think 标签，且 think 在前
        if completion.count("<think>") != 1 or completion.count("</think>") != 1:
            # print("think count invalid", flush=True)
            return 0.0
        
        # 2. 分割内容
        parts = completion.split("<think>")[1].split("</think>")
        think_content = parts[0].strip()
        after_content = parts[1].strip()
        # print(after_content, flush=True)

        # 3. 校验 think 内部和外部均不能为空
        if not think_content or not after_content:
            return 0.0

        # 4. 提取 \boxed{...} 里的内容并比对
        # 使用 (.+?) 非贪婪匹配，或者 (.+) 贪婪匹配最后一个 }
        match = re.search(r"\\boxed\{(.+)\}", after_content)

        # print(f"{match}: {ground_truth}", flush=True)
        
        if match and match.group(1).strip() == str(ground_truth).strip():
            return 1.0
            
    except Exception as e:
        # print(str(e), flush=True)
        pass
        
    return 0.0


@dataclass
class AsyncMathWorkerConfig(AsyncSingleTurnWorkerConfig):
	"""数学 worker 专用配置，目前直接复用父类字段，便于后续扩展。"""

	use_cot_reward: bool = False
	use_ref_answers: bool = False
	max_num_ref_answers: int = 1


class AsyncMathWorker(AsyncSingleTurnWorker):
	"""处理单条数学样本的 worker，负责调用 SGLang 并用 rule-based RM 打分。"""

	CONFIG_CLS = AsyncMathWorkerConfig

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

	async def _score_responses(self, sample: MultiResponseSample, raw_sample: dict) -> None:
		"""遍历模型响应，调用数学评分函数并写回奖励字段。"""
		# rewards = []
		# for response in sample.responses:
		# 	if self._config.force_thinking:
		# 		response = self._config.begin_of_thinking + response
		# 	rewards.append(await reward_fn(response, sample.ground_truth))
		# sample.rewards = deepcopy(rewards)
		# sample.reward_metrics_list = [dict() for _ in range(len(sample.responses))]
		# sample.avg_reward_metrics = []
		responses = sample.responses or []
		rewards = [None] * len(responses)
		reward_metrics_list = [None for _ in responses]
		# if not responses:
		# 	sample.rewards = rewards
		# 	sample.reward_metrics_list = reward_metrics_list
		# 	return

		scoring_again_flag = False
		while True:
			tasks = []
			index_map = []
			for idx, response in enumerate(responses):
				# if response is None:
				# 	continue
				index_map.append(idx)
				tasks.append(
					self._compute_reward(
						sample.dataset_type,
						response,
						sample.ground_truth,
						sample.prompt,
					)
				)

			if tasks:
				results = await asyncio.gather(*tasks)
				if not scoring_again_flag and self._config.use_ref_answers and sum([r[0] for r in results]) == 0.0:
					ref_answers = raw_sample.get("ref_answers", [])
					num_ref_answers = min(len(ref_answers), self._config.max_num_ref_answers)
					num_ref_answers = min(num_ref_answers, self._config.n_samples - 1)
					if num_ref_answers > 0:
						responses[:num_ref_answers] = ref_answers[:num_ref_answers]
						sample.responses = responses
						self._sync_group_messages_from_responses(sample)
						scoring_again_flag = True
						self.logger.debug(f"using {num_ref_answers} ref answers")
				else:
					for idx, (reward, reward_metrics) in zip(index_map, results):
						rewards[idx] = reward
						reward_metrics_list[idx] = reward_metrics
					scoring_again_flag = False
			if not scoring_again_flag:
				break
		sample.rewards = deepcopy(rewards)
		sample.reward_metrics_list = deepcopy(reward_metrics_list)
		sample.avg_reward_metrics = ["format", "correctness", "language"]

		# for response in responses:

	async def _compute_reward(
		self,
		dataset_type: str,
		response: str,
		ground_truth: str,
		prompt: str,
	) -> Tuple[float, dict]:
		"""根据是否跳过思考内容选择不同的 rule-based RM。"""
		pool = get_process_pool()
		loop = asyncio.get_running_loop()
		if self._config.force_thinking:
			response = self._config.begin_of_thinking + response
		if not self._config.use_cot_reward:
			return await loop.run_in_executor(
				pool, rule_based_score, dataset_type, response, ground_truth, prompt
			)
		return await loop.run_in_executor(
			pool, rule_based_score_cot, dataset_type, response, ground_truth, prompt
		)
