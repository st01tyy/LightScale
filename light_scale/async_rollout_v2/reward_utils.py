from typing import List, Optional, Sequence

import numpy as np


def compute_normed_rewards(
	rewards: Sequence[Optional[float]],
	valid_flags: Sequence[bool],
	advantage_estimator: str,
) -> List[float]:
	"""根据有效样本奖励计算归一化奖励，其他位置填充 0."""
	if len(rewards) != len(valid_flags):
		raise ValueError("rewards 与 valid_flags 长度不一致")
	if not rewards:
		return []
	valid_indices = [idx for idx, flag in enumerate(valid_flags) if flag]
	normed_rewards = [0.0] * len(rewards)
	if not valid_indices:
		return normed_rewards
	valid_values = []
	for idx in valid_indices:
		reward = rewards[idx]
		if reward is None:
			raise ValueError("valid_flags 为 True 的样本 reward 不应为 None")
		valid_values.append(reward)
	rewards_array = np.array(valid_values, dtype=np.float32)
	estimator = advantage_estimator.lower()
	if estimator == "grpo":
		normed = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-9)
	elif estimator == "rloo" and len(rewards_array) > 1:
		baseline = (rewards_array.sum() - rewards_array) / (len(rewards_array) - 1) + 1e-9
		normed = rewards_array - baseline
	elif estimator == "rloo" and len(rewards_array) == 1:
		normed = rewards_array - rewards_array.mean()
	elif estimator in ("reinforce++", "dapo"):
		normed = rewards_array - rewards_array.mean()
	else:
		raise ValueError(f"未知的 advantage_estimator: {advantage_estimator}")
	for idx, value in zip(valid_indices, normed):
		normed_rewards[idx] = float(value)
	return normed_rewards
