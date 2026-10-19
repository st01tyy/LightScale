"""Overlap metric helpers for async rollout v2."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from light_scale.async_rollout_v2.services.sglang_native_service import TopLogprobList


OverlapBranchPayload = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def build_overlap_branch_payload(
	teacher_top_logprobs: Sequence[Optional[TopLogprobList]],
	student_top_logprobs: Sequence[Optional[TopLogprobList]],
	loss_mask: Sequence[int],
	topk: int,
) -> OverlapBranchPayload:
	"""Pack valid-position top-k data into fixed-shape NumPy arrays."""

	if topk <= 0:
		raise ValueError(f"topk must be positive, got {topk}")

	position_count = min(len(teacher_top_logprobs), len(student_top_logprobs), len(loss_mask))
	valid_indices = [idx for idx in range(position_count) if int(loss_mask[idx]) == 1]
	valid_positions = len(valid_indices)

	teacher_token_ids = np.empty((valid_positions, topk), dtype=np.int32)
	teacher_logprobs = np.empty((valid_positions, topk), dtype=np.float32)
	student_token_ids = np.empty((valid_positions, topk), dtype=np.int32)
	student_logprobs = np.empty((valid_positions, topk), dtype=np.float32)

	for row, idx in enumerate(valid_indices):
		_fill_top_logprobs_row(
			top_logprobs=teacher_top_logprobs[idx],
			topk=topk,
			token_ids_row=teacher_token_ids[row],
			logprobs_row=teacher_logprobs[row],
		)
		_fill_top_logprobs_row(
			top_logprobs=student_top_logprobs[idx],
			topk=topk,
			token_ids_row=student_token_ids[row],
			logprobs_row=student_logprobs[row],
		)

	return teacher_token_ids, teacher_logprobs, student_token_ids, student_logprobs


def compute_overlap_metrics_batch(
	branch_payloads: Sequence[OverlapBranchPayload],
	topk: int,
) -> list[Dict[str, float]]:
	"""Compute overlap metrics for one sample worth of branches."""

	if topk <= 0:
		raise ValueError(f"topk must be positive, got {topk}")
	return [compute_overlap_metrics_for_branch(payload, topk) for payload in branch_payloads]


def compute_overlap_metrics_for_branch(
	branch_payload: OverlapBranchPayload,
	topk: int,
) -> Dict[str, float]:
	"""Compute branch-level overlap ratio and overlap advantage."""

	if topk <= 0:
		raise ValueError(f"topk must be positive, got {topk}")

	teacher_token_ids, teacher_logprobs, student_token_ids, student_logprobs = branch_payload
	valid_positions = int(teacher_token_ids.shape[0])
	if valid_positions == 0:
		return {
			"opd_overlap_ratio": 0.0,
			"opd_overlap_advantage": 0.0,
		}

	matches = teacher_token_ids[:, :, None] == student_token_ids[:, None, :]
	shared_teacher_mask = np.any(matches, axis=2)
	shared_counts = np.sum(shared_teacher_mask, axis=1, dtype=np.int32)
	overlap_ratio = float(np.mean(shared_counts.astype(np.float32) / float(topk)))

	shared_pair_counts = np.sum(matches, axis=(1, 2), dtype=np.int32)
	valid_advantage_mask = shared_pair_counts > 0
	if not np.any(valid_advantage_mask):
		return {
			"opd_overlap_ratio": overlap_ratio,
			"opd_overlap_advantage": 0.0,
		}

	logprob_diffs = teacher_logprobs[:, :, None] - student_logprobs[:, None, :]
	shared_diff_sums = np.sum(logprob_diffs * matches, axis=(1, 2), dtype=np.float64)
	overlap_advantage = float(
		np.mean(
			shared_diff_sums[valid_advantage_mask]
			/ shared_pair_counts[valid_advantage_mask].astype(np.float64)
		)
	)

	return {
		"opd_overlap_ratio": overlap_ratio,
		"opd_overlap_advantage": overlap_advantage,
	}


def _fill_top_logprobs_row(
	top_logprobs: Optional[TopLogprobList],
	topk: int,
	token_ids_row: np.ndarray,
	logprobs_row: np.ndarray,
) -> None:
	for col, entry in enumerate(top_logprobs[:topk]):
		logprob = entry[0]
		token_id = entry[1]
		token_ids_row[col] = int(token_id)
		logprobs_row[col] = 0.0 if logprob is None else float(logprob)