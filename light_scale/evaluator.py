"""Rollout-only evaluator."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from numbers import Number
from queue import Empty as QueueEmpty
from queue import Queue as MpQueue
from threading import Event as MpEvent
from threading import Thread
from typing import Any, Dict, List, Optional, Set

from light_scale.data import MultiResponseSample
from light_scale.logger_utils import get_logging_queue, setup_logger_v2_main_process
from light_scale.async_rollout_v2.rollout_thread import rollout_thread_main


@dataclass
class EvaluatorConfig:
    async_rollout_cfg_path: str
    rollout_batch_size: int
    dump_path: str
    log_file_path: Optional[str] = None
    n_samples: int = 1
    passed_iters: int = 0
    light_scale_log_level: str = "info"
    init_timeout_seconds: float = 300.0
    sample_poll_timeout_seconds: float = 0.5


class Evaluator:
    AVG_METRICS = ["reward", "completion_tokens", "total_tokens"]
    RESULTS_FILENAME = "results.jsonl"
    COMPLETED_IDS_FILENAME = "completed_sample_ids.txt"
    SUMMARY_FILENAME = "summary.json"

    def __init__(self, config: EvaluatorConfig):
        if config.rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        if config.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not config.dump_path:
            raise ValueError("dump_path must be provided")

        self.config = config
        log_level = getattr(logging, str(config.light_scale_log_level).upper(), logging.INFO)
        self.logger = setup_logger_v2_main_process(
            "light_scale.eval",
            setup_distributed=False,
            level=log_level,
            log_file_path=config.log_file_path,
        )
        self.log_level = log_level

        os.makedirs(self.config.dump_path, exist_ok=True)
        self.results_path = os.path.join(self.config.dump_path, self.RESULTS_FILENAME)
        self.completed_ids_path = os.path.join(self.config.dump_path, self.COMPLETED_IDS_FILENAME)
        self.summary_path = os.path.join(self.config.dump_path, self.SUMMARY_FILENAME)
        self.completed_sample_ids = self._load_completed_sample_ids()
        self.per_type_rollout_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.per_type_avg_metrics: Dict[str, Set[str]] = {}
        self.processed_prompt_count = 0
        self.processed_response_count = 0

        self.input_queue: Optional[MpQueue] = None
        self.output_queue: Optional[MpQueue] = None
        self.stop_event: Optional[MpEvent] = None
        self.failed_event: Optional[MpEvent] = None
        self.start_event: Optional[MpEvent] = None
        self.rollout_thread: Optional[Thread] = None

    def evaluate(self) -> Dict[str, Any]:
        self._start_rollout_thread()
        try:
            rollout_step = self.config.passed_iters + 1
            self.logger.info("starting eval rollout")
            self.input_queue.put(rollout_step)
            step_start_time = time.time()
            summary = self._consume_rollout_until_end(rollout_step, step_start_time)
            self._log_step_summary(summary)
            self._write_summary(summary)
            return summary
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.rollout_thread is not None:
            self.rollout_thread.join(timeout=5)

    def _start_rollout_thread(self) -> None:
        self.input_queue = MpQueue()
        self.output_queue = MpQueue()
        self.stop_event = MpEvent()
        self.failed_event = MpEvent()
        self.start_event = MpEvent()
        self.rollout_thread = Thread(
            target=rollout_thread_main,
            args=(
                self.config.async_rollout_cfg_path,
                self.config.passed_iters,
                self.config.rollout_batch_size,
                self.input_queue,
                self.output_queue,
                self.stop_event,
                get_logging_queue(),
                self.start_event,
                self.failed_event,
                self.log_level,
                "eval",
                self.completed_ids_path,
            ),
            daemon=True,
        )
        self.rollout_thread.start()

        deadline = time.time() + self.config.init_timeout_seconds
        while time.time() < deadline:
            if self.start_event.is_set() and not self.failed_event.is_set():
                self.logger.info("rollout thread initialized")
                return
            if self.failed_event.is_set():
                raise RuntimeError("rollout thread initialization failed")
            time.sleep(0.1)
        raise TimeoutError("timed out waiting for rollout thread initialization")

    def _consume_rollout_until_end(self, rollout_step: int, start_time: float) -> Dict[str, Any]:
        while True:
            if self.failed_event is not None and self.failed_event.is_set():
                raise RuntimeError(f"rollout thread failed while collecting step {rollout_step}")
            if self.stop_event is not None and self.stop_event.is_set() and self.output_queue.empty():
                raise RuntimeError(f"rollout stopped early while collecting step {rollout_step}")
            try:
                sample = self.output_queue.get(timeout=self.config.sample_poll_timeout_seconds)
            except QueueEmpty:
                if self.rollout_thread is not None and not self.rollout_thread.is_alive():
                    raise RuntimeError(
                        f"rollout thread exited before step {rollout_step} collected enough samples"
                    )
                continue
            if not isinstance(sample, MultiResponseSample):
                raise TypeError(f"unexpected rollout output type: {type(sample)}")
            self.logger.info("eval sample received: sample_id=%s, reward=%s", sample.sample_id, sample.rewards)
            if sample.is_end_of_rollout():
                return self._build_summary(rollout_step, time.time() - start_time)
            self._persist_sample(sample)
            self._update_rollout_metrics(sample)

    def _build_summary(
        self,
        rollout_step: int,
        duration_seconds: float,
    ) -> Dict[str, Any]:
        metrics_by_dataset: Dict[str, Dict[str, float]] = {}
        for data_type, rollout_metrics in self.per_type_rollout_metrics.items():
            avg_metrics = self.per_type_avg_metrics.get(data_type, set())
            metrics_for_log: Dict[str, float] = {}
            for name, values in rollout_metrics.items():
                numeric_values = [value for value in values if isinstance(value, Number)]
                if not numeric_values:
                    continue
                if name in self.AVG_METRICS or name in avg_metrics:
                    metrics_for_log[name] = sum(numeric_values) / len(numeric_values)
                else:
                    metrics_for_log[name] = sum(numeric_values)
            if "completion_tokens" in metrics_for_log:
                metrics_for_log["completion_tokens"] = int(metrics_for_log["completion_tokens"] / self.config.n_samples)
            if "total_tokens" in metrics_for_log:
                metrics_for_log["total_tokens"] = int(metrics_for_log["total_tokens"] / self.config.n_samples)
            metrics_by_dataset[data_type] = metrics_for_log

        return {
            "step": rollout_step,
            "num_prompts": self.processed_prompt_count,
            "num_responses": self.processed_response_count,
            "num_completed_sample_ids": len(self.completed_sample_ids),
            "duration_seconds": duration_seconds,
            "metrics_by_dataset": metrics_by_dataset,
        }

    def _update_rollout_metrics(
        self,
        batch_sample: MultiResponseSample,
    ) -> None:
        dataset_type = batch_sample.dataset_type
        rollout_metrics = self.per_type_rollout_metrics.setdefault(dataset_type, {})
        avg_metrics = self.per_type_avg_metrics.setdefault(dataset_type, set())

        rollout_metrics.setdefault("reward", []).extend(batch_sample.rewards or [])
        if batch_sample.completion_tokens is not None:
            rollout_metrics.setdefault("completion_tokens", []).append(batch_sample.completion_tokens)
        if batch_sample.total_tokens is not None:
            rollout_metrics.setdefault("total_tokens", []).append(batch_sample.total_tokens)

        reward_source = batch_sample.normed_rewards if batch_sample.normed_rewards is not None else (batch_sample.rewards or [])
        rollout_metrics.setdefault("invalid_samples", []).extend([1 if reward == 0.0 else 0 for reward in reward_source])

        for reward_metric in batch_sample.reward_metrics_list or []:
            if reward_metric is None:
                continue
            for key, value in reward_metric.items():
                rollout_metrics.setdefault(key, []).append(value)

        for metric_name in batch_sample.avg_reward_metrics or []:
            avg_metrics.add(metric_name)

        self.processed_prompt_count += 1
        self.processed_response_count += len(batch_sample.responses or [])

    def _log_step_summary(self, summary: Dict[str, Any]) -> None:
        step_label = summary["step"]
        self.logger.info(
            "eval step %s finished: prompts=%s, responses=%s, duration=%.2fs",
            step_label,
            summary["num_prompts"],
            summary["num_responses"],
            summary["duration_seconds"],
        )
        for data_type, metrics in summary["metrics_by_dataset"].items():
            if not metrics:
                self.logger.info("eval step %s, data_type=%s: no numeric metrics", step_label, data_type)
                continue
            ordered_items = ", ".join(f"{key}: {value}" for key, value in metrics.items())
            self.logger.info("eval step %s, data_type=%s, %s", step_label, data_type, ordered_items)

    def _persist_sample(self, sample: MultiResponseSample) -> None:
        if sample.sample_id is None:
            raise ValueError("eval mode requires sample.sample_id for resumable persistence")

        normalized_sample_id = str(sample.sample_id)
        if normalized_sample_id in self.completed_sample_ids:
            self.logger.warning("sample_id %s already completed, skip duplicate result", normalized_sample_id)
            return

        with open(self.results_path, mode="a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(self._sample_to_dump_dict(sample), ensure_ascii=False))
            file_obj.write("\n")
            file_obj.flush()

        with open(self.completed_ids_path, mode="a", encoding="utf-8") as file_obj:
            file_obj.write(normalized_sample_id)
            file_obj.write("\n")
            file_obj.flush()

        self.completed_sample_ids.add(normalized_sample_id)

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        with open(self.summary_path, mode="w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")

    def _load_completed_sample_ids(self) -> Set[str]:
        if not os.path.exists(self.completed_ids_path):
            return set()
        completed_ids: Set[str] = set()
        with open(self.completed_ids_path, mode="r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    completed_ids.add(line)
        return completed_ids

    @staticmethod
    def _sample_to_dump_dict(sample: MultiResponseSample) -> Dict[str, Any]:
        sample_dict = asdict(sample)
        sample_dict.pop("group_content_ids", None)
        sample_dict.pop("group_loss_mask", None)
        sample_dict.pop("group_teacher_log_probs", None)
        return sample_dict