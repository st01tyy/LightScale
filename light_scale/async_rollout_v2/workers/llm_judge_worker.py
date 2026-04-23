"""Async LLM judge worker for async rollout v2."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from light_scale.data import MultiResponseSample
from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.services.sglang_service import (
    AsyncSGLangService,
)
from light_scale.async_rollout_v2.utils.llm_judge import (
    LLMJudgeConfig,
    build_judge_messages,
    judge_responses,
    judge_single_response,
    maybe_extract_cot_response,
    parse_judge_json,
    to_verdict,
)
from light_scale.async_rollout_v2.workers.base_worker import AsyncSingleTurnWorker
from light_scale.async_rollout_v2.workers.base_worker import AsyncSingleTurnWorkerConfig
from light_scale.async_rollout_v2.reward_utils import compute_normed_rewards

@dataclass
class AsyncLLMJudgeWorkerConfig(AsyncSingleTurnWorkerConfig):
    """LLM Judge worker 配置。"""

    # Judge specific fields
    judge_timeout: float = 600.0
    judge_retry: int = 2
    judge_json_retries: int = 2
    judge_temperature: float = 0.2
    judge_parallelism: int = 0
    judge_max_tokens: int = 1024
    use_cot_reward: bool = False
    use_ref_answers: bool = False
    max_num_ref_answers: int = 1


class AsyncLLMJudgeWorker(AsyncSingleTurnWorker):
    """异步 LLM Judge worker：先采样，再逐条判定打分。"""

    CONFIG_CLS = AsyncLLMJudgeWorkerConfig
    JUDGE_SERVICE_NAME = "judge_model"
    REQUIRED_SERVICE_NAMES = AsyncSingleTurnWorker.REQUIRED_SERVICE_NAMES + [JUDGE_SERVICE_NAME]
    USES_LLM_JUDGE = True

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
        self._judge_service = self._require_judge_service()

    def _require_judge_service(self) -> AsyncSGLangService:
        service = self.service_dict.get(self.JUDGE_SERVICE_NAME)
        if service is None:
            raise RuntimeError(
                f"AsyncLLMJudgeWorker 初始化失败：缺少 {self.JUDGE_SERVICE_NAME} 服务实例"
            )
        if not isinstance(service, AsyncSGLangService):
            raise TypeError(
                f"AsyncLLMJudgeWorker 需要 AsyncSGLangService 类型的 judge 服务，当前为 {type(service).__name__}"
            )
        return service

    async def _score_responses(self, sample: MultiResponseSample, raw_sample: dict) -> None:
        responses = sample.responses or []
        if not responses:
            sample.rewards = []
            sample.reward_metrics_list = []
            sample.avg_reward_metrics = []
            return

        processed_responses = [self._maybe_extract_response(response) for response in responses]
        rewards, reward_metrics_list = await judge_responses(
            judge_service=self._judge_service,
            sample=sample,
            responses=processed_responses,
            judge_config=LLMJudgeConfig.from_object(self._config, fallback_max_tokens=self._config.max_tokens),
            logger=self.logger,
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

    async def _judge_single_response(self, sample: MultiResponseSample, response: str) -> Tuple[float, dict]:
        return await judge_single_response(
            judge_service=self._judge_service,
            sample=sample,
            response=response,
            judge_config=LLMJudgeConfig.from_object(self._config, fallback_max_tokens=self._config.max_tokens),
            logger=self.logger,
        )

    def _maybe_extract_response(self, response: str) -> str:
        return maybe_extract_cot_response(
            response,
            use_cot_reward=self._config.use_cot_reward,
            end_of_thinking=self._config.end_of_thinking,
        )

    def _build_judge_messages(self, sample: MultiResponseSample, response: str) -> List[Dict[str, str]]:
        return build_judge_messages(sample, response)

    def _parse_judge_json(self, raw_text: str) -> dict:
        return parse_judge_json(raw_text)

    def _to_verdict(self, value) -> Optional[bool]:
        return to_verdict(value)

    def _postprocess_sample(self, sample: MultiResponseSample) -> None:
        responses = sample.responses or []
        if not responses:
            sample.normed_rewards = []
            return
        if sample.rewards is None:
            sample.rewards = [None] * len(responses)

        reward_metrics_list = sample.reward_metrics_list or [dict() for _ in responses]
        valid_flags = []
        for idx, (resp, reward) in enumerate(zip(responses, sample.rewards)):
            judge_ok = int(reward_metrics_list[idx].get("judge_ok", 0)) if idx < len(reward_metrics_list) else 0
            valid_flags.append(resp is not None and reward is not None and judge_ok == 1)

        sample.normed_rewards = compute_normed_rewards(
            rewards=sample.rewards,
            valid_flags=valid_flags,
            advantage_estimator=self._config.advantage_estimator,
        )

        for idx, metric in enumerate(reward_metrics_list):
            if int(metric.get("judge_ok", 0)) == 0:
                sample.normed_rewards[idx] = 0.0
