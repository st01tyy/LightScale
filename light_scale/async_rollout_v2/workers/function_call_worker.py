"""Base worker for tool-calling chat rollouts."""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from light_scale.async_rollout_v2.reward_utils import compute_normed_rewards
from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.services.sglang_service import (
    AsyncSGLangService,
    SGLangInputTooLongError,
    SGLangChatCompletionTask,
)
from light_scale.async_rollout_v2.utils.chat_template_utils import (
    convert_openai_trace_to_messages,
    count_tokens,
    extract_compat_response,
    get_cached_template_artifacts,
    normalize_openai_messages,
    normalize_tools,
)
from light_scale.async_rollout_v2.workers.base_worker import AsyncBaseWorker
from light_scale.data import Message, MultiResponseSample


@dataclass
class AsyncFunctionCallWorkerConfig:
    n_samples: int
    max_tokens: int
    advantage_estimator: str
    tokenizer_path: str
    chat_template_path: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    request_timeout: Optional[float] = None
    actor_retry: int = 2
    max_tool_rounds: int = 8
    actor_service_name: str = "actor_model"
    tokenizer_trust_remote_code: bool = True
    add_generation_prompt: bool = False
    actor_extra_body: Optional[Dict[str, Any]] = None
    normalize_assistant_tool_arguments: bool = False
    actor_context_length: Optional[int] = None
    context_budget_margin_tokens: int = 256
    length_limit_policy: str = "invalidate"
    problem_key: str = "problem"
    ground_truth_key: str = "ground_truth"
    sample_id_key: str = "sample_id"
    dataset_type_key: str = "dataset_type"


@dataclass
class FunctionCallBranchResult:
    response: str
    messages: List[Message]
    reward_metrics: Dict[str, Any]
    completion_tokens: int
    total_tokens: int
    valid: bool
    reward: float = 0.0
    valid_for_advantage: bool = False


class AsyncFunctionCallWorker(AsyncBaseWorker):
    CONFIG_CLS = AsyncFunctionCallWorkerConfig
    REQUIRED_SERVICE_NAMES = ["actor_model"]

    TOOL_ROUND_LIMIT_EXIT_REASON = "max_tool_rounds_exceeded"

    def __init__(
        self,
        input_data: dict,
        service_dict: Dict[str, AsyncBaseService],
        stop_event,
        log_level: int,
        teacher_service_name: Optional[str] = None,
        **worker_cfg,
    ):
        super().__init__(
            input_data=input_data,
            service_dict=service_dict,
            stop_event=stop_event,
            log_level=log_level,
            teacher_service_name=teacher_service_name,
        )
        self._config = self.CONFIG_CLS(**worker_cfg)
        self._actor_service = self._require_sglang_service(self._config.actor_service_name)
        self._tokenizer, self._chat_template = get_cached_template_artifacts(
            tokenizer_path=self._config.tokenizer_path,
            trust_remote_code=self._config.tokenizer_trust_remote_code,
            chat_template_path=self._config.chat_template_path,
            owner_name=type(self).__name__,
        )

    async def run(self) -> MultiResponseSample:
        sample = self._build_sample(self.input_data)
        if self.stop_event.is_set():
            self._clear_sample(sample)
            return sample

        branch_tasks = [
            asyncio.create_task(self._run_single_branch(sample, branch_idx))
            for branch_idx in range(self._config.n_samples)
        ]
        branch_results = await asyncio.gather(*branch_tasks, return_exceptions=True)

        outcomes: List[FunctionCallBranchResult] = []
        for branch_result in branch_results:
            if isinstance(branch_result, Exception):
                self.logger.warning("Function call branch failed: %s", branch_result)
                outcomes.append(self._build_failed_branch(sample, str(branch_result)))
            else:
                outcomes.append(branch_result)

        sample.responses = [outcome.response for outcome in outcomes]
        sample.group_messages = [outcome.messages for outcome in outcomes]
        sample.completion_tokens = sum(outcome.completion_tokens for outcome in outcomes)
        sample.total_tokens = sum(outcome.total_tokens for outcome in outcomes)

        sample.rewards = [outcome.reward for outcome in outcomes]
        sample.reward_metrics_list = [deepcopy(outcome.reward_metrics) for outcome in outcomes]
        sample.avg_reward_metrics = self._build_avg_reward_metrics(sample.reward_metrics_list)
        sample.normed_rewards = compute_normed_rewards(
            rewards=sample.rewards,
            valid_flags=[outcome.valid_for_advantage for outcome in outcomes],
            advantage_estimator=self._config.advantage_estimator,
        )
        return sample

    def _build_sample(self, raw_sample: Dict[str, Any]) -> MultiResponseSample:
        problem = str(raw_sample[self._config.problem_key])
        return MultiResponseSample(
            prompt=problem,
            dataset_type=raw_sample[self._config.dataset_type_key],
            ground_truth=str(raw_sample[self._config.ground_truth_key]),
            problem=problem,
            sample_id=raw_sample.get(self._config.sample_id_key, None),
        )

    async def _run_single_branch(
        self,
        sample: MultiResponseSample,
        branch_idx: int,
    ) -> FunctionCallBranchResult:
        branch_tag = f"{sample.sample_id}:{branch_idx}"
        trace_messages = self._build_initial_messages(sample)
        tools = normalize_tools(self._build_tools())
        reward_metrics = self._build_default_reward_metrics()
        reward_metrics.update(deepcopy(self._build_initial_reward_metrics()))
        estimated_total_tokens = self._initialize_estimated_trace_tokens(trace_messages, tools)

        for round_idx in range(self._config.max_tool_rounds):
            if self.stop_event.is_set():
                raise RuntimeError("stop_event set during function call rollout")

            if self._is_context_budget_exceeded(estimated_total_tokens):
                reward_metrics["context_budget_exceeded"] = 1
                reward_metrics["exit_reason"] = "local_context_budget"
                branch_result = self._build_branch_result(trace_messages, tools, reward_metrics)
                return await self._finalize_branch_result(sample, branch_result)

            task = self._build_actor_chat_task(
                messages=trace_messages,
                tools=tools,
            )
            try:
                result = await self._actor_service.submit(task)
            except SGLangInputTooLongError as err:
                reward_metrics["service_context_overflow"] = 1
                reward_metrics["exit_reason"] = "service_context_overflow"
                reward_metrics["context_overflow_input_tokens"] = err.input_tokens
                reward_metrics["context_overflow_limit_tokens"] = err.max_context_tokens
                branch_result = self._build_branch_result(trace_messages, tools, reward_metrics)
                return await self._finalize_branch_result(sample, branch_result)

            assistant_message = self._extract_assistant_message(result)
            trace_messages.append(assistant_message)
            if self._config.normalize_assistant_tool_arguments:
                self._normalize_tool_call_arguments(trace_messages[-1])
            estimated_total_tokens = self._sync_estimated_trace_tokens_with_result(
                result=result,
                estimated_total_tokens=estimated_total_tokens,
            )

            if assistant_message.get("finish_reason") == "length":
                reward_metrics["actor_finish_reason_length"] = 1
                reward_metrics.setdefault("exit_reason", "actor_finish_reason_length")

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                branch_result = self._build_branch_result(trace_messages, tools, reward_metrics)
                return await self._finalize_branch_result(sample, branch_result)

            for tool_pos, tool_call in enumerate(tool_calls):
                remaining_context_tokens = self._get_remaining_context_tokens(estimated_total_tokens)
                tool_message, metric_updates = await self._handle_tool_call(
                    tool_call=tool_call,
                    branch_tag=branch_tag,
                    round_idx=round_idx,
                    tool_pos=tool_pos,
                    sample=sample,
                    trace_messages=trace_messages,
                    remaining_context_tokens=remaining_context_tokens,
                )
                self._merge_reward_metrics(reward_metrics, metric_updates)
                trace_messages.append(tool_message)
                estimated_total_tokens = self._update_estimated_trace_tokens_with_local_message(
                    estimated_total_tokens=estimated_total_tokens,
                    message=tool_message,
                    trace_messages=trace_messages,
                )

        reward_metrics.setdefault("exit_reason", self.TOOL_ROUND_LIMIT_EXIT_REASON)
        self._on_tool_round_limit_exceeded(reward_metrics)
        return await self._run_tool_round_limit_final_turn(
            sample=sample,
            trace_messages=trace_messages,
            tools=tools,
            reward_metrics=reward_metrics,
        )

    def _build_actor_chat_task(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any] = None,
    ) -> SGLangChatCompletionTask:
        return SGLangChatCompletionTask(
            prompt="",
            n_samples=1,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            presence_penalty=self._config.presence_penalty,
            retry=self._config.actor_retry,
            timeout=self._config.request_timeout,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

    async def _run_tool_round_limit_final_turn(
        self,
        *,
        sample: MultiResponseSample,
        trace_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        reward_metrics: Dict[str, Any],
    ) -> FunctionCallBranchResult:
        final_turn_messages = trace_messages + [self._build_tool_round_limit_notice_message()]
        task = self._build_actor_chat_task(
            messages=final_turn_messages,
            tools=tools,
            tool_choice="none",
        )
        try:
            result = await self._actor_service.submit(task)
        except SGLangInputTooLongError as err:
            reward_metrics["service_context_overflow"] = 1
            reward_metrics["context_overflow_input_tokens"] = err.input_tokens
            reward_metrics["context_overflow_limit_tokens"] = err.max_context_tokens
            branch_result = self._build_branch_result(trace_messages, tools, reward_metrics)
            return await self._finalize_branch_result(sample, branch_result)

        assistant_message = self._extract_assistant_message(result)
        if assistant_message.get("finish_reason") == "length":
            reward_metrics["actor_finish_reason_length"] = 1
            reward_metrics.setdefault("exit_reason", "actor_finish_reason_length")
        if assistant_message.get("tool_calls"):
            reward_metrics["tool_limit_final_turn_called_tool"] = 1
            assistant_message = deepcopy(assistant_message)
            assistant_message.pop("tool_calls", None)
        final_turn_messages.append(assistant_message)
        branch_result = self._build_branch_result(final_turn_messages, tools, reward_metrics)
        return await self._finalize_branch_result(sample, branch_result)

    def _build_branch_result(
        self,
        raw_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        reward_metrics: Dict[str, Any],
    ) -> FunctionCallBranchResult:
        normalized_messages = normalize_openai_messages(raw_messages)
        rendered_messages = convert_openai_trace_to_messages(
            tokenizer=self._tokenizer,
            messages=normalized_messages,
            tools=tools,
            chat_template=self._chat_template,
            add_generation_prompt=self._config.add_generation_prompt,
        )
        completion_tokens, total_tokens = count_tokens(self._tokenizer, rendered_messages)
        return FunctionCallBranchResult(
            response=extract_compat_response(normalized_messages),
            messages=rendered_messages,
            reward_metrics=deepcopy(reward_metrics),
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            valid=bool(rendered_messages),
        )

    def _build_failed_branch(
        self,
        sample: MultiResponseSample,
        reason: str,
    ) -> FunctionCallBranchResult:
        masked_content = sample.prompt if sample.prompt else (reason if reason else "Function call worker failed")
        messages = [Message(content=masked_content, is_masked=True)]
        reward_metrics = self._build_default_reward_metrics()
        reward_metrics.update(deepcopy(self._build_failure_reward_metrics()))
        return FunctionCallBranchResult(
            response="",
            messages=messages,
            reward_metrics=reward_metrics,
            completion_tokens=0,
            total_tokens=sum(
                len(self._tokenizer.encode(message.content, add_special_tokens=False))
                for message in messages
            ),
            valid=False,
        )

    def _extract_assistant_message(self, result) -> Dict[str, Any]:
        if getattr(result, "messages", None):
            message = deepcopy(result.messages[0])
        else:
            text = result.responses[0][0] if result.responses else ""
            message = {"role": "assistant", "content": text}
        message.setdefault("role", "assistant")
        message.setdefault("content", "")
        return message

    def _normalize_tool_call_arguments(self, message: Dict[str, Any]) -> None:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue

            arguments = function_payload.get("arguments")
            if not isinstance(arguments, str):
                continue

            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed_arguments, dict):
                function_payload["arguments"] = parsed_arguments

    def _build_actor_extra_body(self) -> Optional[Dict[str, Any]]:
        # TODO: Reasoning-specific request extensions are intentionally unsupported
        # in function-call workers for now. Revisit this hook only after the rollout
        # path can represent reasoning outputs and exit reasons explicitly.
        return deepcopy(self._config.actor_extra_body) if self._config.actor_extra_body is not None else None

    def _initialize_estimated_trace_tokens(
        self,
        trace_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> int:
        if self._config.actor_context_length is None:
            return 0
        normalized_messages = normalize_openai_messages(trace_messages)
        rendered_messages = convert_openai_trace_to_messages(
            tokenizer=self._tokenizer,
            messages=normalized_messages,
            tools=tools,
            chat_template=self._chat_template,
            add_generation_prompt=self._config.add_generation_prompt,
        )
        _, total_tokens = count_tokens(self._tokenizer, rendered_messages)
        return total_tokens

    def _sync_estimated_trace_tokens_with_result(
        self,
        *,
        result,
        estimated_total_tokens: int,
    ) -> int:
        if self._config.actor_context_length is None:
            return 0

        result_total_tokens = getattr(result, "total_tokens", None)
        if isinstance(result_total_tokens, int) and result_total_tokens > 0:
            return result_total_tokens
        return estimated_total_tokens

    def _update_estimated_trace_tokens_with_local_message(
        self,
        *,
        estimated_total_tokens: int,
        message: Dict[str, Any],
        trace_messages: List[Dict[str, Any]],
    ) -> int:
        if self._config.actor_context_length is None:
            return 0
        return estimated_total_tokens + self._estimate_local_message_token_delta(
            message=message,
            trace_messages=trace_messages,
        )

    def _get_prompt_token_limit(self) -> Optional[int]:
        if self._config.actor_context_length is None:
            return None
        return max(
            self._config.actor_context_length
            - self._config.context_budget_margin_tokens,
            0,
        )

    def _get_remaining_context_tokens(self, estimated_total_tokens: int) -> Optional[int]:
        prompt_token_limit = self._get_prompt_token_limit()
        if prompt_token_limit is None:
            return None
        return max(prompt_token_limit - estimated_total_tokens, 0)

    def _is_context_budget_exceeded(self, estimated_total_tokens: int) -> bool:
        prompt_token_limit = self._get_prompt_token_limit()
        if prompt_token_limit is None:
            return False
        return estimated_total_tokens >= prompt_token_limit

    def _has_length_limit_issue(self, reward_metrics: Dict[str, Any]) -> bool:
        return any(
            int(reward_metrics.get(key, 0)) == 1
            for key in (
                "context_budget_exceeded",
                "service_context_overflow",
                "actor_finish_reason_length",
            )
        )

    def _apply_length_limit_policy(
        self,
        branch_result: FunctionCallBranchResult,
    ) -> FunctionCallBranchResult:
        if not self._has_length_limit_issue(branch_result.reward_metrics):
            return branch_result

        branch_result.reward_metrics["length_limit_hit"] = 1
        policy = self._config.length_limit_policy.strip().lower()
        if policy == "invalidate":
            branch_result.valid_for_advantage = False
            return branch_result
        if policy == "zero_reward":
            branch_result.reward = 0.0
            return branch_result
        raise ValueError(f"Unknown length_limit_policy: {self._config.length_limit_policy}")

    async def _finalize_branch_result(
        self,
        sample: MultiResponseSample,
        branch_result: FunctionCallBranchResult,
    ) -> FunctionCallBranchResult:
        branch_result.valid_for_advantage = branch_result.valid
        return self._apply_length_limit_policy(branch_result)

    def _merge_reward_metrics(
        self,
        reward_metrics: Dict[str, Any],
        metric_updates: Optional[Dict[str, Any]],
    ) -> None:
        if not metric_updates:
            return
        for key, value in metric_updates.items():
            if isinstance(value, (int, float)) and isinstance(reward_metrics.get(key), (int, float)):
                reward_metrics[key] = min(reward_metrics[key], value)
            else:
                reward_metrics[key] = value

    def _build_avg_reward_metrics(self, reward_metrics_list: List[Dict[str, Any]]) -> List[str]:
        avg_reward_metrics: List[str] = []
        for reward_metrics in reward_metrics_list:
            for key in reward_metrics.keys():
                if key not in avg_reward_metrics:
                    avg_reward_metrics.append(key)
        return avg_reward_metrics

    def _clear_sample(self, sample: MultiResponseSample) -> None:
        sample.responses = []
        sample.group_messages = []
        sample.rewards = []
        sample.reward_metrics_list = []
        sample.avg_reward_metrics = []
        sample.normed_rewards = []
        sample.completion_tokens = 0
        sample.total_tokens = 0

    def _require_sglang_service(self, service_name: str) -> AsyncSGLangService:
        service = self.service_dict.get(service_name)
        if service is None:
            raise RuntimeError(f"{type(self).__name__} missing service '{service_name}'")
        if not isinstance(service, AsyncSGLangService):
            raise TypeError(
                f"{type(self).__name__} requires AsyncSGLangService for '{service_name}', got {type(service).__name__}"
            )
        return service
 
    def _build_initial_reward_metrics(self) -> Dict[str, Any]:
        return {}

    def _build_default_reward_metrics(self) -> Dict[str, Any]:
        return {
            "context_budget_exceeded": 0,
            "service_context_overflow": 0,
            "actor_finish_reason_length": 0,
            "length_limit_hit": 0,
        }

    def _build_failure_reward_metrics(self) -> Dict[str, Any]:
        return deepcopy(self._build_initial_reward_metrics())

    def _build_tool_round_limit_notice_message(self) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": (
                "The tool-use round limit has been reached. Do not call any more tools. "
                "Based only on the previous observations, provide your final answer now."
            ),
        }

    def _on_tool_round_limit_exceeded(self, reward_metrics: Dict[str, Any]) -> None:
        reward_metrics["tool_round_limit_exceeded"] = 1

    def _build_initial_messages(self, sample: MultiResponseSample) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _build_tools(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _estimate_local_message_token_delta(
        self,
        *,
        message: Dict[str, Any],
        trace_messages: List[Dict[str, Any]],
    ) -> int:
        raise NotImplementedError

    async def _handle_tool_call(
        self,
        *,
        tool_call: Dict[str, Any],
        branch_tag: str,
        round_idx: int,
        tool_pos: int,
        sample: MultiResponseSample,
        trace_messages: List[Dict[str, Any]],
        remaining_context_tokens: Optional[int],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError
