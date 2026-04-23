"""Service/worker registries for async rollout v2."""

from typing import Dict, Type

from light_scale.async_rollout_v2.services.base_service import AsyncBaseService
from light_scale.async_rollout_v2.services.rock_service import AsyncRockService
from light_scale.async_rollout_v2.services.sglang_service import AsyncSGLangService
from light_scale.async_rollout_v2.workers.base_worker import AsyncBaseWorker, AsyncSingleTurnWorker
from light_scale.async_rollout_v2.workers.math_worker import AsyncMathWorker
from light_scale.async_rollout_v2.workers.math_tool_worker import AsyncMathToolWorker
from light_scale.async_rollout_v2.workers.llm_judge_worker import AsyncLLMJudgeWorker
from light_scale.async_rollout_v2.workers.rock_worker import RockWorker


SERVICE_CLASS_REGISTRY: Dict[str, Type[AsyncBaseService]] = {
	"rock": AsyncRockService,
	"sglang": AsyncSGLangService,
}


WORKER_CLASS_REGISTRY: Dict[str, Type[AsyncBaseWorker]] = {
	"single_turn": AsyncSingleTurnWorker,
	"math": AsyncMathWorker,
	"math_tool": AsyncMathToolWorker,
	"llm_judge": AsyncLLMJudgeWorker,
	"rock": RockWorker,
}
