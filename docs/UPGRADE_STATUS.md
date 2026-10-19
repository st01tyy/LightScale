# Upgrade integration status

This source snapshot incorporates the Megatron 0.18.0 migration and the existing
SGLang 0.5.9 integration, together with the accumulated generic training, packing,
checkpoint conversion, and rollout improvements.

## Validation boundary

This integration was prepared by static source inspection. No test suite,
training job, distributed job, or runtime dependency import was executed in the
integration environment. Compatibility with a particular deployment must still be
validated in an appropriately provisioned environment.

The existing `build.dockerfile` is preserved unchanged. It is not an upgrade
validation artifact or a verified environment for this snapshot. Older version
references elsewhere in the documentation may also require follow-up alignment.

The `megatron_old/` tree is retained as a legacy reference, not as the active
training implementation. The active implementation is under `megatron/`.

## External expert IDs: consumer implemented, producer incomplete

The existing external expert-ID feature is intentionally retained, including
its trainer-side data contract, packed metadata staging, MoE routing, and
activation-recompute plumbing. The rollout/SGLang producer that populates
`MultiResponseSample.group_expert_ids` is not complete in this snapshot.

Consequently, this snapshot does **not** claim an end-to-end external expert-ID
workflow. Do not enable the feature expecting the current rollout path to produce
the required metadata automatically. Enabling it without supplying valid expert
IDs can fail trainer input validation. No producer development was performed as
part of the integration.

Preserved data invariants include:

- IDs represent global logical experts, not EP-local IDs or dispatch slots.
- Valid IDs use `int32`; physical-padding entries use `-1`.
- Physical padding and loss masking have different meanings.
- Recompute uses the explicit expert-ID tensors rather than a global replay FIFO.

## Standalone verifier examples

The unused legacy `llm_as_judge` verifier has been removed. The async rollout
judge implementation remains in `workers/llm_judge_worker.py` and
`utils/llm_judge.py` under `light_scale/async_rollout_v2/`. Retained callable
verifier APIs are unchanged. Their standalone example blocks no longer contain
deployment-specific endpoints or dataset paths. Supply these explicitly:

- `verifier/code.py`: `CODE_SANDBOX_URLS` (comma-separated endpoints) and
  `CODE_EVAL_DATA_PATH` (a local JSONL dataset).
- `tests/test.sh`: `GRAD_TEST_TOKENIZER` is required; `GRAD_TEST_DATA` and
  `GRAD_TEST_OUTPUT` optionally override the local example paths.

Internal-only rollout workers and model patches are not part of the public
integration snapshot. The common service and worker registries remain available.
