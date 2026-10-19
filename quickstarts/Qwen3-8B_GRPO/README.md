# Qwen3-8B GRPO Quickstart

This quickstart shows how to run standard GRPO-based RL post-training on the DeepScaleR dataset with Qwen3-8B-Base as the actor model.

## Hardware Requirement

- Verified target GPU: H800 80GB
- Minimum GPU count for this quickstart: 16 GPUs

## Goal

- Dataset: DeepScaleR
- Model: Qwen3-8B-Base
- Training type: GRPO with async rollout
- Output: actor checkpoints, rollout experiences, TensorBoard logs, exported Hugging Face checkpoint

## What You Need to Prepare in Advance

Download the dataset JSON file:

```bash
wget https://modelscope.cn/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/master/deepscaler.json
```

Download the base model:

```bash
git clone https://www.modelscope.cn/Qwen/Qwen3-8B-Base.git
```

Recommended workspace assumption for the examples below:

- Dataset file: `~/data/deepscaler.json`
- Base model directory: `~/models/Qwen3-8B-Base`

## Files in This Directory

- [config.yml](config.yml): async GRPO configuration
- [01-process-dataset.sh](01-process-dataset.sh): converts the raw DeepScaleR JSON into a Hugging Face dataset used by async rollout
- [02-convert-model.sh](02-convert-model.sh): converts the base model into Megatron format and creates an HF symlink used by the rollout service
- [03-train.sh](03-train.sh): launches GRPO training through `headquarters_v2.py`
- [04-convert-result.sh](04-convert-result.sh): exports the final actor checkpoint back to Hugging Face format
- [process_deepscaler_for_ds_r1_zero.py](process_deepscaler_for_ds_r1_zero.py): dataset processing script

## Step 1: Process the Dataset

Purpose:
Convert DeepScaleR samples into prompt, ground-truth, and dataset-type fields for the GRPO rollout worker.

Script:
[01-process-dataset.sh](01-process-dataset.sh)

Arguments:

- `$1`: raw dataset JSON path

Example:

```bash
cd quickstarts/Qwen3-8B_GRPO
bash 01-process-dataset.sh ~/data/deepscaler.json
```

What this script does:

- Runs [process_deepscaler_for_ds_r1_zero.py](process_deepscaler_for_ds_r1_zero.py)
- Writes the processed Hugging Face dataset to `./deepscaler_r1_zero`
- Builds prompts in the reasoning-style format expected by the GRPO rollout pipeline

## Step 2: Convert the Base Model

Purpose:
Prepare both the Megatron-format training checkpoint and the Hugging Face path used by the rollout service.

Script:
[02-convert-model.sh](02-convert-model.sh)

Arguments:

- `$1`: base model path

Example:

```bash
cd quickstarts/Qwen3-8B_GRPO
bash 02-convert-model.sh ~/models/Qwen3-8B-Base
```

What this script does:

- Creates `../../hf_models/Qwen3-8B-Base` as a symlink to the original model
- Converts the model into Megatron format at `../../megatron_models/Qwen3-8B-Base`

## Step 3: Launch GRPO Training

Script:
[03-train.sh](03-train.sh)

Example:

```bash
cd quickstarts/Qwen3-8B_GRPO
bash 03-train.sh
```

What this script runs:

```bash
python3 headquarters_v2.py --config ./quickstarts/Qwen3-8B_GRPO/config.yml
```

Note:
The script contains a reminder that you should define the distributed environment variables such as `NODE_RANK` and `NODE_LIST` for your cluster before launching multi-node training.

Important configuration items in [config.yml](config.yml):

- `training.output_dir`: `./quickstarts/Qwen3-8B_GRPO/training_outputs`
- `training.rollout_batch_size`: `128`
- `training.global_batch_size`: `1024`
- `training.n_samples`: `8`
- `training.max_steps`: `314`
- `training.skip_zero_reward_sample`: `True`
- `algorithm.advantage_estimator`: `grpo`
- `actor.tp`: `2`
- `actor.pp`: `2`
- `actor.dp`: `2`
- `async_rollout.data`: `./quickstarts/Qwen3-8B_GRPO/deepscaler_r1_zero`
- `async_rollout.services[0].resource_cfg.params.model_path`: `./hf_models/Qwen3-8B-Base`
- `async_rollout.workers[0].params.max_tokens`: `6500`
- `async_rollout.workers[1].type`: `math`

These parameters mean this quickstart uses async rollout with an SGLang actor service, 8 samples per prompt, and math-domain rollout handling for the `deepscaler` dataset type.

## Step 4: Convert the Final Actor Checkpoint Back to Hugging Face Format

Purpose:
Export the trained actor checkpoint for downstream inference or evaluation.

Script:
[04-convert-result.sh](04-convert-result.sh)

Arguments:

- `$1`: original base model path

Example:

```bash
cd quickstarts/Qwen3-8B_GRPO
bash 04-convert-result.sh ~/models/Qwen3-8B-Base
```

Output:

- Final Hugging Face checkpoint:
	`./training_outputs/hf_checkpoints/step_300`

## Where to Check Training Progress

- Actor training log:
	`./training_outputs/actor_log/rank_0.log`
- TensorBoard log directory:
	`./training_outputs/tensorboard_log/`
- Rollout experiences:
	`./training_outputs/experiences/`
- Megatron checkpoints:
	`./training_outputs/checkpoints`

Typical TensorBoard usage:

```bash
tensorboard --logdir quickstarts/Qwen3-8B_GRPO/training_outputs/tensorboard_log
```

## Training Results Reference

![reward](./result.png)

## Minimal End-to-End Example

```bash
cd quickstarts/Qwen3-8B_GRPO
bash 01-process-dataset.sh ~/data/deepscaler.json
bash 02-convert-model.sh ~/models/Qwen3-8B-Base
bash 03-train.sh
bash 04-convert-result.sh ~/models/Qwen3-8B-Base
```

## Summary

Use this path when you want the standard LiteScale RL workflow: online sampling, reward-driven updates, async rollout, and actor checkpoint export.