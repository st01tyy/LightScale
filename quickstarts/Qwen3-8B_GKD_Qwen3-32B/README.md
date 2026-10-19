# Qwen3-8B GKD with Qwen3-32B Quickstart

This quickstart shows how to run LiteScale's online GKD workflow with Qwen3-8B as the student and Qwen3-32B as the teacher on the DeepScaleR preview dataset.

## Hardware Requirement

- Verified target GPU: H800 80GB
- Minimum GPU count for this quickstart: 24 GPUs

## Goal

- Dataset: DeepScaleR
- Student model: Qwen3-8B
- Teacher model: Qwen3-32B
- Training type: online GKD
- Output: student checkpoints, student and teacher logs, exported student Hugging Face checkpoint

## What You Need to Prepare in Advance

Download the dataset JSON file:

```bash
wget https://modelscope.cn/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/master/deepscaler.json
```

Download the student and teacher models:

```bash
git clone https://www.modelscope.cn/Qwen/Qwen3-8B.git
git clone https://www.modelscope.cn/Qwen/Qwen3-32B.git
```

Recommended workspace assumption for the examples below:

- Dataset file: `~/data/deepscaler.json`
- Student model directory: `~/models/Qwen3-8B`
- Teacher model directory: `~/models/Qwen3-32B`

## Files in This Directory

- [config.yml](config.yml): online GKD training configuration
- [01-process-dataset.sh](01-process-dataset.sh): converts the raw DeepScaleR JSON into a Hugging Face dataset used by async rollout
- [02-convert-model.sh](02-convert-model.sh): converts both student and teacher checkpoints into the formats used by LiteScale
- [03-train.sh](03-train.sh): launches GKD training through `headquarters_v2.py`
- [04-convert-result.sh](04-convert-result.sh): exports the trained student checkpoint back to Hugging Face format
- [process_deepscaler_for_gkd.py](process_deepscaler_for_gkd.py): dataset processing logic for the GKD path

## Step 1: Process the Dataset

Purpose:
Convert DeepScaleR samples into chat-template prompts used by the student rollout path.

Script:
[01-process-dataset.sh](01-process-dataset.sh)

Arguments:

- `$1`: student tokenizer or model path
- `$2`: raw dataset JSON path

Example:

```bash
cd quickstarts/Qwen3-8B_GKD_Qwen3-32B
bash 01-process-dataset.sh ~/models/Qwen3-8B ~/data/deepscaler.json
```

What this script does:

- Runs [process_deepscaler_for_gkd.py](process_deepscaler_for_gkd.py)
- Applies the Qwen chat template with `enable_thinking=False`
- Writes the Hugging Face dataset to `./deepscaler_gkd`

## Step 2: Convert the Student and Teacher Models

Purpose:
Prepare both models for the distributed training path and for serving.

Script:
[02-convert-model.sh](02-convert-model.sh)

Arguments:

- `$1`: student model path
- `$2`: teacher model path

Example:

```bash
cd quickstarts/Qwen3-8B_GKD_Qwen3-32B
bash 02-convert-model.sh ~/models/Qwen3-8B ~/models/Qwen3-32B
```

What this script does:

- Creates HF symlinks:
	`../../hf_models/Qwen3-8B`
	`../../hf_models/Qwen3-32B`
- Converts the student model to Megatron format:
	`../../megatron_models/Qwen3-8B`
- Converts the teacher model to Megatron format:
	`../../megatron_models/Qwen3-32B`

## Step 3: Launch Online GKD Training

Purpose:
Run student training while using the teacher model as the reference/distillation source.

Script:
[03-train.sh](03-train.sh)

Example:

```bash
cd quickstarts/Qwen3-8B_GKD_Qwen3-32B
bash 03-train.sh
```

What this script runs:

```bash
python3 headquarters_v2.py --config ./quickstarts/Qwen3-8B_GKD_Qwen3-32B/config.yml
```

Note:
As in the GRPO quickstart, the script contains a reminder that you should define the distributed environment variables such as `NODE_RANK` and `NODE_LIST` before launching multi-node training.

Important configuration items in [config.yml](config.yml):

- `training.output_dir`: `./quickstarts/Qwen3-8B_GKD_Qwen3-32B/training_outputs`
- `training.from_pretrained`: `./megatron_models/Qwen3-8B`
- `training.max_steps`: `628`
- `training.rollout_batch_size`: `128`
- `training.global_batch_size`: `128`
- `training.n_samples`: `1`
- `distillation.enabled`: `True`
- `reference.load_path`: `./megatron_models/Qwen3-32B`
- `distillation.logits_express.batch_size`: `8`
- `distillation.gkd.gkd_sparse_topk_enabled`: `True`
- `distillation.gkd.gkd_topk`: `200`
- `actor.tp`: `2`
- `actor.pp`: `2`
- `reference.tp`: `8`
- `reference.pp`: `1`
- `async_rollout.data`: `./quickstarts/Qwen3-8B_GKD_Qwen3-32B/deepscaler_gkd`

These settings mean this quickstart is configured for online distillation with a separate teacher model, sparse top-k GKD enabled, and single-sample rollout generation per prompt.

## Step 4: Convert the Final Student Checkpoint Back to Hugging Face Format

Purpose:
Export the student checkpoint after distillation training.

Script:
[04-convert-result.sh](04-convert-result.sh)

Arguments:

- `$1`: original student base model path

Example:

```bash
cd quickstarts/Qwen3-8B_GKD_Qwen3-32B
bash 04-convert-result.sh ~/models/Qwen3-8B
```

Output:

- Final Hugging Face student checkpoint:
	`./training_outputs/hf_checkpoints/step_600`

## Where to Check Training Progress

- Student training logs:
	`./training_outputs/actor_log`
- Teacher/reference logs:
	`./training_outputs/ref_log`
- Saved Megatron checkpoints:
	`./training_outputs/checkpoints`
- Final exported student checkpoint:
	`./training_outputs/hf_checkpoints/step_600`

Typical TensorBoard location, if enabled by your runtime environment, will still be under the `training_outputs` tree created by the training stack.

## Training Results Reference

<img src="./result.png" alt="result" style="zoom:60%;" />

## Minimal End-to-End Example

```bash
cd quickstarts/Qwen3-8B_GKD_Qwen3-32B
bash 01-process-dataset.sh ~/models/Qwen3-8B ~/data/deepscaler.json
bash 02-convert-model.sh ~/models/Qwen3-8B ~/models/Qwen3-32B
bash 03-train.sh
bash 04-convert-result.sh ~/models/Qwen3-8B
```

## Summary

Use this path when you want online teacher-student distillation rather than plain SFT or standard RL. Prepare both student and teacher models before training, then follow the four scripts in order.