# LiteScale

**Lightweight and Scalable Post-training: The Ray-Free, Debug-Friendly Alignment Stack with Megatron-native simplicity.**

LiteScale is a distributed post-training framework built on top of Megatron and SGLang. It is designed for research and training engineers who need a practical stack for large-model post-training without adding a heavyweight orchestration layer. The framework currently targets three core workloads: On-policy generalized knowledge distillation (GKD), GRPO-based reinforcement learning, and SFT.

LiteScale is organized around a simple idea: keep the training path Megatron-native, keep the serving path decoupled and replaceable, and keep the full system lightweight enough to debug and scale without Ray-style control-plane complexity.

![LiteScale整体架构](./docs/figure1_online_rl_gkd_architecture.png)

## Why LiteScale

- **Unified post-training stack** for GKD, GRPO, and SFT under one configuration model.
- **Distributed GKD training** with top-k forward KL, JSD, and reverse KL support.
- **Ray-free system design** for simpler debugging, lower orchestration overhead, and easier scale-up.
- **Async on-policy training** based on the implementation of gradient accumulation over multiple forward-backward steps.
- **Train-serve decoupling** through an async rollout stack built around SGLang-based inference services.
- **Flexible rollout customization** with pluggable services and workers for domain-specific post-training loops.

## Core Capabilities

### 1. Distributed GKD

LiteScale supports online teacher-student distillation inside the training loop instead of treating distillation as a separate offline preprocessing stage. The current implementation is built around distributed logit exchange and supports three practical loss choices:

LiteScale designed a distributed high-speed Logits transfer mechanism from teachers to students, implemented distributed computation methods for three training objectives, and fully reproduced the [GKD distillation training strategy](https://arxiv.org/pdf/2306.13649).

- Top-k forward KL
- JSD
- Reverse KL

This path is designed for large-scale training where full-vocabulary communication is often too expensive, and where online teacher signals must remain compatible with distributed Megatron execution.

### 2. GRPO with Async Train-Serve Decoupling

LiteScale supports GRPO-based post-training with a clear emphasis on asynchronous train-serve decoupling. 

- **Modular data sampling architecture**: compose inference, judging, verification, and worker-side processing as separate modules instead of hard-coding them into one trainer.
- **Easy customization and extension**: adapt the rollout pipeline to domain-specific workflows by replacing or extending services and workers.
- **Precision-preserving on-policy async training**: build on Megatron's multi-step gradient accumulation path to scale asynchronous online optimization without sacrificing numerical correctness, optimizer semantics, or training efficiency.

### 3. Supervised Fine-Tuning

LiteScale also supports SFT as the simplest path into the framework. This makes the stack useful not only for RL-style alignment but also for standard instruction tuning and distillation-assisted supervised training.

## Installation

LiteScale can be used through a prebuilt container environment created from [build.dockerfile](build.dockerfile). This is the recommended path when you want a reproducible runtime with SGLang, CUDA-related dependencies, and common training packages prepared in one image.

Example image build:

```bash
docker build -f build.dockerfile -t litescale:latest .
```

After the image is built, you can start your own container workflow on top of it for training, rollout services, or debugging.

Main package versions:

| Package | Version |
| --- | --- |
| Megatron-LM | 0.13.0 |
| SGLang | 0.4.10 |
| Pai-Megatron-Patch | 0.12.0 |
| transformers | 4.55.0 |
| datasets | 4.0.0 |
| flash-attention | 2.7.4 |
| pylatexenc | 2.10 |
| math-verify | 0.7.0 |

## Quickstart

### Supervised fine-tuning Qwen3-8B-Base on the MetaMathQA dataset

Use this path if you want the simplest end-to-end onboarding flow for LiteScale: supervised fine-tuning Qwen3-8B-Base on the MetaMathQA dataset.

- Quickstart directory: [quickstarts/Qwen3-8B_SFT](quickstarts/Qwen3-8B_SFT)
- Config: [quickstarts/Qwen3-8B_SFT/config.yml](quickstarts/Qwen3-8B_SFT/config.yml)
- Dataset: MetaMathQA-395K
- Model: Qwen3-8B-Base
- For dataset preparation, model conversion, example commands, and step-by-step execution details, see [quickstarts/Qwen3-8B_SFT/README.md](quickstarts/Qwen3-8B_SFT/README.md)

### GRPO training Qwen3-8B-Base on the DeepScaleR dataset with async rollout

Use this path if you want the standard online RL workflow: GRPO training Qwen3-8B-Base on the DeepScaleR dataset with async rollout and reward-driven actor updates.

- Quickstart directory: [quickstarts/Qwen3-8B_GRPO](quickstarts/Qwen3-8B_GRPO)
- Config: [quickstarts/Qwen3-8B_GRPO/config.yml](quickstarts/Qwen3-8B_GRPO/config.yml)
- Dataset: DeepScaleR-Preview-Dataset
- Model: Qwen3-8B-Base
- Training type: GRPO
- For dataset preparation, rollout setup, example commands, and step-by-step execution details, see [quickstarts/Qwen3-8B_GRPO/README.md](quickstarts/Qwen3-8B_GRPO/README.md)

### GKD training Qwen3-8B with Qwen3-32B as the teacher on the DeepScaleR dataset

Use this path if your goal is online teacher-student distillation: train a Qwen3-8B student on the DeepScaleR dataset with guidance from a Qwen3-32B teacher through LiteScale's GKD path.

- Quickstart directory: [quickstarts/Qwen3-8B_GKD_Qwen3-32B](quickstarts/Qwen3-8B_GKD_Qwen3-32B)
- Config: [quickstarts/Qwen3-8B_GKD_Qwen3-32B/config.yml](quickstarts/Qwen3-8B_GKD_Qwen3-32B/config.yml)
- Dataset: DeepScaleR-Preview-Dataset
- Student model: Qwen3-8B
- Teacher model: Qwen3-32B
- For dataset preparation, dual-model setup, example commands, and step-by-step execution details, see [quickstarts/Qwen3-8B_GKD_Qwen3-32B/README.md](quickstarts/Qwen3-8B_GKD_Qwen3-32B/README.md)

## Configuration Overview

LiteScale uses a unified YAML-driven configuration model. At a high level, the most important top-level sections are:

| Section | Purpose |
| --- | --- |
| `main` | Selects the training entrypoint. |
| `training` | Defines data, checkpoints, optimization, and logging behavior. |
| `algorithm` | Defines RL-specific optimization behavior such as KL handling and advantage estimation. |
| `actor` | Defines Megatron parallelism and actor-side execution settings. |
| `rollout` | Defines the serving engine and online sampling behavior. |
| `reference` | Defines the optional reference-model path for KL regularization or distillation-related workflows. |
| `distillation` | Enables and configures online distillation features. |
| `gkd` | Configures temperatures and GKD-specific hyperparameters through the distillation block. |
| `async_rollout` | Defines async services, workers, and rollout-side orchestration. |

For concrete templates, see:

- [examples/config_template.yaml](examples/config_template.yaml)
- [examples/sft_config_template.yaml](examples/sft_config_template.yaml)
- [examples/async_rl_example.yaml](examples/async_rl_example.yaml)

The async RL template targets `main_async_actor`. `main_actor_model` is deprecated and retained only for compatibility during migration.

## Customization

LiteScale is intended to be modified at the subsystem level rather than only through launch-time flags.

### Customize rollout services

The async rollout stack is service-oriented. To extend serving behavior, inspect:

- [light_scale/async_rollout_v2/services/base_service.py](light_scale/async_rollout_v2/services/base_service.py)
- [light_scale/async_rollout_v2/services/sglang_service.py](light_scale/async_rollout_v2/services/sglang_service.py)
- [light_scale/async_rollout_v2/services/rock_service.py](light_scale/async_rollout_v2/services/rock_service.py)

### Customize rollout workers

To adapt rollout-time processing to new domains, inspect:

- [light_scale/async_rollout_v2/workers/base_worker.py](light_scale/async_rollout_v2/workers/base_worker.py)
- [light_scale/async_rollout_v2/workers/math_worker.py](light_scale/async_rollout_v2/workers/math_worker.py)
- [light_scale/async_rollout_v2/workers/math_tool_worker.py](light_scale/async_rollout_v2/workers/math_tool_worker.py)
- [light_scale/async_rollout_v2/workers/function_call_worker.py](light_scale/async_rollout_v2/workers/function_call_worker.py)
- [light_scale/async_rollout_v2/workers/llm_judge_worker.py](light_scale/async_rollout_v2/workers/llm_judge_worker.py)
- [light_scale/async_rollout_v2/workers/rock_worker.py](light_scale/async_rollout_v2/workers/rock_worker.py)

### Customize scoring and verification

To extend reward logic, scoring, or verification behavior, inspect:

- [light_scale/score_utils.py](light_scale/score_utils.py)
- [verifier](verifier)

### Customize training internals

To study or extend the core trainers, inspect:

- [light_scale/async_grpo_trainer.py](light_scale/async_grpo_trainer.py)
- [light_scale/sft_trainer.py](light_scale/sft_trainer.py)

## Project Notice

- This project is intended for learning and technical exchange around large language model post-training. Please evaluate usage risks yourself. The authors disclaim liability for risks or losses arising from use of this project.
- This project was developed within the China Mobile Jiutian: [https://jiutian.10086.cn/](https://jiutian.10086.cn/)
- Contact email: tao_tyy@sina.com

## License and Acknowledgements

LiteScale builds on Megatron-related infrastructure and integrates with SGLang-based serving workflows. The repository uses Apache-2.0 for original LiteScale code while preserving upstream notices for derived code. See [LICENSE](LICENSE) for the main license text and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and redistribution notes.