# LiteScale

**LiteScale 是一个基于 Megatron 和 SGLang 构建的分布式大语言模型后训练框架。避开臃肿的重量级编排层，旨在为大模型研究人员提供一套轻量、可扩展的后训练技术栈。**

目前，LiteScale 主要实现后训练三大主要工作流：GKD在线知识蒸馏、 GRPO强化学习，以及监督微调（SFT）。

**核心设计理念：** 训练链路坚持 Megatron 原生，推理采样链路彻底解耦且高度可插拔。系统极致轻量，无需引入类似 Ray 控制面的复杂机制，从而大幅降低调试和扩展的门槛。

![LiteScale整体架构](./docs/figure1_online_rl_gkd_architecture.png)

## 为什么使用LiteScale

- **统一的后训练架构**：使用同一套配置体系无缝切换 GKD、GRPO 和 SFT 任务。
- **分布式GKD蒸馏训练**：分别实现了分布式 Top-k Forward KL、JSD 和 Reverse KL 计算。
- **去 Ray 化设计**：告别繁重的编排开销，大幅降低调试与自定义门槛。
- **异步在线训练**：实现了Megatron多个前-后向训练步的梯度累积（Gradient Accumulation）。
- **训练与推理彻底解耦**：基于 SGLang 推理服务构建异步 Rollout 栈，实现 Train-Serve 真正分离。
- **灵活的 Rollout 定制机制**：提供可插拔的 Service 和 Worker，轻松适配各种垂直领域的后训练数据流。

## 核心能力

### 1. 分布式GKD在线知识蒸馏
LiteScale设计出分布式师生高速Logits传输，实现了三种训练目标的分布式计算方法，完整复现了谷歌提出的[GKD蒸馏训练策略](https://arxiv.org/pdf/2306.13649)：
- Top-k Forward KL
- JSD
- Reverse KL

此方案专为大规模训练场景设计：有效避免了全词表通信带来的高昂开销。

### 2. 异步GRPO强化学习训练
LiteScale 支持基于多步梯度累积的在线强化学习：

- **模块化数据采样结构设计**：推理、评判、验证以及 Worker 侧数据处理都可以作为独立模块自由组合，而不需要硬编码进单体 Trainer。
- **易于自定义扩展**：可以按领域需求替换或扩展 services 和 workers，快速适配不同后训练工作流。
- **精度无损的在线异步训练**：基于 Megatron 的多步梯度累积能力，在扩展到大 batch 和分布式部署时，仍能保持数值正确性、优化器语义和训练效率的稳定。

### 3. 监督微调 (SFT)

LiteScale支持序列Packing方式的监督微调训练

## 环境安装

LiteScale 可以通过 [build.dockerfile](build.dockerfile) 构建镜像后直接使用。这是更推荐的安装方式，因为它可以一次性准备好 SGLang、CUDA 相关依赖以及常用训练环境，便于复现和部署。

示例构建命令：

```bash
docker build -f build.dockerfile -t litescale:latest .
```

镜像构建完成后，你可以在此基础上启动自己的容器环境，用于训练、rollout service 或调试。

主要环境包版本：

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

## 快速开始

### 在 MetaMathQA 数据集上对Qwen3-8B-Base进行监督微调

- Quickstart 目录：[quickstarts/Qwen3-8B_SFT](quickstarts/Qwen3-8B_SFT)
- 配置文件：[quickstarts/Qwen3-8B_SFT/config.yml](quickstarts/Qwen3-8B_SFT/config.yml)
- 数据集：MetaMathQA-395K
- 模型：Qwen3-8B-Base
- 具体步骤参考 [quickstarts/Qwen3-8B_SFT/README.md](quickstarts/Qwen3-8B_SFT/README.md)

### 在 DeepScaleR 数据集上对Qwen3-8B-Base进行异步 GRPO 训练

- Quickstart 目录：[quickstarts/Qwen3-8B_GRPO](quickstarts/Qwen3-8B_GRPO)
- 配置文件：[quickstarts/Qwen3-8B_GRPO/config.yml](quickstarts/Qwen3-8B_GRPO/config.yml)
- 数据集：DeepScaleR
- 模型：Qwen3-8B-Base
- 训练类型：GRPO
- 具体步骤参考 [quickstarts/Qwen3-8B_GRPO/README.md](quickstarts/Qwen3-8B_GRPO/README.md)

### 使用 Qwen3-32B 作为教师模型在 DeepScaleR 数据集上GKD蒸馏训练 Qwen3-8B

- Quickstart 目录：[quickstarts/Qwen3-8B_GKD_Qwen3-32B](quickstarts/Qwen3-8B_GKD_Qwen3-32B)
- 配置文件：[quickstarts/Qwen3-8B_GKD_Qwen3-32B/config.yml](quickstarts/Qwen3-8B_GKD_Qwen3-32B/config.yml)
- 数据集：DeepScaleR
- Student 模型：Qwen3-8B
- Teacher 模型：Qwen3-32B
- 具体步骤参考 [quickstarts/Qwen3-8B_GKD_Qwen3-32B/README.md](quickstarts/Qwen3-8B_GKD_Qwen3-32B/README.md)

## 配置总览

LiteScale 使用统一的 YAML 配置模型。从高层看，最重要的顶层配置段如下：

| Section | 作用 |
| --- | --- |
| `main` | 选择训练入口。 |
| `training` | 定义数据、checkpoint、优化和日志行为。 |
| `algorithm` | 定义 RL 相关优化行为，例如 KL 处理和 advantage 估计。 |
| `actor` | 定义 Megatron 并行方式和 actor 侧执行设置。 |
| `rollout` | 定义服务引擎和在线采样行为。 |
| `reference` | 定义可选的 reference model 路径，用于 KL 正则或与蒸馏相关的工作流。 |
| `distillation` | 启用并配置在线蒸馏特性。 |
| `gkd` | 通过 distillation block 配置温度等 GKD 专属超参数。 |
| `async_rollout` | 定义异步服务、worker 和 rollout 侧编排逻辑。 |

具体模板可参考：

- [examples/config_template.yaml](examples/config_template.yaml)
- [examples/sft_config_template.yaml](examples/sft_config_template.yaml)
- [examples/async_rl_example.yaml](examples/async_rl_example.yaml)

异步 RL 模板默认使用 `main_async_actor`。`main_actor_model` 已进入废弃兼容阶段，不再作为新任务推荐入口。

## 架构设计

LiteScale架构设计详见：[DESIGN.md](DESIGN.md)

## 自定义框架

LiteScale 更适合在子系统层面进行修改，而不仅仅依赖启动时参数进行定制。

### 定制 rollout services

异步 rollout 栈采用面向 service 的设计。如果你想扩展 serving 行为，可以查看：

- [light_scale/async_rollout_v2/services/base_service.py](light_scale/async_rollout_v2/services/base_service.py)
- [light_scale/async_rollout_v2/services/sglang_service.py](light_scale/async_rollout_v2/services/sglang_service.py)
- [light_scale/async_rollout_v2/services/rock_service.py](light_scale/async_rollout_v2/services/rock_service.py)

### 定制 rollout workers

如果你想让 rollout 阶段适配新的领域处理逻辑，可以查看：

- [light_scale/async_rollout_v2/workers/base_worker.py](light_scale/async_rollout_v2/workers/base_worker.py)
- [light_scale/async_rollout_v2/workers/math_worker.py](light_scale/async_rollout_v2/workers/math_worker.py)
- [light_scale/async_rollout_v2/workers/math_tool_worker.py](light_scale/async_rollout_v2/workers/math_tool_worker.py)
- [light_scale/async_rollout_v2/workers/function_call_worker.py](light_scale/async_rollout_v2/workers/function_call_worker.py)
- [light_scale/async_rollout_v2/workers/llm_judge_worker.py](light_scale/async_rollout_v2/workers/llm_judge_worker.py)
- [light_scale/async_rollout_v2/workers/rock_worker.py](light_scale/async_rollout_v2/workers/rock_worker.py)

### 定制打分与验证逻辑

如果你想扩展 reward 逻辑、打分逻辑或验证行为，可以查看：

- [light_scale/score_utils.py](light_scale/score_utils.py)
- [verifier](verifier)

### 定制训练内部实现

如果你想研究或扩展核心 trainer，可以查看：

- [light_scale/async_grpo_trainer.py](light_scale/async_grpo_trainer.py)
- [light_scale/sft_trainer.py](light_scale/sft_trainer.py)

## 项目声明

- 本项目用于大语言模型后训练的学习与交流，请自行评估使用风险，作者对因使用本项目产生的风险或损失不承担责任。
- 本项目由作者于中移九天公司开发：[https://jiutian.10086.cn/](https://jiutian.10086.cn/)
- 联系邮箱：tao_tyy@sina.com

## 许可证与致谢

LiteScale 构建于 Megatron 相关基础设施之上，并依赖于SGLang的相关接口能力。仓库的许可证条款及上游致谢信息请参见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
