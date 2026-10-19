# Qwen3-8B SFT 快速开始

这份 quickstart 用来说明如何使用 LiteScale 在 MetaMathQA 数据集上对 Qwen3-8B-Base 做监督微调。

## 硬件要求

- 已验证环境：H800 80GB
- 建议最少卡数：8 卡

## 目标

- 数据集：MetaMathQA-395K
- 基座模型：Qwen3-8B-Base
- 训练类型：SFT
- 产出内容：训练中的 Megatron 格式 checkpoint，以及转换后的 Hugging Face checkpoint

## 开始前需要准备什么

先准备数据集：

```bash
wget https://modelscope.cn/datasets/swift/MetaMathQA/resolve/master/MetaMathQA-395K.json
```

再下载千问3 8B Base模型：

```bash
git clone https://www.modelscope.cn/Qwen/Qwen3-8B-Base.git
```

接下来的操作示例默认你将数据和模型保存到下方位置：

- 仓库根目录：`Megatron-RL-Dumped`
- 数据集文件：`~/data/MetaMathQA-395K.json`
- 基础模型目录：`~/models/Qwen3-8B-Base`

## 目录文件说明

- [config.yml](config.yml)：`headquarters.py` 使用的训练配置
- [01-process-dataset.sh](01-process-dataset.sh)：把原始 MetaMathQA JSON 处理成可训练的数据集
- [02-convert-model.sh](02-convert-model.sh)：把基础模型转换成 Megatron 格式
- [03-train.sh](03-train.sh)：启动 SFT 训练
- [04-convert-result.sh](04-convert-result.sh)：把最终 Megatron checkpoint 转回 Hugging Face 格式
- [convert_metamathqa_to_qwen3.py](convert_metamathqa_to_qwen3.py)：数据转换逻辑

## 第一步：处理数据集

目的：
把 MetaMathQA 转成千问3模型的对话模板格式，再进一步打包成固定长度的训练样本。

脚本：
[01-process-dataset.sh](01-process-dataset.sh)

参数说明：

- `$1`：tokenizer 或模型路径
- `$2`：原始数据集 JSON 路径

示例：

```bash
cd quickstarts/Qwen3-8B_SFT
bash 01-process-dataset.sh ~/models/Qwen3-8B-Base ~/data/MetaMathQA-395K.json
```

这个脚本会做几件事：

- 调用 [convert_metamathqa_to_qwen3.py](convert_metamathqa_to_qwen3.py) 完成样本格式转换和分词
- 将中间结果写到 `./MetaMathQA_Qwen3`
- 调用 `tools/pack_hf_datasets.py` 生成 8k 长度的 packed 数据
- 最终产出配置里使用的数据目录：
  `./MetaMathQA_Qwen3_packed_shuffle_mode_0_8k_seed_42_bs_4`

这里几个关键参数值得注意：

- `--target-length-in-k 8`：把样本打包到 8k token 长度
- `--pad-token-id 151643`：Qwen3 使用的 padding token id
- `--batch-size 4`：数据打包时的 batch size

## 第二步：把基础模型转换成 Megatron 格式

目的：
把 Hugging Face 版本的 Qwen3-8B-Base 转成 LiteScale 训练所需的 Megatron checkpoint。

脚本：
[02-convert-model.sh](02-convert-model.sh)

参数说明：

- `$1`：基础模型路径

示例：

```bash
cd quickstarts/Qwen3-8B_SFT
bash 02-convert-model.sh ~/models/Qwen3-8B-Base
```

输出位置：

- Megatron checkpoint 目录：
  `../../megatron_models/Qwen3-8B-Base`

## 第三步：启动 SFT 训练

脚本：
[03-train.sh](03-train.sh)

示例：

```bash
cd quickstarts/Qwen3-8B_SFT
bash 03-train.sh
```

实际执行的命令是：

```bash
python3 headquarters.py --config ./quickstarts/Qwen3-8B_SFT/config.yml
```

[config.yml](config.yml) 里比较重要的配置包括：

- `training.output_dir`：`./quickstarts/Qwen3-8B_SFT/training_outputs`
- `training.from_pretrained`：`./megatron_models/Qwen3-8B-Base`
- `training.data`：打包后的 MetaMathQA 数据目录
- `training.max_steps`：`9257`
- `training.global_batch_size`：`32`
- `training.micro_batch_size`：`1`
- `training.max_length`：`8192`
- `training.sequence_packing`：`True`
- `actor.tp`：`1`
- `actor.pp`：`1`
- `actor.dp`：`8`

这些配置对应的是一条典型的 8k packed SFT 路径，主要使用数据并行，不额外切分张量并行和流水并行。

## 第四步：把最终结果转回 Hugging Face 格式

目的：
把训练产出的 Megatron checkpoint 导出成 Hugging Face 格式，便于后续推理或评测。

脚本：
[04-convert-result.sh](04-convert-result.sh)

参数说明：

- `$1`：原始基础模型路径，转换时会作为参考基座

示例：

```bash
cd quickstarts/Qwen3-8B_SFT
bash 04-convert-result.sh ~/models/Qwen3-8B-Base
```

输出位置：

- 最终 Hugging Face checkpoint：
  `./training_outputs/hf_checkpoints/step_9252`

## 到哪里看训练进度

- 主训练日志：
  `./training_outputs/train_log/rank_0.log`
- TensorBoard 日志目录：
  `./training_outputs/tensorboard_log`
- 训练过程中保存的 Megatron checkpoints：
  `./training_outputs/checkpoints`

常见的 TensorBoard 查看方式：

```bash
tensorboard --logdir quickstarts/Qwen3-8B_SFT/training_outputs/tensorboard_log
```

## 训练结果参考

<img src="./result.png" alt="loss" style="zoom: 60%;" />

## 最小跑通示例

```bash
cd quickstarts/Qwen3-8B_SFT
bash 01-process-dataset.sh ~/models/Qwen3-8B-Base ~/data/MetaMathQA-395K.json
bash 02-convert-model.sh ~/models/Qwen3-8B-Base
bash 03-train.sh
bash 04-convert-result.sh ~/models/Qwen3-8B-Base
```

## 一句话总结

如果你想先用最短路径把 LiteScale 跑起来，SFT 是最合适的入口：一套数据、一个基座模型、没有在线 rollout，也不涉及 reward 建模。