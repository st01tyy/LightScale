# 除Megatron已有配置外，各组件配置类

import argparse
import dataclasses
from typing import List, Optional, Type, TypeVar
from dataclasses import dataclass, field

@dataclass
class AlgorithmConfig:
    rollout_batch_size: int = field(default=128, metadata={"required": False})
    advantage_estimator: str = field(default="grpo", metadata={"required": False})
    init_kl_coef: float = field(default=0.0, metadata={"required": False})
    use_kl_loss: bool = field(default=False, metadata={"required": False})
    kl_estimator: str = field(default="k1", metadata={"required": False})
    num_repeat_times: int = field(default=1, metadata={"required": False})
    clip_eps: float = field(default=0.2, metadata={"required": False})
    dump_experience: bool = field(default=False, metadata={"required": False})
    dump_path: str = field(default=None, metadata={"required": False})
    dump_tensors: bool = field(default=False, metadata={"required": False})
    use_cot_reward: bool = field(default=False, metadata={"required": False})
    force_thinking: bool = field(default=False, metadata={"required": False})
    begin_of_thinking: str = field(default=None, metadata={"required": False})
    use_outcome_rewards_as_advantages: bool = field(default=False, metadata={"required": False})
    micro_forward_batch_size: int = field(default=1, metadata={"required": False})
    light_scale_log_level: str = field(default="info", metadata={"required": False, "aliases": ["mrl_log_level"]})
    early_stop_steps: int = field(default=-1, metadata={"required": False})
    pad_to_max_length: bool = field(default=False, metadata={"required": False})
    skip_zero_reward_sample: bool = field(default=False, metadata={"required": False})
    mlp_weight_merging_batch_size: int = field(default=0, metadata={"required": False})
    policy_loss_limit: float = field(default=None, metadata={"required": False})
    kl_loss_limit: float = field(default=None, metadata={"required": False})
    total_world_size: int = field(default=None, metadata={"required": False})
    entropy_loss_coef: float = field(default=0.0, metadata={"required": False})
    sandbox_fusion_hostfile: str = field(default=None, metadata={"required": False})
    entropy_loss_threshold: float = field(default=0.3, metadata={"required": False})
    moe_weight_merging_layer_batch_size: int = field(default=0, metadata={"required": False})
    moe_weight_merging_expert_batch_size: int = field(default=0, metadata={"required": False})
    policy_loss_coef: float = field(default=1.0, metadata={"required": False})
    negate_kl_loss: bool = field(default=False, metadata={"required": False})
    distillation_enabled: bool = field(default=False, metadata={"required": False})

@dataclass
class RolloutServiceConfig:
    rollout_base_url_list: List[str]
    # weight_update_url: str
    rollout_model_name: str
    server_world_size: int
    weight_update_group_port: int = field(default=65500, metadata={"required": False})
    num_workers: int = field(default=1000, metadata={"required": False})
    max_tokens: int = field(default=100, metadata={"required": False})
    stop: str = field(default=None, metadata={"required": False})
    add_stop: bool = field(default=False, metadata={"required": False})
    sampling_temperature: float = field(default=1.0, metadata={"required": False})
    sampling_top_k: int = field(default=-1, metadata={"required": False})
    sampling_top_p: float = field(default=1.0, metadata={"required": False})
    sampling_presence_penalty: float = field(default=1.0, metadata={"required": False})
    n_samples: int = field(default=8, metadata={"required": False})
    sampling_pool_size: int = field(default=0, metadata={"required": False})

@dataclass
class ReferenceModelConfig:
    reference_service_url: str = field(default=None, metadata={"required": False})
    reference_dp_size: int = field(default=None, metadata={"required": False})

@dataclass
class ReferenceModelServingConfig:
    actor_master_addr: str
    data_transfer_group_port: int = field(default=65400, metadata={"required": False})
    distillation_enabled: bool = field(default=False, metadata={"required": False})
    light_scale_log_level: str = field(default="info", metadata={"required": False, "aliases": ["mrl_log_level"]})

@dataclass
class BenchmarkConfig:
    inference_only: bool = field(default=False, metadata={"required": False})
    do_optimize: bool = field(default=False, metadata={"required": False})

@dataclass
class WeightUpdaterTestConfig:
    online_test: bool = field(default=False, metadata={"required": False})
    safetensors_save_path: str = field(default=None, metadata={"required": False})
    mlp_weight_merging_batch_size: int = field(default=0, metadata={"required": False})
    rollout_base_url_list: List[str] = field(default=None, metadata={"required": False})
    rollout_model_name: str = field(default=None, metadata={"required": False})
    server_world_size: int = field(default=None, metadata={"required": False})
    weight_update_group_port: int = field(default=65500, metadata={"required": False})
    moe_weight_merging_layer_batch_size: int = field(default=0, metadata={"required": False})
    moe_weight_merging_expert_batch_size: int = field(default=0, metadata={"required": False})

@dataclass
class DistributedLockServerConfig:
    dist_lock_server_port: int = field(default=13299, metadata={"required": False})

@dataclass
class LogitsExpressConfig:
    logits_dtype: str = field(default="fp32", metadata={"required": False})
    logits_transfer_batch_size: int = field(default=4, metadata={"required": False})
    logits_pg_base_port: int = field(default=45000, metadata={"required": False})
    gkd_sparse_topk_enabled: bool = field(default=False, metadata={"required": False})
    gkd_topk: int = field(default=128, metadata={"required": False})
    topk_sentinel_index: int = field(default=-1, metadata={"required": False})
    topk_sentinel_value: float = field(default=-1e9, metadata={"required": False})

@dataclass
class GKDConfig:
    student_temperature: float = field(default=1.0, metadata={"required": False})
    teacher_temperature: float = field(default=1.0, metadata={"required": False})
    gkd_beta: float = field(default=0.0, metadata={"required": False})
    seq_kd: bool = field(default=False, metadata={"required": False})

@dataclass
class CheckpointSavingConfig:
    save_test_step: int = field(default=None, metadata={"required": False})

@dataclass
class SFTConfig:
    ignore_token_id: int = field(default=-100, metadata={"required": False})
    sequence_packing: bool = field(default=False, metadata={"required": False})
    sft_data_shuffle: bool = field(default=False, metadata={"required": False})
    sft_token_mean: bool = field(default=True, metadata={"required": False})
    early_stop_steps: int = field(default=None, metadata={"required": False})
    light_scale_log_level: str = field(default="info", metadata={"required": False, "aliases": ["mrl_log_level"]})

@dataclass
class AsyncRolloutConfig:
    async_rollout_cfg_path: str = field(default=None, metadata={"required": True})
    weight_update_group_port: int = field(default=65500, metadata={"required": False})
    max_train_batch_size: int = field(default=-1, metadata={"required": False})

T = TypeVar("T")

def create_parser_from_dataclass(cls: Type[T], parser: argparse.ArgumentParser) -> None:
    """根据 dataclass 自动向 ArgumentParser 添加参数"""
    for field in dataclasses.fields(cls):
        arg_name = f"--{field.name}"
        arg_type = field.type

        is_optional = not field.metadata.get("required", True)

        kwargs = {"required": not is_optional}

        # 处理 List[T] 和 Optional[List[T]]
        if hasattr(arg_type, "__origin__") and arg_type.__origin__ is list:
            element_type = arg_type.__args__[0]  # 获取 List 内的元素类型
            kwargs["type"] = element_type
            kwargs["nargs"] = "+"  # 允许多个值
        else:
            kwargs["type"] = arg_type

        # 处理默认值
        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs["default"] = field.default_factory()
        
        if arg_type == bool:
            kwargs["action"] = "store_true"

        print(f"{arg_name}, {kwargs}")
        parser.add_argument(arg_name, **kwargs)

def args_provider(parser: argparse.ArgumentParser):
    actor_service_group = parser.add_argument_group("Actor Inference Service Config")
    create_parser_from_dataclass(RolloutServiceConfig, actor_service_group)

    reference_service_group = parser.add_argument_group("Reference Service Config")
    create_parser_from_dataclass(ReferenceModelConfig, reference_service_group)

    training_group = parser.add_argument_group("Training Config")
    create_parser_from_dataclass(AlgorithmConfig, training_group)

    return parser

def parse_configs(args):
    # 解析到 dataclass
    rollout_service_config = RolloutServiceConfig(**{k: v for k, v in vars(args).items() if k in RolloutServiceConfig.__annotations__})
    reference_config = ReferenceModelConfig(**{k: v for k, v in vars(args).items() if k in ReferenceModelConfig.__annotations__})
    training_config = AlgorithmConfig(**{k: v for k, v in vars(args).items() if k in AlgorithmConfig.__annotations__})

    return rollout_service_config, reference_config, training_config
    