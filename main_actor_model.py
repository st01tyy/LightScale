import dataclasses
from megatron.core import mpu
import argparse
from light_scale.config import (
    RolloutServiceConfig,
    ReferenceModelConfig,
    AlgorithmConfig,
    DistributedLockServerConfig,
    LogitsExpressConfig,
    GKDConfig,
)
from light_scale.config_utils import create_parser_from_dataclass
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
import torch.distributed as dist
from megatron.training.training import get_model, get_optimizer_param_scheduler
from pretrain_gpt import model_provider
from megatron.training.checkpointing import load_checkpoint
from light_scale.logger_utils import setup_logger
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from light_scale.sync_processor import SGLangSaver, ActorReferenceDataUpdater
# from light_scale.weight_utils import WeightUpdater
from light_scale.weight_utils_v2 import DenseWeightUpdater, MoeWeightUpdater
import requests
from requests.exceptions import Timeout, RequestException
import time
from light_scale.grpo_trainer import GRPOTrainer
from light_scale.llm_caller import wait_until_rollout_and_ref_server_ready
import os
import logging
from megatron.training.initialize import set_jit_fusion_options
from light_scale import sandbox_fusion_utils
from light_scale.distributed_lock import LockServerProcess, DistributedLock
from light_scale import dist_utils
from light_scale.logits_express import LogitsExpress

def args_provider(parser: argparse.ArgumentParser):
    actor_service_group = parser.add_argument_group("Actor Inference Service Config")
    create_parser_from_dataclass(RolloutServiceConfig, actor_service_group)

    reference_service_group = parser.add_argument_group("Reference Service Config")
    create_parser_from_dataclass(ReferenceModelConfig, reference_service_group)

    algorithm_config_group = parser.add_argument_group("Algorithm")
    create_parser_from_dataclass(AlgorithmConfig, algorithm_config_group)

    dist_lock_group = parser.add_argument_group("Distributed Lock")
    create_parser_from_dataclass(DistributedLockServerConfig, dist_lock_group)

    # Distillation-related configs
    logits_express_group = parser.add_argument_group("Logits Express Config")
    create_parser_from_dataclass(LogitsExpressConfig, logits_express_group)

    gkd_group = parser.add_argument_group("GKD Config")
    create_parser_from_dataclass(GKDConfig, gkd_group)

    return parser

def parse_configs(args):
    # 解析到 dataclass
    rollout_service_config = RolloutServiceConfig(**{k: v for k, v in vars(args).items() if k in RolloutServiceConfig.__annotations__})
    reference_config = ReferenceModelConfig(**{k: v for k, v in vars(args).items() if k in ReferenceModelConfig.__annotations__})

    return rollout_service_config, reference_config

def setup_model_and_optimizer():
    model = get_model(model_provider)
    kwargs = {}
    args = get_args()
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    optimizer = get_megatron_optimizer(config, model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    return model, optimizer, opt_param_scheduler

def main():
    # 加载base模型时，命令行传参："no_load_rng": True，续训时不用s
    initialize_megatron(args_defaults={"variable_seq_lengths": True, "moe_token_dispatcher_type": "alltoall"}, extra_args_provider=args_provider)
    margs = get_args()
    # TODO: 检查margs配置项，只支持transformer engine，moe必须group gemm，不支持共享专家
    log_level_name = getattr(margs, "light_scale_log_level", getattr(margs, "mrl_log_level", "info"))
    logger = setup_logger("light_scale", level=logging.DEBUG if log_level_name == "debug" else logging.INFO)
    if dist.get_rank() == 0:
        logger.warning(
            "main_actor_model.py is deprecated and will be removed in a future release. "
            "Use main_async_actor.py for new GRPO training jobs."
        )
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()
    logger.info("initialize completed")

    # 两个选项在初始化时被写死，需要手动设置，传参无效
    if not margs.pad_to_max_length:
        assert margs.tp_comm_overlap is False, "tp comm bucket will determine seq length by args.seq_length"
        margs.variable_seq_lengths = True
    else:
        margs.variable_seq_lengths = False
    margs.moe_token_dispatcher_type = "alltoall"

    if margs.force_thinking:
        assert margs.begin_of_thinking is not None
    dist.barrier()

    # 初始化分布式锁服务器
    dist_lock_process = None
    if dist.get_rank() == 0:
        dist_lock_process = LockServerProcess(
            host='0.0.0.0',
            port=margs.dist_lock_server_port
        )
        dist_lock_process.start_lock_server()

    # 初始化代码沙盒
    if dist.get_rank() == 0:
        if margs.sandbox_fusion_hostfile is not None:
            margs.sandbox_fusion_urls = sandbox_fusion_utils.init_sandbox_fusion_urls_from_hostfile(margs.sandbox_fusion_hostfile)
            logger.info(f"initialized {len(margs.sandbox_fusion_urls)} sandbox fusion urls")
        else:
            margs.sandbox_fusion_urls = None
    dist.barrier()

    logger.info("parsing configs")
    rollout_service_config, reference_config = parse_configs(margs)

    logger.info("setup_model_and_optimizer")
    model, optim, scheduler = setup_model_and_optimizer()
    dist.barrier()

    # Specify --no-load-optim or --finetune to prevent attempting to load the optimizer state
    logger.info("loading checkpoint")
    passed_iters, _ = load_checkpoint(model, optim, scheduler)
    logger.info(f"checkpoint loaded, {passed_iters} iters passed")
    dist.barrier()

    logger.info(f"seq_length: {margs.seq_length}")

    data_updater = None
    if dist.get_rank() == 0 and (margs.init_kl_coef > 1e-8 or margs.distillation_enabled):
        assert margs.reference_dp_size is not None and margs.reference_dp_size > 0
        logger.info("initializing actor/reference data updater")
        data_updater = ActorReferenceDataUpdater(
            actor_master_addr=os.environ.get("MASTER_ADDR"),
            update_group_port=os.environ.get("MASTER_PORT"),
            is_actor=True,
            timeout_minutes=margs.distributed_timeout_minutes
        )

    logits_express = None
    if margs.distillation_enabled:
        logits_express = LogitsExpress(data_updater, is_teacher=False)

    # 等待rollout和ref server就绪
    if dist.get_rank() == 0:
        wait_until_rollout_and_ref_server_ready(rollout_service_config, reference_config, retry=100)
    dist.barrier()
    logger.info("rollout and ref server is ready")

    logger.info("initializing weight updater")
    sync_processor = None
    dense_weight_updater = None
    moe_weight_updater = None
    if dist_utils.is_pp_src_rank():
        dist_lock = DistributedLock(
            "weight_update",
            host=os.environ['MASTER_ADDR'],
            port=margs.dist_lock_server_port
        )
        sync_processor = SGLangSaver(
            rollout_service_config.rollout_base_url_list,
            rollout_service_config.server_world_size,
            mpu.get_pipeline_model_parallel_rank(),
            dist_lock,
            rollout_service_config.weight_update_group_port + int(mpu.get_pipeline_model_parallel_rank())
        )
    dist.barrier()
    if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        dense_weight_updater = DenseWeightUpdater(model, sync_processor)
    if margs.num_experts is not None and margs.num_experts > 0 and mpu.get_expert_data_parallel_rank() == 0:
        moe_weight_updater = MoeWeightUpdater(model, sync_processor)
    dist.barrier()
    logger.info("initialized weight updater")

    if passed_iters > 0:
        if dense_weight_updater is not None:
            logger.info("updating dense model weight to inference service")
            dense_weight_updater()
        dist.barrier()
        if moe_weight_updater is not None:
            logger.info("updating moe model weight to inference service")
            moe_weight_updater()
        dist.barrier()

    trainer = GRPOTrainer(passed_iters, model, optim, scheduler, dense_weight_updater, moe_weight_updater, data_updater, logits_express)
    logger.info("trainer initialized")
    dist.barrier()

    trainer.train()
    dist.barrier()
    
    if dist_lock_process is not None:
        dist_lock_process.shutdown_lock_server()
    logger.info("done")

if __name__ == '__main__':
    main()