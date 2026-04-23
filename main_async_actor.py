import argparse
import dataclasses
import logging
import os

import torch.distributed as dist

from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.training.global_vars import get_args
from megatron.training.training import get_model, get_optimizer_param_scheduler
from megatron.training.checkpointing import load_checkpoint
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from pretrain_gpt import model_provider
from light_scale.config import (
    AlgorithmConfig,
    AsyncRolloutConfig,
    DistributedLockServerConfig,
    LogitsExpressConfig,
    GKDConfig,
    ReferenceModelConfig,
    CheckpointSavingConfig,
)
from light_scale.config_utils import create_parser_from_dataclass
from light_scale.logger_utils import setup_logger_v2_main_process
from light_scale.sync_processor import ActorReferenceDataUpdater
from light_scale.logits_express import LogitsExpress
from light_scale.distributed_lock import LockServerProcess
from light_scale import sandbox_fusion_utils
from light_scale.async_grpo_trainer import GRPOTrainer
import torch


def args_provider(parser: argparse.ArgumentParser):
    async_rollout_group = parser.add_argument_group("Async Rollout Config")
    create_parser_from_dataclass(AsyncRolloutConfig, async_rollout_group)

    reference_service_group = parser.add_argument_group("Reference Service Config")
    create_parser_from_dataclass(ReferenceModelConfig, reference_service_group)

    algorithm_config_group = parser.add_argument_group("Algorithm")
    create_parser_from_dataclass(AlgorithmConfig, algorithm_config_group)

    dist_lock_group = parser.add_argument_group("Distributed Lock")
    create_parser_from_dataclass(DistributedLockServerConfig, dist_lock_group)

    checkpoint_group = parser.add_argument_group("Checkpoint Saving")
    create_parser_from_dataclass(CheckpointSavingConfig, checkpoint_group)

    logits_express_group = parser.add_argument_group("Logits Express Config")
    create_parser_from_dataclass(LogitsExpressConfig, logits_express_group)

    gkd_group = parser.add_argument_group("GKD Config")
    create_parser_from_dataclass(GKDConfig, gkd_group)

    parser.add_argument("--n_samples", type=int, default=8)
    return parser


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
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    initialize_megatron(
        args_defaults={"variable_seq_lengths": True, "moe_token_dispatcher_type": "alltoall"},
        extra_args_provider=args_provider,
    )
    margs = get_args()

    log_level_name = getattr(margs, "light_scale_log_level", getattr(margs, "mrl_log_level", "info"))
    log_level = getattr(logging, str(log_level_name).upper(), logging.INFO)
    logger = setup_logger_v2_main_process("light_scale", level=log_level)

    set_jit_fusion_options()
    logger.info("initialize completed")

    if not margs.pad_to_max_length:
        assert margs.tp_comm_overlap is False, "tp comm bucket will determine seq length by args.seq_length"
        margs.variable_seq_lengths = True
    else:
        margs.variable_seq_lengths = False
    margs.moe_token_dispatcher_type = "alltoall"

    if margs.force_thinking:
        assert margs.begin_of_thinking is not None

    dist.barrier()

    logger.info("setup_model_and_optimizer")
    model, optim, scheduler = setup_model_and_optimizer()
    dist.barrier()

    logger.info("loading checkpoint")
    passed_iters, _ = load_checkpoint(model, optim, scheduler)
    logger.info("checkpoint loaded, %s iters passed", passed_iters)
    dist.barrier()

    dist_lock_process = None
    if dist.get_rank() == 0:
        dist_lock_process = LockServerProcess(
            host="0.0.0.0",
            port=margs.dist_lock_server_port,
        )
        dist_lock_process.start_lock_server()

    if dist.get_rank() == 0:
        if margs.sandbox_fusion_hostfile is not None:
            margs.sandbox_fusion_urls = sandbox_fusion_utils.init_sandbox_fusion_urls_from_hostfile(
                margs.sandbox_fusion_hostfile
            )
            logger.info("initialized %s sandbox fusion urls", len(margs.sandbox_fusion_urls))
        else:
            margs.sandbox_fusion_urls = None
    dist.barrier()

    data_updater = None
    if dist.get_rank() == 0 and (margs.init_kl_coef > 1e-8 or margs.distillation_enabled):
        assert margs.reference_dp_size is not None and margs.reference_dp_size > 0
        logger.info("initializing actor/reference data updater")
        data_updater = ActorReferenceDataUpdater(
            actor_master_addr=os.environ.get("MASTER_ADDR"),
            update_group_port=os.environ.get("MASTER_PORT"),
            is_actor=True,
            timeout_minutes=margs.distributed_timeout_minutes,
        )

    logits_express = None
    if margs.distillation_enabled:
        logits_express = LogitsExpress(data_updater, is_teacher=False)

    trainer = GRPOTrainer(
        passed_iters,
        model,
        optim,
        scheduler,
        data_updater,
        logits_express,
    )
    logger.info("trainer initialized")
    dist.barrier()

    trainer.train()
    dist.barrier()

    if dist_lock_process is not None:
        dist_lock_process.shutdown_lock_server()



if __name__ == "__main__":
    main()