import dataclasses
import logging
import argparse

import torch.distributed as dist

from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import get_args
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.training.training import get_model, get_optimizer_param_scheduler
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

from light_scale.logger_utils import setup_logger
from light_scale.sft_trainer import SFTTrainer
from pretrain_gpt import model_provider
from light_scale.config import (
    SFTConfig,
    CheckpointSavingConfig
)
from light_scale.config_utils import create_parser_from_dataclass

def args_provider(parser: argparse.ArgumentParser):
    checkpoint_saving_group = parser.add_argument_group("Checkpoint Saving Config")
    create_parser_from_dataclass(CheckpointSavingConfig, checkpoint_saving_group)

    sft_group = parser.add_argument_group("SFT Config")
    create_parser_from_dataclass(SFTConfig, sft_group)

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
    initialize_megatron(args_defaults={"variable_seq_lengths": True}, extra_args_provider=args_provider)
    margs = get_args()
    log_level_name = getattr(margs, "light_scale_log_level", getattr(margs, "mrl_log_level", "info"))
    logger = setup_logger(
        "light_scale", level=logging.DEBUG if log_level_name == "debug" else logging.INFO
    )
    set_jit_fusion_options()
    logger.info("initialize completed")

    # sequence length校验
    if margs.sequence_parallel:
        assert margs.seq_length % margs.tensor_model_parallel_size == 0, \
            "When using sequence parallelism, seq_length must be divisible by tensor_model_parallel_size"
    if margs.context_parallel_size > 1:
        assert margs.seq_length % (margs.context_parallel_size * 2) == 0, \
            "When using context parallelism, seq_length must be divisible by context_parallel_size * 2"

    if margs.context_parallel_size > 1:
        raise NotImplementedError("context parallelism is not supported yet")

    # 暂不支持mtp
    assert not margs.mtp_num_layers, "mtp is not supported for now"
    
    # 启用packing时mbs必须为1
    if margs.sequence_packing:
        assert margs.micro_batch_size == 1, "micro_batch_size must be 1 when sequence_packing is enabled"
        logger.warning("setting variable_seq_lengths to False")
        margs.variable_seq_lengths = False
    else:
        assert margs.tp_comm_overlap is False, "tp comm bucket will determine seq length by args.seq_length"
        logger.warning("setting variable_seq_lengths to True")
        margs.variable_seq_lengths = True

    if margs.moe_token_dispatcher_type == "allgather":
        logger.warning("setting moe_token_dispatcher_type to alltoall")
        margs.moe_token_dispatcher_type = "alltoall"

    dist.barrier()
    logger.info("setup_model_and_optimizer")
    model, optimizer, scheduler = setup_model_and_optimizer()
    dist.barrier()

    # Specify --no-load-optim or --finetune to prevent attempting to load the optimizer state
    passed_iters, _ = load_checkpoint(model, optimizer, scheduler)
    logger.info("Loaded checkpoint with %d iterations", passed_iters)
    dist.barrier()

    trainer = SFTTrainer(passed_iters, model, optimizer, scheduler)
    trainer.train()
    dist.barrier()
    logger.info("done")


if __name__ == "__main__":
    main()
