from megatron.core import mpu
import argparse
from light_scale.config import WeightUpdaterTestConfig, RolloutServiceConfig, DistributedLockServerConfig
from light_scale.config_utils import create_parser_from_dataclass
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
import torch.distributed as dist
from megatron.training.training import get_model
from pretrain_gpt import model_provider
from megatron.training.checkpointing import load_checkpoint
from light_scale.logger_utils import setup_logger
from light_scale.sync_processor import SafetensorsSaver, SGLangSaver
from light_scale.weight_utils_v2 import DenseWeightUpdater, MoeWeightUpdater
import logging

from light_scale.distributed_lock import LockServerProcess, DistributedLock
from light_scale.llm_caller import wait_until_rollout_and_ref_server_ready
from light_scale import dist_utils
import os

logger = None

def args_provider(parser: argparse.ArgumentParser):
    weight_updater_test_config_group = parser.add_argument_group("Weight Updater Test")
    create_parser_from_dataclass(WeightUpdaterTestConfig, weight_updater_test_config_group)

    dist_lock_group = parser.add_argument_group("Distributed Lock")
    create_parser_from_dataclass(DistributedLockServerConfig, dist_lock_group)

    return parser

def main():
    # 加载base模型时，命令行传参："no_load_rng": True，续训时不用
    initialize_megatron(args_defaults={"micro_batch_size": 1, "no_load_rng": True, "variable_seq_lengths": True, "moe_token_dispatcher_type": "alltoall"}, extra_args_provider=args_provider)
    global logger
    logger = setup_logger("light_scale", level=logging.DEBUG)
    logger.info("initialize completed")

    margs = get_args()
    margs.variable_seq_lengths = True
    margs.moe_token_dispatcher_type = "alltoall"
    dist.barrier()

    logger.info(margs.variable_seq_lengths)
    logger.info(margs.moe_token_dispatcher_type)
    logger.info(f"safetensors save path: {margs.safetensors_save_path}")

    model = get_model(model_provider, wrap_with_ddp=False)
    logger.info("model initailized")

    load_checkpoint(model, None, None)
    logger.info("model loaded")
    dist.barrier()

    # 初始化分布式锁服务器
    dist_lock_process = None
    if dist.get_rank() == 0:
        dist_lock_process = LockServerProcess(
            host='0.0.0.0',
            port=margs.dist_lock_server_port,
        )
        dist_lock_process.start_lock_server(lock_server_log_level=logging.DEBUG)

    logger.info("initializing weight updater")
    sync_processor = None
    dense_weight_updater = None
    moe_weight_updater = None
    if margs.online_test:
        # 等待rollout和ref server就绪
        rollout_service_config = RolloutServiceConfig(
            rollout_base_url_list=margs.rollout_base_url_list,
            rollout_model_name=margs.rollout_model_name,
            server_world_size=margs.server_world_size,
            weight_update_group_port=margs.weight_update_group_port
        )
        if dist.get_rank() == 0:
            wait_until_rollout_and_ref_server_ready(rollout_service_config, None, retry=100)
        dist.barrier()
        logger.info("rollout and ref server is ready")
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
                rollout_service_config.weight_update_group_port
            )
        dist.barrier()
    else:
        dist_lock = DistributedLock(
            "weight_update",
            host=os.environ['MASTER_ADDR'],
            port=margs.dist_lock_server_port
        )
        if dist_utils.is_pp_src_rank():
            sync_processor = SafetensorsSaver(
                file_name=f"{margs.safetensors_save_path}/pp_{mpu.get_pipeline_model_parallel_rank()}",
                dist_lock=dist_lock
            )
        dist.barrier()
    if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        dense_weight_updater = DenseWeightUpdater(model, sync_processor)
    if margs.num_experts is not None and margs.num_experts > 0 and mpu.get_expert_data_parallel_rank() == 0:
        moe_weight_updater = MoeWeightUpdater(model, sync_processor)
    dist.barrier()
    logger.info("initialized weight updater")

    if dense_weight_updater is not None:
        logger.info("updating dense model weight to inference service")
        dense_weight_updater()
    if moe_weight_updater is not None:
        logger.info("updating moe model weight to inference service")
        moe_weight_updater()
    dist.barrier()
    
    logger.info("done")

if __name__ == '__main__':
    main()