from torch import Tensor
from light_scale import dist_utils
from megatron.core import mpu
import argparse
from light_scale.config import ReferenceModelServingConfig, LogitsExpressConfig, GKDConfig
from light_scale.config_utils import create_parser_from_dataclass
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
import torch.distributed as dist
from megatron.training.training import get_model
from pretrain_gpt import model_provider
from megatron.training.checkpointing import load_checkpoint
from light_scale.logger_utils import setup_logger
from light_scale.sync_processor import ActorReferenceDataUpdater
from light_scale.grpo_utils import compute_batch_logp
import torch
from dataclasses import dataclass
import os
import numpy as np
from typing import List
from light_scale.data import BatchExperience
from light_scale.dataset import create_distributed_dataloader
from light_scale.logits_express import LogitsExpress
from light_scale.grpo_utils import compute_batch_logits
import logging

logger = None

def args_provider(parser: argparse.ArgumentParser):
    reference_serving_group = parser.add_argument_group("Reference Serving Config")
    create_parser_from_dataclass(ReferenceModelServingConfig, reference_serving_group)

    # Distillation-related configs (used when distillation is enabled)
    logits_express_group = parser.add_argument_group("Logits Express Config")
    create_parser_from_dataclass(LogitsExpressConfig, logits_express_group)

    gkd_group = parser.add_argument_group("GKD Config")
    create_parser_from_dataclass(GKDConfig, gkd_group)

    return parser

def parse_configs(args):
    # 解析到 dataclass
    reference_serving_config = ReferenceModelServingConfig(**{k: v for k, v in vars(args).items() if k in ReferenceModelServingConfig.__annotations__})

    return reference_serving_config

def rank_0_wait_for_inputs(data_updater: ActorReferenceDataUpdater) -> List[BatchExperience]:
    dist.barrier(data_updater.update_group)
    num_batches_tensor = data_updater.actor_send_reference_receive_2D_tensor(None, dtype=torch.int64, shape_tensor=torch.LongTensor([1, 1])).squeeze(dim=1)
    batch_inputs: List[BatchExperience] = [None] * num_batches_tensor
    for i in range(num_batches_tensor):
        input_ids = data_updater.actor_send_reference_receive_2D_tensor(None, dtype=torch.int64)
        labels = data_updater.actor_send_reference_receive_2D_tensor(None, dtype=torch.int64, shape_tensor=torch.LongTensor([input_ids.shape[0], input_ids.shape[1]]))
        batch_inputs[i] = BatchExperience(input_ids=input_ids, labels=labels)
    return batch_inputs

def dispatch_inputs(batch_input: BatchExperience) -> BatchExperience:
    # sync shape for cpu pin memory init
    if dist.get_rank() == 0:
        batch_size, seq_length = batch_input.input_ids.shape
        shape_tensor = torch.tensor([batch_size, seq_length], dtype=torch.int64, device=dist_utils.get_device())
        dist_utils._sync_2D_input_data(shape_tensor.unsqueeze(dim=1), torch.int64, shape_tensor=torch.LongTensor([2, 1]))
    else:
        shape_tensor = dist_utils._sync_2D_input_data(None, torch.int64, shape_tensor=torch.LongTensor([2, 1])).squeeze(dim=1)

    input_ids = None
    labels = None

    if dist.get_rank() == 0:
        # keep the gpu tensor
        input_ids = batch_input.input_ids
        labels = batch_input.labels
    
    if batch_input is None:
        batch_input = BatchExperience()
    
    batch_input.input_ids = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
    batch_input.labels = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
    if dist.get_rank() == 0:
        batch_input.ref_logps = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)

    input_ids = dist_utils._sync_2D_input_data(input_ids, torch.int64, shape_tensor)
    labels = dist_utils._sync_2D_input_data(labels, torch.int64, shape_tensor)

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        batch_input.input_ids.copy_(input_ids, non_blocking=True)
        batch_input.labels.copy_(labels, non_blocking=True)
    s.synchronize()

    return batch_input

def compute_and_send_teacher_logits(model, batch_input: BatchExperience, margs, logits_express: LogitsExpress):
    # 计算一个batch experience的logits
    # 按照logits_transfer_batch_size分批计算
    logits_transfer_batch_size = margs.logits_transfer_batch_size // mpu.get_data_parallel_world_size()
    data_iter = None
    if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
        dataloader = create_distributed_dataloader([batch_input], margs.micro_batch_size)
        data_iter = iter(dataloader)
    dist.barrier()
    # 通过控制num_microbatches实现分批计算
    if dist.get_rank() == 0:
        num_microbatches = logits_transfer_batch_size // margs.micro_batch_size
        num_iters = (batch_input.input_ids.shape[0] // mpu.get_data_parallel_world_size()) // logits_transfer_batch_size
        num_batches_tensor = torch.tensor([num_microbatches, num_iters], dtype=torch.int64, device=dist_utils.get_device())
        dist.broadcast(num_batches_tensor, src=0)
    else:
        num_batches_tensor = torch.tensor([0, 0], dtype=torch.int64, device=dist_utils.get_device())
        dist.broadcast(num_batches_tensor, src=0)
        num_microbatches = num_batches_tensor[0].item()
        num_iters = num_batches_tensor[1].item()
    dist.barrier()
    
    assert num_microbatches > 0, "logits_transfer_batch_size太小，无法满足一个micro_batch的计算"
    logger.debug(f"num_iters: {num_iters}, num_microbatches: {num_microbatches}")
    for i in range(num_iters):
        logger.debug(f"iter_num: {i}")
        logits_list = compute_batch_logits(
            model=model,
            data_iterator=data_iter,
            num_microbatches=num_microbatches,
            micro_batch_size=margs.micro_batch_size,
            iter_num=i,
        )
        if mpu.is_pipeline_last_stage():
            if getattr(margs, "gkd_sparse_topk_enabled", False):
                topk_indices = torch.cat([v['topk_indices'] for v in logits_list], dim=0)
                topk_values = torch.cat([v['topk_values'] for v in logits_list], dim=0)
                logits_express.teacher_send_student_receive_topk(
                    indices_global=topk_indices,
                    values=topk_values,
                )
            else:
                # 发送logits到actor（dense 路径）
                logits_to_transfer = torch.cat([v['logits'] for v in logits_list], dim=0)

                logger.debug(f"logits_to_transfer shape: {logits_to_transfer.shape}")
                if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                    dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                    if dump_path is None:
                        raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                    np.save(f"{dump_path}/teacher_send_logits_iter_{i}_tp_{mpu.get_tensor_model_parallel_rank()}_dp_{mpu.get_data_parallel_rank()}.npy", logits_to_transfer.detach().cpu().numpy())
                # else:
                #     dump_path = "/root/work/filestorage/GroupPostTrain/taoyuyang/projects/yingxiao_kd/Megatron-RL/training_outputs/yingxiao_gkd_7B_30B/dump_dir_2"
                #     if dump_path is None:
                #         raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                #     np.save(f"{dump_path}/teacher_send_logits_iter_{i}_tp_{mpu.get_tensor_model_parallel_rank()}_dp_{mpu.get_data_parallel_rank()}.npy", logits_to_transfer.detach().cpu().numpy())
                logits_express.teacher_send_student_receive(logits_to_transfer)
        logger.debug("waiting for last stage to send logits")
        dist.barrier()
    logger.debug("finished sending logits for a batch input")

def main():
    # ValueError: Token dispatcher type: allgather does not support variable sequence length, please use alltoall dispatcher instead.
    initialize_megatron(args_defaults={"no_load_rng": True, "variable_seq_lengths": True, "moe_token_dispatcher_type": "alltoall"}, extra_args_provider=args_provider)
    global logger

    margs = get_args()
    margs.variable_seq_lengths = True
    margs.moe_token_dispatcher_type = "alltoall"
    dist.barrier()

    log_level_name = getattr(margs, "light_scale_log_level", getattr(margs, "mrl_log_level", "info"))
    logger = setup_logger("light_scale", level=logging.DEBUG if log_level_name == "debug" else logging.INFO)
    logger.info("initialize completed")

    logger.info(margs.variable_seq_lengths)
    logger.info(margs.moe_token_dispatcher_type)

    logger.info("parsing configs")
    reference_serving_config = parse_configs(margs)

    model = get_model(model_provider, wrap_with_ddp=False)
    logger.info("model initailized")

    load_checkpoint(model, None, None)
    logger.info("model loaded")
    dist.barrier()

    data_updater = None
    if dist.get_rank() == 0:
        data_updater = ActorReferenceDataUpdater(
            actor_master_addr=reference_serving_config.actor_master_addr,
            update_group_port=reference_serving_config.data_transfer_group_port,
            is_actor=False,
            timeout_minutes=margs.distributed_timeout_minutes
        )
    dist.barrier()

    logger.info("warming up pp group comm")
    # very import and necesarry
    tensor = torch.zeros((1024,), dtype=torch.bfloat16, device=dist_utils.get_device())
    dist.all_reduce(tensor, group=mpu.get_pipeline_model_parallel_group())
    dist.barrier()
    logits_express = None
    if margs.distillation_enabled:
        logits_express = LogitsExpress(data_updater, is_teacher=True)
    dist.barrier()
    iter_num = 0
    while True:
        batch_inputs = None
        torch.cuda.empty_cache()

        logger.info("waiting for input")
        if dist.get_rank() == 0:
            batch_inputs = rank_0_wait_for_inputs(data_updater)
            logger.info(f"received {len(batch_inputs)} input batches")
        dist.barrier()
        logger.info("received inputs")

        if dist.get_rank() == 0:
            num_batches_tensor = torch.tensor([len(batch_inputs)], dtype=torch.int64, device=dist_utils.get_device())
            dist.broadcast(num_batches_tensor, src=0)
        else:
            num_batches_tensor = torch.tensor([0], dtype=torch.int64, device=dist_utils.get_device())
            dist.broadcast(num_batches_tensor, src=0)
        
        num_batches = num_batches_tensor.item()
        if batch_inputs is None:
            batch_inputs: List[BatchExperience] = [None] * num_batches
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            for i in range(num_batches):
                batch_inputs[i] = dispatch_inputs(batch_inputs[i])
        
        if dist.get_rank() == 0:
            n_batches_list = [batch_input.input_ids.shape[0] // (margs.micro_batch_size * mpu.get_data_parallel_world_size()) \
                                for batch_input in batch_inputs]
            n_batches_tensor = torch.tensor(n_batches_list, dtype=torch.int64, device=dist_utils.get_device())
        else:
            n_batches_tensor = torch.zeros((num_batches,), dtype=torch.int64, device=dist_utils.get_device())
        dist.broadcast(n_batches_tensor, src=0)
        n_batches_list = n_batches_tensor.cpu().tolist()
        logger.debug(n_batches_list)

        data_iter = None
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            dataloader = create_distributed_dataloader(batch_inputs, margs.micro_batch_size)
            data_iter = iter(dataloader)

        for i, batch_input in enumerate(batch_inputs):
            iter_num += 1

            if margs.distillation_enabled:
                logger.info("computing and sending teacher logits")
                compute_and_send_teacher_logits(model, batch_input, margs, logits_express)
                logger.info("teacher logits sent")
                continue

            logger.info("computing ref logps")
            ref_logps = compute_batch_logp(
                model=model,
                data_iterator=data_iter,
                num_microbatches=n_batches_list[i],
                micro_batch_size=margs.micro_batch_size
            )

            print("========", flush=True)
            print(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", None)), flush=True)
            print(os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None)), flush=True)
            if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1 and mpu.get_tensor_model_parallel_rank() == 0:
                # for debug
                dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                if dump_path is None:
                    raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                if batch_input is not None:
                    if batch_input.input_ids is not None:
                        np.save(f"{dump_path}/ref_iter_{iter_num}_dp_{mpu.get_data_parallel_rank()}_cp_{mpu.get_context_parallel_rank()}_input_ids.npy", batch_input.input_ids.detach().cpu().numpy())
                    if batch_input.labels is not None:
                        np.save(f"{dump_path}/ref_iter_{iter_num}_dp_{mpu.get_data_parallel_rank()}_cp_{mpu.get_context_parallel_rank()}_labels.npy", batch_input.labels.detach().cpu().numpy())
                if ref_logps is not None:
                    np.save(f"{dump_path}/ref_iter_{iter_num}_dp_{mpu.get_data_parallel_rank()}_cp_{mpu.get_context_parallel_rank()}_ref_logps.npy", ref_logps.detach().cpu().numpy())

            if dist.get_rank() == 0:
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    batch_input.ref_logps.copy_(ref_logps, non_blocking=True)
                s.synchronize()
            
            dist.barrier()
        
        if margs.distillation_enabled:
            dist.barrier()
            continue
        logger.info("ref_logp caculated, sending back...")
        if dist.get_rank() == 0:
            dist.barrier(data_updater.update_group)
            for batch_input in batch_inputs:
                data_updater.actor_receive_reference_send_2D_tensor(tensor=batch_input.ref_logps.to(device=dist_utils.get_device(), non_blocking=True), dtype=torch.float32)
        dist.barrier()


if __name__ == '__main__':
    main()