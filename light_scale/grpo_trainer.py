from typing import List, Optional, Tuple, Union, Iterator, Dict
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
import numpy as np
import random
from tqdm import tqdm
import json
import time
import logging
import datetime
import sys

from datasets import load_from_disk

from light_scale.grpo_utils import compute_approx_kl, compute_token_reward, compute_returns, masked_mean, vocab_parallel_entropy
from light_scale.math import verify_gsm8k_sample
from megatron.training.global_vars import get_args, get_tokenizer
from light_scale.llm_caller import InferenceServiceCaller
from megatron.core import mpu
from light_scale import dist_utils
from light_scale.grpo_utils import compute_batch_logp
from light_scale.logp_utils import from_parallel_logits_to_logprobs
from torch.distributed import ReduceOp
from functools import partial
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from light_scale.logger_utils import setup_logger
from megatron.training.checkpointing import save_checkpoint as save_megatron_checkpoint
from megatron.training.training import enable_forward_pre_hook, disable_forward_pre_hook
from torch.utils.tensorboard import SummaryWriter
from verifier.rule_based_rm_cot import compute_score as compute_score_cot
from verifier.rule_based_rm import compute_score
from light_scale.sync_processor import ActorReferenceDataUpdater
import math
from light_scale import score_utils

from megatron.core.utils import (
    get_model_config
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads

from light_scale.dataset import create_distributed_dataloader
from light_scale.data import Sample, MultiResponseSample, BatchExperience
from light_scale.logits_express import LogitsExpress
from light_scale.gkd_utils import distributed_log_softmax, safe_all_reduce

import torch.nn.functional as F

import torch.distributed.nn.functional as FF

class RolloutDataloader:
    def __init__(self, hf_dataset_path: str, rollout_batch_size: int, passed_iters: int, seed: int):
        logger = setup_logger("light_scale")
        self.rollout_batch_size = rollout_batch_size
        self.passed_iters = passed_iters

        logger.info(f"loading from {hf_dataset_path}")
        samples = load_from_disk(hf_dataset_path)
        self.samples = samples.shuffle(seed=seed)
        assert len(self.samples) > 0
        self.cur_id = (self.passed_iters * self.rollout_batch_size + len(self.samples)) % len(self.samples)

    def _read_single_sample(self):
        # 读取一个sample，更新cur_id
        if self.cur_id == len(self.samples):
            self.cur_id = 0
        sample = self.samples[self.cur_id]
        self.cur_id += 1
        return sample
    
    def _process_single_sample(self, sample):
        # 预处理一条sample
        # for debug: 直接读取sample中的prompt字段
        assert "prompt" in sample
        return {
            "prompt": sample["prompt"],
            "ground_truth": sample["ground_truth"],
            "dataset_type": sample["dataset_type"]
        }

    def __next__(self):
        # 读取rollout batch个sample
        samples = [self._read_single_sample() for _ in range(self.rollout_batch_size)]

        # 预处理得到prompts
        prompts = [self._process_single_sample(sample) for sample in samples]

        return prompts

# copy from NeMo
def _ceil_to_nearest(n, m, ceil_to_power_2=False):
    if ceil_to_power_2:
        # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
        return 2 ** math.ceil(math.log2(n))
    else:
        return (n + m - 1) // m * m

class GRPOTrainer:
    def __init__(self, passed_iters: int, model, optim, scheduler, dense_weight_updater, moe_weight_updater, data_updater: ActorReferenceDataUpdater, logits_express: Optional[LogitsExpress] = None):
        # TODO: 配置参数解析，待定，1）基于megatron参数解析，添加参数；2）自己定义参数，hack megatron参数解析，参考convert script，用argv
        # 目前采用1）
        args = get_args()
        logger = setup_logger("light_scale")
        logger.warning(f"seed: {args.seed}")
        random.seed(args.seed) # TODO: 复用megatron rng state

        # 稳妥性保护：当前同步 trainer 仅支持 dense teacher logits 蒸馏路径。
        # sparse topk 蒸馏已在 async trainer 中实现；这里显式拒绝，避免 actor/ref 侧通信语义不一致导致卡死。
        if bool(getattr(args, "distillation_enabled", False)) and bool(getattr(args, "gkd_sparse_topk_enabled", False)):
            raise NotImplementedError(
                "Sync GRPOTrainer does not support gkd_sparse_topk_enabled yet. "
                "Please use main_async_actor.py (async_grpo_trainer) or disable gkd_sparse_topk_enabled."
            )

        train_batch_size = args.global_batch_size
        num_repeat_times = args.num_repeat_times
        assert (args.rollout_batch_size * args.n_samples) % train_batch_size == 0
        train_iters_per_rollout_step = (args.rollout_batch_size * args.n_samples) // (train_batch_size * num_repeat_times)
        assert (passed_iters + train_iters_per_rollout_step) % train_iters_per_rollout_step == 0, "Checkpoint should NOT be saved inside a rollout step"

        rollout_service = None
        dataloader = None
        tokenizer = None
        tensorboard_writer = None
        if dist.get_rank() == 0:
            rollout_service = InferenceServiceCaller(
                url_list=[f"{url}/v1" for url in args.rollout_base_url_list],
                model_name=args.rollout_model_name,
                batch_size=args.sampling_pool_size if args.sampling_pool_size > 0 else None
            )
            
            dataloader = RolloutDataloader(
                hf_dataset_path=args.data_path[0],
                rollout_batch_size=args.rollout_batch_size,
                passed_iters=passed_iters // train_iters_per_rollout_step,
                seed=args.seed
            )

            assert args.tensorboard_dir is not None
            tensorboard_writer = SummaryWriter(
                log_dir=args.tensorboard_dir,
                filename_suffix="light_scale"
            )

            tokenizer = get_tokenizer()._tokenizer
            assert tokenizer.pad_token_id is not None

            self.rollout_metrics = dict()

        self.rollout_service = rollout_service
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.tensorboard_writer = tensorboard_writer

        self.args = args
        self.passed_iters = passed_iters
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.dense_weight_updater = dense_weight_updater
        self.moe_weight_updater = moe_weight_updater
        self.data_updater = data_updater
        self.logger = logger
        self.logits_express = logits_express

        self.is_old_actor_logp_required = True

        self.s1 = torch.cuda.Stream()
        self.s2 = torch.cuda.Stream()

        self.n_batches_list = None

        self.logits_cache = None

    def _attach_distill_segments(self, batch_experience_list: List[BatchExperience]):
        """为每个 BatchExperience 计算并附加 distill_segments（DP 对齐段）。

        最小改动原则：仅当开启蒸馏且存在 logits_express 时生效；否则不做任何事。
        段的计算依赖于本 rank 的 DP 坐标与 Teacher/Student 的 DP 规模关系，
        与 TP 无关；使用我们在 LogitsExpress 中实现的 simulate_student_segments。
        """
        if not getattr(self.args, 'distillation_enabled', False):
            return
        if self.logits_express is None:
            return
        for be in batch_experience_list:
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                total_bs = int(be.input_ids.shape[0])
                iter_transfer_bs = int(self.args.logits_transfer_batch_size)  # 按“全局”样本数定义的每轮传输量
                assert iter_transfer_bs > 0, "logits_transfer_batch_size 必须为正"
                assert total_bs % iter_transfer_bs == 0, \
                    f"total_bs={total_bs} 需能被 logits_transfer_batch_size={iter_transfer_bs} 整除以便整轮传输"

                num_iters_total = total_bs // iter_transfer_bs
                all_segments = None
                if mpu.is_pipeline_last_stage():
                    all_segments = []
                    for i in range(num_iters_total):
                        iter_offset = i * iter_transfer_bs
                        seg_i = self.logits_express.simulate_student_segments_via_neighbors(total_bs, iter_num=i)
                        all_segments.extend(seg_i)
                
                # 由于只有最后一个pp stage持有neighbors，所以需要每一个last stage的rank p2p将segments传递给自己的pp first stage rank
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    if mpu.is_pipeline_first_stage():
                        buffer = [None]
                        dist.recv_object_list(buffer, src=mpu.get_pipeline_model_parallel_last_rank())
                        all_segments = buffer[0]
                    else:
                        buffer = [all_segments]
                        dist.send_object_list(buffer, dst=mpu.get_pipeline_model_parallel_first_rank())
                # 存为常规 python (start,length) 元组，供 Dataset 使用
                be.distill_segments = [(int(s), int(l)) for (s, l) in all_segments]
            dist.barrier()

    def train(self):
        """Training main function.
        """
        args = self.args
        logger = self.logger
        passed_iters = self.passed_iters
        train_batch_size = args.global_batch_size
        num_repeat_times = args.num_repeat_times
        train_iters_per_rollout_step = (args.rollout_batch_size * args.n_samples // train_batch_size) * num_repeat_times
        self.logger.warning(f"train_iters_per_rollout_step = ({args.rollout_batch_size} * {args.n_samples} // {train_batch_size}) * {num_repeat_times} = {train_iters_per_rollout_step}")
        if train_iters_per_rollout_step == 1:
            if self.args.init_kl_coef > 1e-8 and self.args.use_kl_loss:
                self.is_old_actor_logp_required = False
            elif self.args.init_kl_coef <= 1e-8:
                self.is_old_actor_logp_required = False
        self.logger.warning(f"is_old_actor_logp_required: {self.is_old_actor_logp_required}")

        # Setup some training config params.
        self.logger.info("setting up some training config params.")
        config = get_model_config(self.model[0])
        config.grad_scale_func = self.optim.scale_loss
        if isinstance(self.model[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, \
                ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.model]
            if len(self.model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.align_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.model]
                if len(self.model) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.align_param_gather:
            config.param_sync_func = [model_chunk.start_param_sync for model_chunk in self.model]
            if len(self.model) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads

        logger.info("warming up pp group comm")
        self.all_reduce_in_pp_group()
        dist.barrier()

        logger.info("start training")
        pbar = tqdm(desc="Training", initial=passed_iters // train_iters_per_rollout_step, total=args.train_iters // train_iters_per_rollout_step)
        while passed_iters < args.train_iters:
            assert (passed_iters + train_iters_per_rollout_step) % train_iters_per_rollout_step == 0, "Something wrong"
            rollout_step = (passed_iters // train_iters_per_rollout_step) + 1
            all_samples = None
            time_costs = dict()
            if dist.get_rank() == 0:
                step_st = time.time()
                st = time.time()
                all_samples = self.rank_0_preprocess_step(iter=rollout_step)
                assert len(all_samples) == args.rollout_batch_size * args.n_samples
                et = time.time()
                time_costs["rollout_preprocess"] = et - st

                rollout_metrics_log = ", ".join([f"{k}: {v}" for k, v in self.rollout_metrics[str(rollout_step)].items()])
                rollout_metrics_log = f"rollout step: {rollout_step}, {rollout_metrics_log}"
                self.logger.info(rollout_metrics_log)

                self.logger.info("writing tensorboard")
                writer = self.tensorboard_writer
                for k, v in self.rollout_metrics[str(rollout_step)].items():
                    writer.add_scalar(f"rollout/{k}", v, rollout_step)
                writer.flush()
            dist.barrier()

            self.logger.info("collating and dispatching experiences")
            st = time.time()
            batch_experience_list: List[BatchExperience] = []
            for i in tqdm(range(0, args.rollout_batch_size * args.n_samples, train_batch_size), desc="collating and dispatching experiences"):
                batch_experience = None
                if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                    if dist.get_rank() == 0:
                        batch_samples = all_samples[i:i+train_batch_size]
                        # batch_experience = self.collate_fn(batch_samples)
                        batch_experience = self.collate_fn_optimized(batch_samples)
                    batch_experience = self.dispatch_batch_experience_before_making_experience(batch_experience)
                batch_experience_list.append(batch_experience)
            dist.barrier()
            assert len(batch_experience_list) == train_iters_per_rollout_step // num_repeat_times
            self.logger.info("syncing number of microbatches")
            if dist.get_rank() == 0:
                logp_n_batches_list = [batch_experience.input_ids.shape[0] // (args.micro_forward_batch_size * mpu.get_data_parallel_world_size()) \
                                       for batch_experience in batch_experience_list]
                train_n_batches_list = [batch_experience.input_ids.shape[0] // (args.micro_batch_size * mpu.get_data_parallel_world_size()) \
                                        for batch_experience in batch_experience_list]
                n_batches_tensor = torch.tensor([logp_n_batches_list, train_n_batches_list], dtype=torch.int64, device=dist_utils.get_device())
            else:
                n_batches_tensor = torch.zeros((2, len(batch_experience_list)), dtype=torch.int64, device=dist_utils.get_device())
            dist.broadcast(n_batches_tensor, src=0)
            self.n_batches_list = n_batches_tensor.cpu().tolist()
            et = time.time()
            time_costs["collate_dispatch"] = et - st
            self.logger.debug(self.n_batches_list)

            if self.args.distillation_enabled:
                self.logger.info("computing teacher logits")
                st = time.time()
                self.compute_teacher_logits(batch_experience_list)
                et = time.time()
                time_costs["compute_teacher_logits"] = et - st

                # 蒸馏：在创建任意 DataLoader 之前，附加 distill_segments，确保 Dataset 与 logits 对齐
                self._attach_distill_segments(batch_experience_list)

            if self.args.init_kl_coef > 1e-8:
                self.logger.info("sending inputs to ref model")
                st = time.time()
                if dist.get_rank() == 0:
                    self.rank_0_send_input_to_ref(batch_experience_list)
                dist.barrier()
                et = time.time()
                time_costs["send_to_ref"] = et - st

            self.logger.info("computing log probs")
            st = time.time()
            self.actor_forward_only(batch_experience_list)
            et = time.time()
            time_costs["compute_actor_logp"] = et - st

            if self.args.init_kl_coef > 1e-8:
                self.logger.info("receiving and syncing ref log probs")
                st = time.time()
                self.receive_and_sync_output_from_ref(batch_experience_list)
                et = time.time()
                time_costs["receive_from_ref"] = et - st

            self.logger.info("computing advantages")
            st = time.time()
            self.compute_advantages(batch_experience_list)
            et = time.time()
            time_costs["compute_advantage"] = et - st

            # train step
            self.logger.info("train steps")
            st = time.time()
            for _ in range(self.args.num_repeat_times):
                data_iter = None
                if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                    dataloader = create_distributed_dataloader(batch_experience_list, self.args.micro_batch_size)
                    data_iter = iter(dataloader)
                for i, batch_experience in tqdm(enumerate(batch_experience_list), desc="running train steps", total=len(batch_experience_list)):
                    if self.args.distillation_enabled:
                        self.distillation_step(data_iter, num_microbatches=self.n_batches_list[1][i], iter_num=passed_iters + 1)
                    else:
                        self.train_step(data_iter, num_microbatches=self.n_batches_list[1][i], iter_num=passed_iters + 1)
                    if self.args.dump_experience and dist.get_rank() == 0:
                        self.dump_experience(passed_iters + 1, batch_experience, dump_path=self.args.dump_path, dump_tensors=self.args.dump_tensors)
                    passed_iters += 1
            dist.barrier()
            et = time.time()
            time_costs["train_steps"] = et - st

            # 参数同步
            st = time.time()
            self.update_params_to_sglang()
            et = time.time()
            time_costs["weight_update"] = et - st

            if dist.get_rank() == 0:
                step_et = time.time()
                time_costs["step"] = step_et - step_st
                logger.info("=========Time Costs=========")
                time_costs_log = ", ".join([f"{k}: {int(v)} seconds" for k, v in time_costs.items()])
                if self.args.total_world_size is not None:
                    step_throughput = self.rollout_metrics[str(rollout_step)]["total_tokens"] / time_costs["step"] / self.args.total_world_size
                    time_costs_log += f", step_throughput: {step_throughput} tokens/s/p"
                    writer.add_scalar(f"metrics/throughput", step_throughput, rollout_step)
                logger.info(time_costs_log)
                self.logger.info("writing tensorboard of time costs")
                writer = self.tensorboard_writer
                for k, v in time_costs.items():
                    writer.add_scalar(f"time/{k}", v, rollout_step)
                writer.flush()
            dist.barrier()

            if passed_iters % args.save_interval == 0:
                self.save_logs_and_checkpoints(passed_iters)
            dist.barrier()

            pbar.update(1)

            if rollout_step == self.args.early_stop_steps:
                self.logger.warning(f"reached early stop steps: {rollout_step}")
                break
        
        pbar.close()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def all_reduce_in_pp_group(self):
        # very import and necesarry
        tensor = torch.zeros((1024,), dtype=torch.bfloat16, device=dist_utils.get_device())
        dist.all_reduce(tensor, group=mpu.get_pipeline_model_parallel_group())

    def dump_experience(self, iter: int, batch_experience: BatchExperience, dump_path: str, dump_tensors: bool = False):
        with open(f"{dump_path}/iter_{iter}_batch_samples.jsonl", mode='w', encoding='utf-8') as f:
            for sample in batch_experience.batch_samples:
                raw_line = json.dumps({
                    "prompt": sample.prompt,
                    "response": sample.response,
                    "reward": sample.reward,
                    "normed_reward": sample.normed_reward,
                    "ground_truth": sample.ground_truth
                }, ensure_ascii=False)
                f.write(raw_line)
                f.write('\n')
                f.flush()
        if dump_tensors:
            np.save(f"{dump_path}/iter_{iter}_input_ids.npy", batch_experience.input_ids.numpy())
            np.save(f"{dump_path}/iter_{iter}_labels.npy", batch_experience.labels.numpy())
            np.save(f"{dump_path}/iter_{iter}_loss_mask.npy", batch_experience.loss_mask.numpy())
            np.save(f"{dump_path}/iter_{iter}_old_actor_logps.npy", batch_experience.old_actor_logps.numpy())
            np.save(f"{dump_path}/iter_{iter}_advantages.npy", batch_experience.advantages.numpy())
    
    def dump_deviant_experience(self, dump_path: str, iter: int, batch_ids: List[int], samples: List[Sample], deviant_reason: str, **tensors):
        assert os.path.exists(dump_path)
        assert len(batch_ids) > 0
        dump_path += "/deviants"
        os.makedirs(dump_path, exist_ok=True)
        batch_id_str = str(batch_ids[0])
        for i in range(1, len(batch_ids)):
            batch_id_str += '-' + str(batch_ids[i])
        dp = mpu.get_data_parallel_rank()
        cp = mpu.get_context_parallel_rank()
        pp = mpu.get_pipeline_model_parallel_rank()
        tp = mpu.get_tensor_model_parallel_rank()
        dump_path_prefix = f"{dump_path}/iter_{iter}_batch_{batch_id_str}_dp_{dp}_cp{cp}_pp{pp}_tp{tp}"
        self.logger.warning(f"dumping deviant experience: {os.path.basename(dump_path_prefix)}")
        for key, tensor in tensors.items():
            if tensor is not None:
                np_array = tensor.detach().cpu().numpy()
                np.save(f"{dump_path_prefix}_{key}.npy", np_array)
        with open(f"{dump_path_prefix}_samples.jsonl", mode='w', encoding='utf-8') as f:
            for sample in samples:
                raw_line = json.dumps({
                    "prompt": sample.prompt,
                    "response": sample.response,
                    "reward": sample.reward,
                    "normed_reward": sample.normed_reward,
                    "ground_truth": sample.ground_truth
                }, ensure_ascii=False)
                f.write(raw_line)
                f.write('\n')
                f.flush()
        with open(f"{dump_path_prefix}_reason.txt", mode='w', encoding='utf-8') as f:
            f.write(deviant_reason)
            f.flush()
                
    def dispatch_batch_experience_before_making_experience(self, batch_experience: BatchExperience) -> BatchExperience:
        # make every rank in first and last stage hold the batch experience they needed
        self.logger.debug("dispatching batch experience")

        # sync shape for cpu pin memory init
        if dist.get_rank() == 0:
            batch_size, seq_length = batch_experience.input_ids.shape
            shape_tensor = torch.tensor([batch_size, seq_length], dtype=torch.int64, device=dist_utils.get_device())
            dist_utils._sync_2D_input_data(shape_tensor.unsqueeze(dim=1), torch.int64, shape_tensor=torch.LongTensor([2, 1]))
        else:
            shape_tensor = dist_utils._sync_2D_input_data(None, torch.int64, shape_tensor=torch.LongTensor([2, 1])).squeeze(dim=1)

        self.logger.debug(f"shape tensor: {shape_tensor}")

        input_ids = None
        labels = None
        loss_mask = None
        outcome_rewards = None

        if dist.get_rank() == 0:
            # keep the gpu tensor
            input_ids = batch_experience.input_ids
            labels = batch_experience.labels
            loss_mask = batch_experience.loss_mask
            outcome_rewards = batch_experience.outcome_rewards
        
        if batch_experience is None:
            batch_experience = BatchExperience()

        batch_experience.input_ids = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
        batch_experience.labels = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
        batch_experience.loss_mask = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        batch_experience.outcome_rewards = torch.zeros((shape_tensor[0],), dtype=torch.float32, device="cpu", pin_memory=True)
        if self.is_old_actor_logp_required:
            batch_experience.old_actor_logps = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        if self.args.init_kl_coef > 1e-8:
            batch_experience.ref_logps = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        # if self.args.init_kl_coef > 1e-8 and not self.args.use_kl_loss:
        #     batch_experience.kls = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        if not self.args.use_outcome_rewards_as_advantages:
            batch_experience.advantages = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)

        input_ids = dist_utils._sync_2D_input_data(input_ids, torch.int64, shape_tensor)
        labels = dist_utils._sync_2D_input_data(labels, torch.int64, shape_tensor)
        loss_mask = dist_utils._sync_2D_input_data(loss_mask, torch.float32, shape_tensor)
        outcome_rewards = dist_utils._sync_2D_input_data(
            outcome_rewards.unsqueeze(dim=0) if outcome_rewards is not None else None, 
            torch.float32, torch.LongTensor([1, shape_tensor[0]])
        ).squeeze(dim=0)

        with torch.cuda.stream(self.s1):
            batch_experience.input_ids.copy_(input_ids, non_blocking=True)
            batch_experience.labels.copy_(labels, non_blocking=True)
        with torch.cuda.stream(self.s2):
            batch_experience.loss_mask.copy_(loss_mask, non_blocking=True)
            batch_experience.outcome_rewards.copy_(outcome_rewards, non_blocking=True)
        self.s1.synchronize()
        self.s2.synchronize()

        return batch_experience
            
    def rank_0_preprocess_step(self, iter: int) -> List[Sample]:
        self.logger.debug("rank 0 preprocessing")
        # 取数据
        raw_samples = next(self.dataloader)

        # 封装
        samples = [MultiResponseSample(**raw_sample) for raw_sample in raw_samples]

        # sglang采样
        samples, num_failed_samples = self.rollout(samples)  # dataloader创建时就是 a list of MultiResponseSample，包含prompt、ground truth、dataset_type
        
        # reward model打分
        # samples = self.score(samples)
        self.score_v2(samples)
        # 数据后处理
        samples = self.postprocess_samples(samples)

        metrics = {
            "rollout_failed_samples": num_failed_samples,
            "avg_completion_tokens": sum([sample.completion_tokens for sample in samples]) // (len(samples) * len(samples[0].responses)),
            "total_tokens": sum([sample.total_tokens for sample in samples]),
            "reward": sum([reward if reward is not None else 0 for sample in samples for reward in sample.rewards]) / (len(samples) * len(samples[0].responses)),
            "normed_reward_zero_samples": sum(1 for sample in samples if all(v == 0.0 for v in sample.normed_rewards))
        }
        for key in samples[0].reward_metrics_list[0].keys():
            assert key != "reward"
            assert key != "completion_tokens"
            metrics[key] = sum([reward_metrics[key] for sample in samples for reward_metrics in sample.reward_metrics_list]) / (len(samples) * len(samples[0].responses))
        self.rollout_metrics[str(iter)] = metrics

        all_samples = self.unfold_samples(samples)
        return all_samples

    def rollout(self, samples: List[MultiResponseSample]) -> List[MultiResponseSample]:
        """调用推理服务，填写MultiResponseSample.responses。
        """
        if self.args.force_thinking:
            for sample in samples:
                sample.prompt += self.args.begin_of_thinking
        # 预处理得到prompts
        prompts = [sample.prompt for sample in samples]
        self.logger.info("calling inference service")
        results = self.rollout_service.batch_completions(
            prompts=prompts,
            n_samples=self.args.n_samples,
            max_tokens=self.args.max_tokens,
            temperature=self.args.sampling_temperature,
            top_p=self.args.sampling_top_p,
            top_k=self.args.sampling_top_k,
            presence_penalty=self.args.sampling_presence_penalty,
            stop=self.args.stop,
            add_stop=self.args.add_stop
        )
        num_failed_samples = 0

        for i, (n_responses, completion_tokens, total_tokens) in enumerate(results):
            assert n_responses is not None and len(n_responses) == self.args.n_samples
            num_failed_samples += sum([1 for r in n_responses if r is None])
            samples[i].responses = n_responses
            samples[i].completion_tokens = completion_tokens
            samples[i].total_tokens = total_tokens
        
        self.logger.info(f"inference service returned results, {num_failed_samples} samples failed")

        return samples, num_failed_samples

    def score(self, samples: List[MultiResponseSample]) -> List[MultiResponseSample]:
        """调用reward model打分服务，填写MultiResponseSample.rewards
        """
        for sample in samples:
            prompt = sample.prompt
            responses = sample.responses
            ground_truth = sample.ground_truth
            dataset_type = sample.dataset_type
            if dataset_type.lower() in ['code_contests', 'apps', 'taco', 'codeforces', 'leetcode', 'code']:
                pass
            rewards = []
            reward_metrics_list = []
            for response in responses:
                if response is None:
                    rewards.append(None)
                    continue
                if self.args.force_thinking:
                    response = self.args.begin_of_thinking + response
                if self.args.use_cot_reward:
                    reward, reward_metrics = compute_score_cot(dataset_type, response, ground_truth, prompt)
                else:
                    reward, reward_metrics = compute_score(dataset_type, response, ground_truth, prompt)
                rewards.append(reward)
                reward_metrics_list.append(reward_metrics)
            sample.rewards = rewards
            sample.reward_metrics_list = reward_metrics_list
        return samples
    
    def score_v2(self, samples: List[MultiResponseSample]) -> List[MultiResponseSample]:
        score_utils.score(samples)

    def postprocess_samples(self, samples: List[MultiResponseSample]) -> List[MultiResponseSample]:
        """对reward进行后处理，填写MultiResponseSample.normed_rewards；未来可以设计更多的数据处理逻辑
        """
        for sample in samples:
            valid_response_ids = [i for i, r in enumerate(sample.responses) if r is not None]
            sample.normed_rewards = [0] * len(sample.responses)
            if len(valid_response_ids) > 0:
                rewards = [sample.rewards[i] for i in valid_response_ids]
                rewards = np.array(rewards, dtype=np.float32)
                if self.args.advantage_estimator == 'grpo':
                    normed_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
                elif self.args.advantage_estimator == 'rloo':
                    baseline = (rewards.sum() - rewards) / (len(rewards) - 1) + float(1e-9)
                    normed_rewards = rewards - baseline
                elif self.args.advantage_estimator in ['reinforce++', 'dapo']:
                    normed_rewards = rewards - rewards.mean()
                for i, valid_id in enumerate(valid_response_ids):
                    sample.normed_rewards[valid_id] = float(normed_rewards[i])
        return samples

    def unfold_samples(self, samples: List[MultiResponseSample]) -> List[Sample]:
        """Unfold and shuffle samples.
        """
        all_samples = []
        for sample in samples:
            prompt = sample.prompt
            responses = sample.responses
            rewards = sample.rewards
            normed_rewards = sample.normed_rewards
            completion_tokens = sample.completion_tokens
            all_samples += [
                Sample(prompt=prompt, response=response, reward=reward, normed_reward=normed_reward, \
                       completion_tokens=completion_tokens / len(sample.responses), ground_truth=sample.ground_truth)
                for response, reward, normed_reward in zip(responses, rewards, normed_rewards)
            ]
        random.shuffle(all_samples)
        return all_samples

    def collate_fn(self, samples: List[Sample]) -> BatchExperience:
        """Padding and truncate samples, 填写BatchExperience.input_ids，labels，loss_mask，outcome_reward
        """
        self.logger.debug("collating data")
        batch_input_ids = []
        batch_labels = []
        batch_loss_mask = []
        outcome_rewards = []

        # n_samples = len(samples[0])

        for sample in samples:
            prompt_ids = self.tokenizer.encode(sample.prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(sample.response, add_special_tokens=False)
            input_ids = prompt_ids + response_ids
            loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            batch_input_ids.append(input_ids)
            batch_loss_mask.append(loss_mask)
            outcome_rewards.append(sample.normed_reward)
        
        if self.args.pad_to_max_length:
            max_length = self.args.seq_length + 1
        else:
            max_length = _ceil_to_nearest(
                max([len(input_ids) for input_ids in batch_input_ids]), 
                math.lcm(mpu.get_context_parallel_world_size() * 2, 64)
            ) + 1

        padded_input_ids = []
        padded_loss_mask = []
        for input_ids, loss_mask in zip(batch_input_ids, batch_loss_mask):
            assert len(input_ids) == len(loss_mask)
            needed = max_length - len(input_ids)
            if needed > 0:
                input_ids += [self.tokenizer.pad_token_id] * needed
                loss_mask += [0] * needed
            padded_input_ids.append(input_ids[:max_length])
            padded_loss_mask.append(loss_mask[:max_length])
        
        batch_input_ids = [input_ids[:-1] for input_ids in padded_input_ids]
        batch_labels = [input_ids[1:] for input_ids in padded_input_ids]
        batch_loss_mask = [loss_mask[1:] for loss_mask in padded_loss_mask]
        
        batch_experience = BatchExperience(
            input_ids=torch.tensor(batch_input_ids, dtype=torch.int64, device=dist_utils.get_device()),
            labels=torch.tensor(batch_labels, dtype=torch.int64, device=dist_utils.get_device()),
            loss_mask=torch.tensor(batch_loss_mask, dtype=torch.float32, device=dist_utils.get_device()),
            outcome_rewards=torch.tensor(outcome_rewards, dtype=torch.float32, device=dist_utils.get_device()),
            batch_samples=samples
        )
        return batch_experience
    
    def collate_fn_optimized(self, raw_samples: List[Sample]) -> BatchExperience:
        if self.args.skip_zero_reward_sample:
            samples: List[Sample] = self.filter_out_zero_reward_sample(raw_samples)
        else:
            samples = raw_samples
        self.logger.debug(f"drop {len(raw_samples) - len(samples)} samples")
        assert len(samples) > 0
        # prompts = [s.prompt if s.normed_reward != 0 else "zero" for s in samples]
        # responses = [s.response if s.normed_reward != 0 else "reward" for s in samples]
        prompts = [s.prompt for s in samples]
        responses = [s.response for s in samples]
        outcome_rewards = torch.tensor([s.normed_reward for s in samples], dtype=torch.float32, device=dist_utils.get_device())

        prompt_encodings = self.tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)
        response_encodings = self.tokenizer(responses, add_special_tokens=False, padding=False, truncation=False)

        batch_input_ids_list = []
        batch_loss_mask_list = []

        for i in range(len(samples)):
            prompt_ids = prompt_encodings['input_ids'][i]
            response_ids = response_encodings['input_ids'][i]
            
            input_ids = prompt_ids + response_ids
            loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            
            batch_input_ids_list.append(torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)) # 先转tensor，方便pad_sequence
            batch_loss_mask_list.append(torch.tensor(loss_mask, dtype=torch.float32, pin_memory=True))
        
        if self.args.pad_to_max_length:
            max_len_for_padding = self.args.seq_length + 1
        else:
            current_max_len = max(len(ids) for ids in batch_input_ids_list)
            max_len_for_padding = _ceil_to_nearest(
                current_max_len,
                math.lcm(mpu.get_context_parallel_world_size() * 2, 64)
            ) + 1
            max_len_for_padding = min(max_len_for_padding, self.args.seq_length + 1)

        processed_input_ids = []
        processed_loss_masks = []

        for ids, mask in zip(batch_input_ids_list, batch_loss_mask_list):
            ids = ids[:max_len_for_padding]
            mask = mask[:max_len_for_padding]
            processed_input_ids.append(ids)
            processed_loss_masks.append(mask)
        
        padded_input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
            processed_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).to(dist_utils.get_device(), non_blocking=True)

        padded_loss_mask_tensor = torch.nn.utils.rnn.pad_sequence(
            processed_loss_masks,
            batch_first=True,
            padding_value=0  # loss_mask 用0填充
        ).to(dist_utils.get_device(), non_blocking=True)

        current_padded_len = padded_input_ids_tensor.shape[1]

        if max_len_for_padding > current_padded_len:
            padding_needed = max_len_for_padding - current_padded_len
            pad_tensor_ids = torch.full((padded_input_ids_tensor.size(0), padding_needed), 
                                        self.tokenizer.pad_token_id, 
                                        dtype=torch.int64, device=dist_utils.get_device())
            pad_tensor_mask = torch.full((padded_loss_mask_tensor.size(0), padding_needed), 
                                         0, 
                                         dtype=torch.float32, device=dist_utils.get_device())

            padded_input_ids_tensor = torch.cat([padded_input_ids_tensor, pad_tensor_ids], dim=1)
            padded_loss_mask_tensor = torch.cat([padded_loss_mask_tensor, pad_tensor_mask], dim=1)
        
        final_input_ids = padded_input_ids_tensor[:, :-1].contiguous()
        final_labels = padded_input_ids_tensor[:, 1:].contiguous()
        final_loss_mask = padded_loss_mask_tensor[:, 1:].contiguous()

        return BatchExperience(
            input_ids=final_input_ids,
            labels=final_labels,
            loss_mask=final_loss_mask,
            outcome_rewards=outcome_rewards,
            batch_samples=samples
        )

    def filter_out_zero_reward_sample(self, samples: List[Sample]) -> List[Sample]:
        sorted_samples = sorted(samples, key=lambda x: abs(x.normed_reward), reverse=True)
        if self.args.reference_dp_size is not None:
            dp_size = math.lcm(mpu.get_data_parallel_world_size() * self.args.micro_forward_batch_size, self.args.reference_dp_size)
        else:
            dp_size = mpu.get_data_parallel_world_size() * self.args.micro_forward_batch_size
        dp_size = math.lcm(dp_size, mpu.get_data_parallel_world_size() * self.args.micro_batch_size)
        # 保证一个batch内的样本数可以被logits_transfer_batch_size整除
        if self.args.distillation_enabled:
            dp_size = math.lcm(dp_size, self.args.logits_transfer_batch_size)
        self.logger.debug(f"dp_size: {dp_size}")
        i = 0
        while i < len(sorted_samples):
            if sorted_samples[i].normed_reward == 0:
                break
            i += 1
        num_valid_samples = i
        if num_valid_samples % dp_size == 0 and num_valid_samples > 0:
            num_pad_samples = 0
        else:
            num_pad_samples = dp_size - num_valid_samples % dp_size
        assert (num_pad_samples + num_valid_samples) % mpu.get_data_parallel_world_size() == 0
        filtered_samples = sorted_samples[:num_valid_samples + num_pad_samples]
        random.shuffle(filtered_samples)
        return filtered_samples

    def prepare_logits_cache(self, num_batches: int):
        if self.logits_cache is not None:
            return
        logits_cache = []
        max_possible_numel = (self.args.global_batch_size // mpu.get_data_parallel_world_size()) * self.args.seq_length * self.args.padded_vocab_size // mpu.get_tensor_model_parallel_world_size()
        self.logger.info(f"prepare_logits_cache, max_possible_numel: {max_possible_numel}")
        for _ in range(num_batches):
            # logits_cache.append(torch.empty((max_possible_numel,), pin_memory=True, device='cpu'))
            logits_cache.append(torch.empty((max_possible_numel,), pin_memory=False, device='cpu'))
        self.logits_cache = logits_cache
        self.logger.info("logits_cache created")

    def wait_for_teacher_logits(self, batch_experience: BatchExperience, batch_id: int):
        # 接收teacher计算一个batch experience的logits
        # 按照logits_transfer_batch_size分批次接收
        # 通过控制num_microbatches实现分批计算
        if mpu.is_pipeline_last_stage():
            logits_transfer_batch_size = self.args.logits_transfer_batch_size // mpu.get_data_parallel_world_size()
            num_iters = (batch_experience.input_ids.shape[0] // mpu.get_data_parallel_world_size()) // logits_transfer_batch_size
            self.logger.debug(f"num_iters: {num_iters}")
            logits_shape = (batch_experience.input_ids.shape[0] // mpu.get_data_parallel_world_size(), batch_experience.input_ids.shape[1], self.args.padded_vocab_size // mpu.get_tensor_model_parallel_world_size())
            cur_numel = (batch_experience.input_ids.shape[0] // mpu.get_data_parallel_world_size()) * batch_experience.input_ids.shape[1] * (self.args.padded_vocab_size // mpu.get_tensor_model_parallel_world_size())
        if mpu.is_pipeline_last_stage():
            # batch_experience.teacher_logits = torch.empty(
            #     (batch_experience.input_ids.shape[0], batch_experience.input_ids.shape[1], self.args.padded_vocab_size // mpu.get_tensor_model_parallel_world_size()), pin_memory=True, dtype=torch.float32, device='cpu'
            # )
            batch_experience.teacher_logits = self.logits_cache[batch_id][:cur_numel].view(*logits_shape)

        if mpu.is_pipeline_last_stage():
            for i in range(num_iters):
                # 接收logits
                received_logits = self.logits_express.teacher_send_student_receive()
                offset = i * logits_transfer_batch_size
                if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                    dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                    if dump_path is None:
                        raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                    np.save(f"{dump_path}/student_recv_logits_iter_{i}_tp_{mpu.get_tensor_model_parallel_rank()}_dp_{mpu.get_data_parallel_rank()}.npy", received_logits.detach().cpu().numpy())
                # else:
                #     dump_path = "/root/work/filestorage/GroupPostTrain/taoyuyang/projects/yingxiao_kd/Megatron-RL/training_outputs/yingxiao_gkd_7B_30B/dump_dir_2"
                #     if dump_path is None:
                #         raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                #     np.save(f"{dump_path}/teacher_send_logits_iter_{i}_tp_{mpu.get_tensor_model_parallel_rank()}_dp_{mpu.get_data_parallel_rank()}.npy", received_logits.detach().cpu().numpy())
                # torch.cuda.synchronize()
                # with torch.cuda.stream(self.s1):
                #     # batch_experience.teacher_logits[offset:offset + self.args.logits_transfer_batch_size] = received_logits.to(device='cpu', non_blocking=True)
                #     dst = batch_experience.teacher_logits[offset:offset + logits_transfer_batch_size]
                #     dst.copy_(received_logits, non_blocking=True)
                # self.s1.synchronize()
                dst = batch_experience.teacher_logits[offset:offset + logits_transfer_batch_size]
                dst.copy_(received_logits, non_blocking=True)
                torch.cuda.synchronize()
            self.logger.debug("waiting for last stage to receive logits")
            dist_utils.wait_for_dp_and_cp_and_tp_neighbors()
        dist.barrier()
        self.logger.debug("finished receiving logits for a batch input")

    def compute_teacher_logits(self, batch_experiences: List[BatchExperience]):
        self.logger.info("receiving teacher logits for all batch experiences")
        if dist.get_rank() == 0:
            self.rank_0_send_input_to_ref(batch_experiences)
        dist.barrier()
        if mpu.is_pipeline_last_stage():
            self.prepare_logits_cache(len(batch_experiences))
        dist.barrier()
        for i, batch_experience in tqdm(enumerate(batch_experiences), desc="waiting for teacher logits", total=len(batch_experiences)):
            self.wait_for_teacher_logits(batch_experience, i)
        self.logger.info("finished receiving all teacher logits")

    def actor_forward_only(self, batch_experience_list: List[BatchExperience]):
        """填写BatchExperience.old_actor_logps
        """
        if not self.is_old_actor_logp_required:
            self.logger.debug("skipping computing old actor logp")
            return
        args = self.args
        # num_microbatches = args.global_batch_size // (args.micro_forward_batch_size * mpu.get_data_parallel_world_size())
        self.logger.debug("computing old actor logps")

        data_iter = None
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            data_loader = create_distributed_dataloader(batch_experience_list, batch_size=args.micro_forward_batch_size)
            data_iter = iter(data_loader)

        if args.use_distributed_optimizer and args.overlap_param_gather:
            disable_forward_pre_hook(self.model)

        for i, batch_experience in tqdm(enumerate(batch_experience_list), desc="computing actor log probs", total=len(batch_experience_list)):
            num_microbatches = self.n_batches_list[0][i]
            self.logger.debug(f"num_microbatches: {num_microbatches}")
            old_actor_logps = compute_batch_logp(
                model=self.model,
                data_iterator=data_iter,
                num_microbatches=num_microbatches,
                micro_batch_size=args.micro_forward_batch_size
            )
            if dist.get_rank() == 0:
                with torch.cuda.stream(self.s1):
                    batch_experience.old_actor_logps.copy_(old_actor_logps, non_blocking=True)
                self.s1.synchronize()

        if dist.get_rank() == 0 or (mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0):
            assert old_actor_logps is not None
            assert old_actor_logps.shape == batch_experience.input_ids.shape
            if dist.get_rank() != 0:
                old_actor_logps = None # 重新同步
        self.logger.debug("old actor logps calculated")
        torch.cuda.empty_cache()
        
        # 在pp[-1]内同步
        # TODO: 在pp[0]内有不必要的同步
        self.logger.debug("syncing old actor logps")
        for batch_experience in tqdm(batch_experience_list, desc="syncing actor log probs"):
            if dist.get_rank() == 0:
                dist_utils._sync_2D_input_data(batch_experience.old_actor_logps.to(device=dist_utils.get_device(), non_blocking=True), dtype=torch.float32)
            elif mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                old_actor_logps = dist_utils._sync_2D_input_data(None, dtype=torch.float32)
                with torch.cuda.stream(self.s1):
                    batch_experience.old_actor_logps.copy_(old_actor_logps, non_blocking=True)
                self.s1.synchronize()

        dist.barrier()
        torch.cuda.empty_cache()
        self.logger.debug("old actor logps synced")

        if args.use_distributed_optimizer and args.overlap_param_gather:
            enable_forward_pre_hook(self.model)
        
    def rank_0_send_input_to_ref(self, batch_experiences: List[BatchExperience]):
        """填写BatchExperience.ref_logps
        """
        dist.barrier(self.data_updater.update_group)
        self.logger.debug("sending number batches")
        num_batches_tensor = torch.tensor([len(batch_experiences)], device=dist_utils.get_device(), dtype=torch.int64)
        self.data_updater.actor_send_reference_receive_2D_tensor(num_batches_tensor.unsqueeze(dim=1), dtype=torch.int64, shape_tensor=torch.LongTensor([1, 1]))
        self.logger.debug("sending input to ref")
        for batch_experience in tqdm(batch_experiences, desc="sending input to ref"):
            input_ids = batch_experience.input_ids.to(device=dist_utils.get_device(), non_blocking=True)
            labels = batch_experience.labels.to(device=dist_utils.get_device(), non_blocking=True)
            self.data_updater.actor_send_reference_receive_2D_tensor(input_ids, dtype=torch.int64)
            self.data_updater.actor_send_reference_receive_2D_tensor(labels, dtype=torch.int64, shape_tensor=torch.LongTensor([input_ids.shape[0], input_ids.shape[1]]))

    def receive_and_sync_output_from_ref(self, batch_experiences: List[BatchExperience]):
        """填写BatchExperience.ref_logps
        """
        if dist.get_rank() == 0:
            dist.barrier(self.data_updater.update_group)
        dist.barrier()
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            for batch_experience in tqdm(batch_experiences, desc="receiving and syncing ref log probs"):
                if dist.get_rank() == 0:
                    ref_logps = self.data_updater.actor_receive_reference_send_2D_tensor(None, torch.float32)
                    dist_utils._sync_2D_input_data(ref_logps, dtype=torch.float32)
                else:
                    ref_logps = dist_utils._sync_2D_input_data(None, dtype=torch.float32)
                with torch.cuda.stream(self.s1):
                    batch_experience.ref_logps.copy_(ref_logps, non_blocking=True)
                self.s1.synchronize()
        dist.barrier()
        self.logger.debug("received logp from ref")

    @torch.no_grad()
    def compute_advantages(self, batch_experiences: List[BatchExperience]):
        """计算并填写BatchExperience.token_reward, advantages
        """
        if self.args.use_outcome_rewards_as_advantages:
            self.logger.debug(f"skipping computing advantages for use_outcome_rewards_as_advantages: {self.args.use_outcome_rewards_as_advantages}")
            return None
        def compute_and_sync(batch_experience: BatchExperience):
            advantages = None
            kls = None
            if dist.get_rank() == 0:
                with torch.cuda.stream(self.s1):
                    if self.is_old_actor_logp_required:
                        old_actor_logps = batch_experience.old_actor_logps.to(device=dist_utils.get_device(), non_blocking=True)
                    if self.args.init_kl_coef > 1e-8 and not self.args.use_kl_loss:
                        ref_logps = batch_experience.ref_logps.to(device=dist_utils.get_device(), non_blocking=True)
                with torch.cuda.stream(self.s2):
                    outcome_rewards = batch_experience.outcome_rewards.to(device=dist_utils.get_device(), non_blocking=True)
                    loss_mask = batch_experience.loss_mask.to(device=dist_utils.get_device(), non_blocking=True)
                self.s1.synchronize()
                self.s2.synchronize()

                # compute kl
                if self.args.init_kl_coef > 1e-8 and not self.args.use_kl_loss:
                    kls = compute_approx_kl(
                        old_actor_logps,
                        ref_logps,
                        loss_mask,
                        kl_estimator=self.args.kl_estimator,
                    )
                else:
                    kls = None

                self.logger.debug("computing token reward")
                token_rewards = compute_token_reward(
                    outcome_rewards,
                    self.args.init_kl_coef,
                    kls,
                    loss_mask=loss_mask,
                    reward_clip_range=None,
                )
                
                self.logger.debug("computing returns")
                advantages = compute_returns(
                    token_rewards,
                    loss_mask,
                    gamma=1.0
                )
                self.logger.debug("syncing advantages")
                dist_utils._sync_2D_input_data(advantages, dtype=torch.float32)
                with torch.cuda.stream(self.s1):
                    batch_experience.advantages.copy_(advantages, non_blocking=True)
                self.s1.synchronize()
            else:
                self.logger.debug("syncing advantages")
                advantages = dist_utils._sync_2D_input_data(advantages, dtype=torch.float32)
                with torch.cuda.stream(self.s1):
                    batch_experience.advantages.copy_(advantages, non_blocking=True)
                self.s1.synchronize()
            self.logger.debug("waiting for advantages syncing")
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            for batch_experience in tqdm(batch_experiences, desc="computing advantages"):
                compute_and_sync(batch_experience)
        
        torch.cuda.empty_cache()
        dist.barrier()

    def train_step(self, data_iterator: Optional[Iterator], num_microbatches: int, iter_num: int = None):
        self.logger.debug("train step")
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optim.zero_grad()

        def loss_func(
            output_tensor: torch.Tensor,
            labels: torch.Tensor,
            loss_mask: torch.Tensor,
            non_loss_data: bool = False,
            old_actor_logps: Optional[torch.Tensor] = None,
            ref_logps: Optional[torch.Tensor] = None,
            advantages: Optional[torch.Tensor] = None,
            debug_i: Optional[torch.Tensor] = None,
            outcome_rewards: torch.Tensor = None
        ):
            assert non_loss_data == False
            if self.is_old_actor_logp_required:
                assert old_actor_logps is not None

            actor_logps = from_parallel_logits_to_logprobs(output_tensor, labels, higher_stability=True, ignore_last=False)
            kl = None

            clip_eps = self.args.clip_eps
            if self.is_old_actor_logp_required:
                logp_diff = actor_logps - old_actor_logps
                ratio = logp_diff.exp()
            else:
                logp_diff = actor_logps - actor_logps.detach()
                ratio = logp_diff.exp()

            if self.args.use_outcome_rewards_as_advantages:
                self.logger.debug("using outcome rewards as advantages")
                assert outcome_rewards is not None
                surr1 = ratio * outcome_rewards
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * outcome_rewards
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = masked_mean(actor_loss, loss_mask, dim=-1) # (batch,)
            else:
                assert advantages is not None
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = masked_mean(actor_loss, loss_mask, dim=-1) # (batch,)
            if mpu.get_context_parallel_world_size() > 1:
                # TODO: there should be a total_length, then allreduce(cp_rank_loss / total_length) 
                dist.all_reduce(actor_loss, op=ReduceOp.AVG, group=mpu.get_context_parallel_group())
            policy_loss = actor_loss.mean()

            # kl loss
            kl_loss = None
            if (self.args.init_kl_coef > 1e-8) and self.args.use_kl_loss:
                kl = compute_approx_kl(
                    actor_logps, 
                    ref_logps, 
                    kl_estimator=self.args.kl_estimator
                )
                kl_loss = masked_mean(kl, loss_mask, dim=-1) # (batch,)
                if mpu.get_context_parallel_world_size() > 1:
                    # TODO: there should be a total_length, then allreduce(cp_rank_loss / total_length)
                    dist.all_reduce(kl_loss, op=ReduceOp.AVG, group=mpu.get_context_parallel_group())
                kl_loss = kl_loss.mean()

            # entropy loss
            entropy_loss = None
            if self.args.entropy_loss_coef > -1:
                unmasked_entropy_loss = vocab_parallel_entropy(output_tensor) # (batch, length)
                entropy_loss = masked_mean(unmasked_entropy_loss, loss_mask, dim=-1) # (batch,)
                if mpu.get_context_parallel_world_size() > 1:
                    dist.all_reduce(entropy_loss, op=ReduceOp.AVG, group=mpu.get_context_parallel_group())
                entropy_loss = entropy_loss.mean()
            
            is_deviant = False
            deviant_reason = ""
            if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                is_deviant = True
                deviant_reason += " invalid policy loss"
            elif self.args.policy_loss_limit is not None and policy_loss > self.args.policy_loss_limit:
                is_deviant = True
                deviant_reason += f" policy loss limit exceeded: {policy_loss.item()}"

            if kl_loss is not None:
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    is_deviant = True
                    deviant_reason += " invalid kl loss"
                elif self.args.kl_loss_limit is not None and kl_loss > self.args.kl_loss_limit:
                    is_deviant = True
                    deviant_reason += f" kl loss limit exceeded: {kl_loss.item()}"

            if entropy_loss is not None:
                if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                    is_deviant = True
                    deviant_reason += " invalid entropy loss"
            
            if is_deviant:
                loss = torch.sum(actor_logps * 0.0)
                self.logger.warning(f"deviant experience: {deviant_reason}")
                if self.args.dump_experience and mpu.get_tensor_model_parallel_rank() == 0:
                    tensors = dict(
                        output_tensor = output_tensor,
                        labels = labels,
                        loss_mask=loss_mask,
                        old_actor_logps = old_actor_logps,
                        ref_logps = ref_logps,
                        advantages = advantages,
                        outcome_rewards = outcome_rewards,
                        actor_logps = actor_logps,
                        kl=kl
                    )
                    self.dump_deviant_experience(self.args.dump_path, iter_num, debug_i[:, 0].tolist(), [], deviant_reason, **tensors)
            else:
                loss = self.args.policy_loss_coef * policy_loss
                if kl_loss is not None:
                    if self.args.negate_kl_loss:
                        loss -= self.args.init_kl_coef * kl_loss
                    else:
                        loss += self.args.init_kl_coef * kl_loss
                if entropy_loss is not None and entropy_loss < self.args.entropy_loss_threshold:
                    loss -= self.args.entropy_loss_coef * entropy_loss
                if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1 and mpu.get_tensor_model_parallel_rank() == 0:
                    dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                    if dump_path is None:
                        raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                    tensors = dict(
                        output_tensor = output_tensor,
                        labels = labels,
                        loss_mask=loss_mask,
                        old_actor_logps = old_actor_logps,
                        ref_logps = ref_logps,
                        advantages = advantages,
                        outcome_rewards = outcome_rewards,
                        actor_logps = actor_logps,
                        kl=kl,
                        entropy=entropy_loss
                    )
                    os.makedirs(f"{dump_path}/loss_func", exist_ok=True)
                    self.dump_deviant_experience(f"{dump_path}/loss_func", iter_num, debug_i[:, 0].tolist(), [], "LIGHT_SCALE_DUMP", **tensors)
            losses_reduced = torch.cat(
                [
                    loss.clone().detach().unsqueeze(dim=0),
                    policy_loss.clone().detach().unsqueeze(dim=0),
                    kl_loss.clone().detach().unsqueeze(dim=0) if kl_loss is not None else \
                        torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                    entropy_loss.clone().detach().unsqueeze(dim=0) if entropy_loss is not None else \
                        torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                ], dim=0
            )
            dist.all_reduce(losses_reduced, op=ReduceOp.AVG, group=mpu.get_data_parallel_group())

            return loss, {"losses": losses_reduced}

        def megatron_forward_step(data_iterator, model):
            forward_args = {
                "input_ids": None,
                "position_ids": None,
                "attention_mask": None
            }
            loss_func_args = {
                "labels": None,
                "old_actor_logps": None,
                "ref_logps": None,
                "advantages": None,
                "loss_mask": None,
                "outcome_rewards": None,
                "debug_i": None
            }
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                assert data_iterator is not None
                batched_item: Dict[str, torch.Tensor] = next(data_iterator)
                if mpu.is_pipeline_first_stage():
                    forward_args["input_ids"] = batched_item["input_ids"].to(device=dist_utils.get_device(), non_blocking=True)
                if mpu.is_pipeline_last_stage():
                    loss_func_args["labels"] = batched_item["labels"].to(device=dist_utils.get_device(), non_blocking=True)
                    loss_func_args["loss_mask"] = batched_item["loss_mask"].to(device=dist_utils.get_device(), non_blocking=True)
                    loss_func_args["debug_i"] = batched_item["idx"]
                    loss_func_args["outcome_rewards"] = batched_item["outcome_rewards"].to(device=dist_utils.get_device(), non_blocking=True)
                    if self.is_old_actor_logp_required:
                        loss_func_args["old_actor_logps"] = batched_item["old_actor_logps"].to(device=dist_utils.get_device(), non_blocking=True)
                    if not self.args.use_outcome_rewards_as_advantages:
                        loss_func_args["advantages"] = batched_item["advantages"].to(device=dist_utils.get_device(), non_blocking=True)
                    if self.args.init_kl_coef > 1e-8:
                        loss_func_args["ref_logps"] = batched_item["ref_logps"].to(device=dist_utils.get_device(), non_blocking=True)

                if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1 and mpu.get_tensor_model_parallel_rank() == 0:
                    # for debug
                    dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                    if dump_path is None:
                        raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                    tensors = dict(
                        input_ids=batched_item["input_ids"],
                        labels = batched_item["labels"],
                        loss_mask = batched_item["loss_mask"]
                    )
                    self.dump_deviant_experience(dump_path, iter_num, batched_item["idx"][:, 0].tolist(), [], "LIGHT_SCALE_DUMP", **tensors)

            output_tensor = model(**forward_args)

            return output_tensor, partial(loss_func, **loss_func_args)

        reduced_losses = get_forward_backward_func()(
            forward_step_func=megatron_forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=False,
            seq_length=self.args.seq_length,
            micro_batch_size=self.args.micro_batch_size
        )

        self.logger.debug("received reduced losses")

        update_successful, grad_norm, num_zeros_in_grad = self.optim.step()
        if update_successful:
            self.scheduler.step(increment=self.args.global_batch_size)
        else:
            self.logger.warning("optim update unsuccessful")

        if reduced_losses:
            losses_for_report = torch.cat([item["losses"].unsqueeze(dim=1) for item in reduced_losses], dim=1).mean(dim=1)
        else:
            losses_for_report = torch.zeros((4,), dtype=torch.float32, device=dist_utils.get_device())

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
                dist.send(losses_for_report, dst=0, group=mpu.get_pipeline_model_parallel_group())
            elif dist.get_rank() == 0:
                dist.recv(losses_for_report, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
            dist.barrier()

        loss_item = losses_for_report[0].item()
        policy_loss_item = losses_for_report[1].item()
        kl_loss_item = losses_for_report[2].item()
        entropy_loss_item = losses_for_report[3].item()
        lr = None
        for param_group in self.optim.param_groups:
            if param_group['is_decoupled_lr']:
                lr = param_group['lr']
            else:
                lr = param_group['lr']
        self.logger.info(f"iter: {iter_num}, loss: {loss_item}, policy_loss: {policy_loss_item}, kl_loss: {kl_loss_item}, entropy_loss: {entropy_loss_item}, \
                         grad_norm: {grad_norm}, learning_rate: {lr}")
        
        if dist.get_rank() == 0:
            writer = self.tensorboard_writer
            writer.add_scalar("train/loss", loss_item, iter_num)
            writer.add_scalar("train/policy_loss", policy_loss_item, iter_num)
            writer.add_scalar("train/kl_loss", kl_loss_item, iter_num)
            writer.add_scalar("train/grad_norm", grad_norm, iter_num)
            writer.add_scalar("train/learning_rate", lr, iter_num)
            writer.add_scalar("train/entropy_loss", entropy_loss_item, iter_num)
            writer.flush()
        dist.barrier()

    def distillation_step(self, data_iterator: Optional[Iterator], num_microbatches: int, iter_num: int = None):
        self.logger.debug("distillation train step")
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optim.zero_grad()

        def loss_func(
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            loss_mask: torch.Tensor,
            beta=0.5, 
            student_temperature=1.0,
            teacher_temperature=1.0,
            reduction="batchmean",
            non_loss_data: bool = False,
            debug_i: Optional[torch.Tensor] = None
        ):
            assert non_loss_data == False

            teacher_logits = teacher_logits / teacher_temperature
            student_logits = student_logits / student_temperature

            teacher_log_probs = distributed_log_softmax(teacher_logits)
            student_log_probs = distributed_log_softmax(student_logits)

            if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                tensors = dict(
                    teacher_logits=teacher_logits,
                    student_logits=student_logits,
                    teacher_log_probs=teacher_log_probs,
                    student_log_probs=student_log_probs
                )
                self.dump_deviant_experience(dump_path, iter_num, debug_i[:, 0].tolist(), [], "LIGHT_SCALE_DUMP", **tensors)

            if beta == 0:
                self.logger.debug("using forward kl")
                local_jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
            elif beta == 1:
                self.logger.debug("using reverse kl")
                local_jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
            else:
                self.logger.debug("using jsd")
                # Compute the log of the mixture distribution
                # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
                beta = torch.tensor(beta, dtype=student_log_probs.dtype)
                mixture_log_probs = torch.logsumexp(
                    torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]),
                    dim=0,
                )

                # Compute KL divergences using F.kl_div
                # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
                kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
                kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

                # Compute the Generalized Jensen-Shannon Divergence
                local_jsd = beta * kl_teacher + (1 - beta) * kl_student

            if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                tensors = dict(
                    local_jsd=local_jsd
                )
                self.dump_deviant_experience(dump_path, iter_num, debug_i[:, 0].tolist(), [], "LIGHT_SCALE_DUMP", **tensors)

            local_jsd = local_jsd.sum(dim=-1)  # Sum over vocab dimension
            global_jsd = local_jsd.clone()
            dist.all_reduce(global_jsd, op=ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

            if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                tensors = dict(
                    global_jsd=global_jsd
                )
                self.dump_deviant_experience(dump_path, iter_num, debug_i[:, 0].tolist(), [], "LIGHT_SCALE_DUMP", **tensors)

            global_jsd = global_jsd * loss_mask  # Apply loss mask

            jsd_loss = global_jsd.sum() / (loss_mask.sum() + 1e-8)  # Mean over non-masked elements
            # self.logger.debug(f"jsd_loss: {jsd_loss}")
            # self.logger.debug(f"jsd loss is zero: {torch.allclose(jsd_loss, torch.zeros_like(jsd_loss), atol=1e-12)}")

            if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                tensors = dict(
                    jsd_loss=jsd_loss,
                    loss_mask=loss_mask
                )
                self.dump_deviant_experience(dump_path, iter_num, debug_i[:, 0].tolist(), [], "LIGHT_SCALE_DUMP", **tensors)

            losses_reduced = torch.cat(
                [
                    jsd_loss.clone().detach().unsqueeze(dim=0)
                ], dim=0
            )
            dist.all_reduce(losses_reduced, op=ReduceOp.AVG, group=mpu.get_data_parallel_group())

            if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                tensors = dict(
                    teacher_logits=teacher_logits,
                    student_logits=student_logits,
                    teacher_log_probs=teacher_log_probs,
                    student_log_probs=student_log_probs
                )
                self.dump_deviant_experience(dump_path, iter_num, debug_i[:, 0].tolist(), [], "LIGHT_SCALE_DUMP", **tensors)

            return jsd_loss, {"losses": losses_reduced}
        
        def megatron_forward_step(data_iterator, model):
            forward_args = {
                "input_ids": None,
                "position_ids": None,
                "attention_mask": None
            }
            loss_func_args = {
                "teacher_logits": None,
                "loss_mask": None,
                "debug_i": None,
                "beta": self.args.gkd_beta,
                "student_temperature": self.args.student_temperature,
                "teacher_temperature": self.args.teacher_temperature
            }
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                assert data_iterator is not None
                batched_item: Dict[str, torch.Tensor] = next(data_iterator)
                if mpu.is_pipeline_first_stage():
                    forward_args["input_ids"] = batched_item["input_ids"].to(device=dist_utils.get_device(), non_blocking=True)
                if mpu.is_pipeline_last_stage():
                    loss_func_args["teacher_logits"] = batched_item["teacher_logits"].to(device=dist_utils.get_device(), non_blocking=True)
                    loss_func_args["loss_mask"] = batched_item["loss_mask"].to(device=dist_utils.get_device(), non_blocking=True)
                    loss_func_args["debug_i"] = batched_item["idx"]

            output_tensor = model(**forward_args)

            return output_tensor, partial(loss_func, **loss_func_args)
        
        reduced_losses = get_forward_backward_func()(
            forward_step_func=megatron_forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=False,
            seq_length=self.args.seq_length,
            micro_batch_size=self.args.micro_batch_size
        )

        self.logger.debug("received reduced losses")

        update_successful, grad_norm, num_zeros_in_grad = self.optim.step()
        if update_successful:
            self.scheduler.step(increment=self.args.global_batch_size)
        else:
            self.logger.warning("optim update unsuccessful")

        if reduced_losses:
            losses_for_report = torch.cat([item["losses"].unsqueeze(dim=1) for item in reduced_losses], dim=1).mean(dim=1)
        else:
            losses_for_report = torch.zeros((1,), dtype=torch.float32, device=dist_utils.get_device())

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
                dist.send(losses_for_report, dst=0, group=mpu.get_pipeline_model_parallel_group())
            elif dist.get_rank() == 0:
                dist.recv(losses_for_report, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
            dist.barrier()

        loss_item = losses_for_report[0].item()
        lr = None
        for param_group in self.optim.param_groups:
            if param_group['is_decoupled_lr']:
                lr = param_group['lr']
            else:
                lr = param_group['lr']
        
        self.logger.info(f"iter: {iter_num}, loss: {loss_item}, \
                         grad_norm: {grad_norm}, learning_rate: {lr}")
        if dist.get_rank() == 0:
            writer = self.tensorboard_writer
            writer.add_scalar("train/loss", loss_item, iter_num)
            writer.add_scalar("train/grad_norm", grad_norm, iter_num)
            writer.add_scalar("train/learning_rate", lr, iter_num)
            writer.flush()
        dist.barrier()

    def update_params_to_sglang(self):
        torch.cuda.synchronize()
        dense_weight_updater = self.dense_weight_updater
        moe_weight_updater = self.moe_weight_updater
        if dense_weight_updater is not None:
            if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
                disable_forward_pre_hook(self.model)
            self.logger.info("updating dense model weight to inference service")
            dense_weight_updater()
            if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
                enable_forward_pre_hook(self.model)
        dist.barrier()
        if moe_weight_updater is not None:
            if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
                disable_forward_pre_hook(self.model)
            self.logger.info("updating moe model weight to inference service")
            moe_weight_updater()
            if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
                enable_forward_pre_hook(self.model)
        dist.barrier()

    def save_logs_and_checkpoints(self, passed_iters):
        self.logger.info(f"saving checkpoint iter {passed_iters}")
        if self.logits_cache is not None:
            self.logger.info(f"delete logits cache")
            # del self.logits_cache
            # self.logits_cache = None
        def save_checkpoint(passed_iters, model, optim, scheduler):
            args = get_args()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                disable_forward_pre_hook(model)
            save_megatron_checkpoint(passed_iters, model, optim, scheduler, num_floating_point_operations_so_far=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                enable_forward_pre_hook(model)
        save_checkpoint(passed_iters, self.model, self.optim, self.scheduler)