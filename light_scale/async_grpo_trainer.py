from typing import List, Optional, Tuple, Union, Iterator, Dict
import os
from numbers import Number

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

from collections import deque

from queue import Queue as MpQueue
from queue import Empty as QueueEmpty

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
from light_scale.logger_utils import setup_logger_v2_main_process
import light_scale.logger_utils as logger_utils
from megatron.training.checkpointing import save_checkpoint as save_megatron_checkpoint
from megatron.training.training import enable_forward_pre_hook, disable_forward_pre_hook
from torch.utils.tensorboard import SummaryWriter
from verifier.rule_based_rm_cot import compute_score as compute_score_cot
from verifier.rule_based_rm import compute_score
from light_scale.sync_processor import ActorReferenceDataUpdater
import math
from light_scale import score_utils
# from multiprocessing.synchronize import Event as MpEvent
from threading import Event as MpEvent
from threading import Thread

from megatron.core.utils import (
    get_model_config
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads

from light_scale.dataset import create_distributed_dataloader
from light_scale.data import BatchExperience, Message, MultiResponseSample, Sample
from light_scale.logits_express import LogitsExpress
from light_scale.gkd_utils import distributed_log_softmax, safe_all_reduce, tp_sum_forward_identity_backward

from light_scale.gkd_utils import (
    distributed_log_softmax,
    distributed_sparse_log_softmax,
    safe_all_reduce,
    tp_sum_forward_identity_backward,
    get_tp_vocab_range,
    scatter_sparse_to_local_vocab_dense,
)


import torch.nn.functional as F

import torch.distributed.nn.functional as FF

from light_scale.async_rollout_v2.rollout_thread import rollout_thread_main
from light_scale.sync_processor import SGLangSaver

from light_scale.distributed_lock import LockServerProcess, DistributedLock

from light_scale.weight_utils_v2 import DenseWeightUpdater, MoeWeightUpdater

import yaml

from concurrent.futures import ThreadPoolExecutor

PROFILE_RANK=-1

# copy from NeMo
def _ceil_to_nearest(n, m, ceil_to_power_2=False):
    if ceil_to_power_2:
        # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
        return 2 ** math.ceil(math.log2(n))
    else:
        return (n + m - 1) // m * m

class GRPOTrainer:
    AVG_METRICS = ['reward', 'completion_tokens', 'total_tokens']

    @staticmethod
    def _build_legacy_messages(prompt: str, response: str) -> List[Message]:
        return [
            Message(content=prompt, is_masked=True),
            Message(content=response, is_masked=False),
        ]

    def _messages_to_token_ids_and_loss_mask(self, messages: List[Message]) -> Tuple[List[int], List[float]]:
        input_ids: List[int] = []
        loss_mask: List[float] = []
        for message in messages:
            message_ids = self.tokenizer.encode(message.content or "", add_special_tokens=False)
            input_ids.extend(message_ids)
            loss_mask.extend([0.0 if message.is_masked else 1.0] * len(message_ids))
        return input_ids, loss_mask

    @staticmethod
    def _sample_uses_teacher_logps(sample: Sample) -> bool:
        return (
            sample.content_ids is not None
            or sample.loss_mask is not None
            or sample.teacher_log_probs is not None
        )

    @staticmethod
    def _sample_to_dump_dict(sample: Sample) -> Dict[str, object]:
        content_ids = sample.content_ids
        loss_mask = sample.loss_mask
        teacher_log_probs = sample.teacher_log_probs
        return {
            "prompt": sample.prompt,
            "response": sample.response,
            "reward": sample.reward,
            "normed_reward": sample.normed_reward,
            "completion_tokens": sample.completion_tokens,
            "reward_metrics": sample.reward_metrics,
            "ground_truth": sample.ground_truth,
            "dataset_type": sample.dataset_type,
            "sample_id": sample.sample_id,
            "content_len": None if content_ids is None else int(len(content_ids)),
            "loss_mask_sum": None if loss_mask is None else float(np.asarray(loss_mask).sum()),
            "teacher_log_probs_len": None if teacher_log_probs is None else int(len(teacher_log_probs)),
        }

    def __init__(
            self, 
            passed_iters: int, 
            model,
            optim,
            scheduler,
            data_updater: ActorReferenceDataUpdater,
            logits_express: Optional[LogitsExpress] = None
        ):
        args = get_args()
        log_level_name = getattr(args, "light_scale_log_level", getattr(args, "mrl_log_level", "info"))
        log_level = getattr(logging, str(log_level_name).upper())
        logger = setup_logger_v2_main_process("light_scale", level=log_level)
        logger.warning(f"seed: {args.seed}")
        random.seed(args.seed) # TODO: 复用megatron rng state

        train_batch_size = args.global_batch_size

        assert args.rollout_batch_size * args.n_samples == train_batch_size

        tokenizer = None
        tensorboard_writer = None
        if dist.get_rank() == 0:
            assert args.tensorboard_dir is not None
            tensorboard_writer = SummaryWriter(
                log_dir=args.tensorboard_dir,
                filename_suffix="light_scale"
            )

            tokenizer = get_tokenizer()._tokenizer
            assert tokenizer.pad_token_id is not None

            self.rollout_metrics = dict()

            self.saver_thread_pool = ThreadPoolExecutor(max_workers=1)

        self.tokenizer = tokenizer
        self.tensorboard_writer = tensorboard_writer

        self.args = args
        self.passed_iters = passed_iters
        self.passed_iters_this_run = passed_iters
        self.model = model
        self.optim = optim
        self.scheduler = scheduler

        self.data_updater = data_updater
        self.logger = logger
        self.log_level = log_level
        self.logits_express = logits_express

        self.is_old_actor_logp_required = False

        self.s1 = torch.cuda.Stream()
        self.s2 = torch.cuda.Stream()

        self.n_batches_list = None

        self.logits_cache = None
        self.topk_indices_cache = None
        self.topk_values_cache = None

        self.minimum_train_batch_size = self._get_minimum_train_batch_size()
        self.max_train_batch_size = self._get_max_train_batch_size()
        # self.minimum_train_batch_size = self.args.global_batch_size
        logger.warning(f"minimum_train_batch_size: {self.minimum_train_batch_size}")
        logger.warning(f"max_train_batch_size: {self.max_train_batch_size}")

        self.long_wait_group = None # 定义一个包含所有rank的group，用于长时间等待的barrier

        self.input_queue = None
        self.output_queue = None
        self.stop_event = None
        self.rollout_process = None

        self.dense_weight_updater = None
        self.moe_weight_updater = None

        self.init_success = False
        self._initialize()

        self.prev_save_future = None
        
        self.do_profile = True

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

    def _get_minimum_train_batch_size(self) -> int:
        if self.args.reference_dp_size is not None:
            dp_size = math.lcm(mpu.get_data_parallel_world_size() * self.args.micro_forward_batch_size, self.args.reference_dp_size)
        else:
            dp_size = mpu.get_data_parallel_world_size() * self.args.micro_forward_batch_size
        dp_size = math.lcm(dp_size, mpu.get_data_parallel_world_size() * self.args.micro_batch_size)
        # 保证一个batch内的样本数可以被logits_transfer_batch_size整除
        if self.args.distillation_enabled:
            dp_size = math.lcm(dp_size, self.args.logits_transfer_batch_size)
        self.logger.debug(f"dp_size: {dp_size}")
        dp_size = math.lcm(dp_size, self.args.n_samples)
        return dp_size

    def _get_max_train_batch_size(self) -> int:
        """返回每个异步训练 step 的最大 train batch size。

        约定：
        - -1 表示不限制；
        - 否则会按 `minimum_train_batch_size` 向下对齐，且最小不低于 `minimum_train_batch_size`。
        """
        raw_max = int(getattr(self.args, "max_train_batch_size", -1))
        assert raw_max == -1 or raw_max > 0, "max_train_batch_size must be -1 or a positive integer"
        if raw_max == -1:
            return -1

        min_bs = int(self.minimum_train_batch_size)
        aligned_max = (raw_max // min_bs) * min_bs
        if aligned_max < min_bs:
            self.logger.warning(
                f"max_train_batch_size={raw_max} < minimum_train_batch_size={min_bs}, "
                f"fallback to minimum_train_batch_size={min_bs}"
            )
            return min_bs

        if aligned_max != raw_max:
            self.logger.warning(
                f"max_train_batch_size={raw_max} is not divisible by minimum_train_batch_size={min_bs}, "
                f"aligned to {aligned_max}"
            )
        return aligned_max
    
    def _initialize(self):
        long_wait_group = dist.new_group(ranks=list(range(dist.get_world_size())), backend='gloo')
        dist.barrier(group=long_wait_group)
        self.long_wait_group = long_wait_group

        '''
    rollout_cfg_path: str,
    input_queue: MpQueue,
    output_queue: MpQueue,
    stop_event: MpEvent,
    logging_queue: MpQueue,
    start_event: MpEvent,
    failed_event: MpEvent,        
        '''

        init_success = False
        if dist.get_rank() == 0:
            rollout_cfg_path = self.args.async_rollout_cfg_path
            input_queue = MpQueue()
            output_queue = MpQueue()
            logging_queue = logger_utils._LOGGING_QUEUE
            start_event = MpEvent()
            stop_event = MpEvent()
            failed_event = MpEvent()

            rollout_process = Thread(
                target=rollout_thread_main,
                args=(
                    rollout_cfg_path,
                    self.passed_iters,
                    self.args.rollout_batch_size,
                    input_queue,
                    output_queue,
                    stop_event,
                    logging_queue,
                    start_event,
                    failed_event,
                    self.log_level,
                ),
                daemon=True,
            )
            rollout_process.start()

            while not start_event.is_set() and not failed_event.is_set():
                time.sleep(1.0)

            init_success = start_event.is_set() and not failed_event.is_set()
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.stop_event = stop_event
            self.rollout_process = rollout_process

        dist.barrier(group=self.long_wait_group)
        init_success = torch.tensor([int(init_success)], dtype=torch.int32, device=dist_utils.get_device())
        dist.broadcast(init_success, src=0)
        if not bool(init_success.item()):
            raise RuntimeError("Rollout process initialization failed.")
        
        sync_processor = None
        if dist_utils.is_pp_src_rank():
            dist_lock = DistributedLock(
                "weight_update",
                host=os.environ['MASTER_ADDR'],
                port=self.args.dist_lock_server_port
            )
            actor_url_list, actor_world_size = self._get_actor_rollout_config()
            sync_processor = SGLangSaver(
                actor_url_list,
                actor_world_size,
                mpu.get_pipeline_model_parallel_rank(),
                dist_lock,
                self.args.weight_update_group_port + int(mpu.get_pipeline_model_parallel_rank())
            )
        dist.barrier(group=self.long_wait_group)

        if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
            dense_weight_updater = DenseWeightUpdater(self.model, sync_processor)
            self.dense_weight_updater = dense_weight_updater
        if self.args.num_experts is not None and self.args.num_experts > 0 and mpu.get_expert_data_parallel_rank() == 0:
            moe_weight_updater = MoeWeightUpdater(self.model, sync_processor)
            self.moe_weight_updater = moe_weight_updater
        dist.barrier(group=self.long_wait_group)
        self.logger.info("initialized weight updater")

        if self.passed_iters > 0:
            if self.dense_weight_updater is not None:
                self.logger.info("updating dense model weight to inference service")
                self.dense_weight_updater()
            dist.barrier()
            if self.moe_weight_updater is not None:
                self.logger.info("updating moe model weight to inference service")
                self.moe_weight_updater()
            dist.barrier()
        
        self.init_success = bool(init_success.item())

    def _get_actor_rollout_config(self):
        with open(self.args.async_rollout_cfg_path) as f:
            async_rollout_cfg = yaml.safe_load(f)
        actor_service_name = async_rollout_cfg["actor_service_name"]
        resources = None
        for service in async_rollout_cfg["services"]:
            if service["name"] == actor_service_name:
                resources = service["resources"]
                break
        assert resources is not None, f"Cannot find actor service config for {actor_service_name}"
        base_url_list = [resource["base_url"] for resource in resources]
        world_size = sum(resource["num_gpus"] for resource in resources)
        return base_url_list, world_size
        

    def async_train_batch(self, rollout_step: int, rollout_output_queue: MpQueue):
        # 进行一次rollout batch的异步多次训练
        self.zero_grad()
        num_received_batch_samples: int = 0 # 从rollout队列中获取到的样本数
        current_samples: deque = deque() if dist.get_rank() == 0 else None # 当前正在等待训练的样本列表
        current_invalid_samples: deque = deque() if dist.get_rank() == 0 else None # 当前正在等待训练的无效样本列表
        time_costs = dict()
        train_metrics = {"loss": [], "policy_loss": [], "kl_loss": [], "entropy_loss": [], "distill_loss": []}
        per_type_rollout_metrics: Dict[str, Dict[str, List[float]]] = dict()
        per_type_avg_metrics: Dict[str, List[str]] = dict()
        batch_num = 0
        start_time = time.time()

        def should_go_on():
            go_on_flag = False
            if dist.get_rank() == 0:
                go_on_flag = num_received_batch_samples < self.args.rollout_batch_size or len(current_samples) > 0
            go_on_flag = torch.tensor([int(go_on_flag)], dtype=torch.int32, device=dist_utils.get_device())
            dist.broadcast(go_on_flag, src=0)
            go_on_flag = bool(go_on_flag.item())
            return go_on_flag
        
        def update_rollout_metrics(batch_sample: MultiResponseSample):
            if batch_sample.dataset_type not in per_type_rollout_metrics:
                per_type_rollout_metrics[batch_sample.dataset_type] = dict()
            rollout_metrcis = per_type_rollout_metrics[batch_sample.dataset_type]
            if batch_sample.dataset_type not in per_type_avg_metrics:
                per_type_avg_metrics[batch_sample.dataset_type] = set()
            avg_metrics = per_type_avg_metrics[batch_sample.dataset_type]
            if "reward" not in rollout_metrcis:
                rollout_metrcis["reward"] = []
            rollout_metrcis["reward"].extend(batch_sample.rewards)
            if "completion_tokens" not in rollout_metrcis:
                rollout_metrcis["completion_tokens"] = []
            rollout_metrcis["completion_tokens"].append(batch_sample.completion_tokens)
            if "total_tokens" not in rollout_metrcis:
                rollout_metrcis["total_tokens"] = []
            if "invalid_samples" not in rollout_metrcis:
                rollout_metrcis["invalid_samples"] = []
            rollout_metrcis["invalid_samples"].extend([1 if r == 0.0 else 0 for r in batch_sample.normed_rewards])
            rollout_metrcis["total_tokens"].append(batch_sample.total_tokens)
            for reward_metric in batch_sample.reward_metrics_list:
                for key, value in reward_metric.items():
                    if key not in rollout_metrcis:
                        rollout_metrcis[key] = []
                    rollout_metrcis[key].append(value)
            for v in batch_sample.avg_reward_metrics:
                avg_metrics.add(v)
        
        while should_go_on():
            batch_experience = None
            batch_num += 1
            if dist.get_rank() == 0:
                batch_samples = []

                while True:
                    try:
                        self.logger.info(f"waiting for rollout samples, current received samples: {num_received_batch_samples}")
                        batch_sample: MultiResponseSample = rollout_output_queue.get(block=False) # TODO: handle rollout metrics
                        update_rollout_metrics(batch_sample)
                        num_received_batch_samples += 1
                        self.logger.debug(f"{num_received_batch_samples} / {self.args.rollout_batch_size} samples received from rollout")
                        samples = self.unfold_samples([batch_sample])
                        self.dump_rollout(rollout_step, samples, dump_path=self.args.dump_path) # TODO: make it async
                        for sample in samples:
                            if sample.normed_reward == 0.0 and self.args.skip_zero_reward_sample:
                                # 需要避免一批rollout出来全是invalid
                                current_invalid_samples.append(sample)
                            else:
                                current_samples.append(sample)
                    except QueueEmpty:
                        if len(current_samples) >= self.minimum_train_batch_size:
                            self.logger.debug("break case 1")
                            batch_size = (len(current_samples) // self.minimum_train_batch_size) * self.minimum_train_batch_size
                            if self.max_train_batch_size > 0:
                                batch_size = min(batch_size, self.max_train_batch_size)
                            # batch_size = 2
                            for _ in range(batch_size):
                                batch_samples.append(current_samples.popleft())
                            break
                        elif num_received_batch_samples == self.args.rollout_batch_size and len(current_samples) > 0:
                            # 已经全推完了，但剩余样本不够一个train batch，需要和无效样本拼车
                            self.logger.debug("break case 2")
                            assert len(current_samples) + len(current_invalid_samples) >= self.minimum_train_batch_size, "something must be wrong"
                            for _ in range(len(current_samples)):
                                batch_samples.append(current_samples.popleft())
                            self.logger.warning(f"adding {self.minimum_train_batch_size - len(batch_samples)} invalid samples for padding")
                            for _ in range(self.minimum_train_batch_size - len(batch_samples)):
                                batch_samples.append(current_invalid_samples.popleft())
                            break
                        elif num_received_batch_samples == self.args.rollout_batch_size and len(current_samples) == 0:
                            # 已经全推完了，且没有剩余样本，直接退出
                            # TODO: 需求要考虑一个batch都是invalid的情况
                            assert len(batch_samples) == 0, "this is wrong"
                            self.logger.debug("break case 3")
                            break
                        else:
                            # 继续等待
                            time.sleep(0.5)

                # TODO: should exit the while loop
                # assert len(batch_samples) > 0, f"current_samples: {len(current_samples)}, current_invalid_samples: {len(current_invalid_samples)}"
                if len(batch_samples) > 0:
                    self.logger.info(f"iter {rollout_step} batch {batch_num} created, samples: {len(batch_samples)}, {num_received_batch_samples} / {self.args.rollout_batch_size} samples received from rollout")
                    batch_experience = self.collate_fn_optimized(batch_samples)
                else:
                    # self.logger.warning("no valid training samples, skipping this batch")
                    batch_experience = None
            dist.barrier(group=self.long_wait_group)
            if dist.get_rank() == 0:
                do_train_flag = torch.tensor([int(batch_experience is not None)], dtype=torch.int32, device=dist_utils.get_device())
            else:
                do_train_flag = torch.tensor([0], dtype=torch.int32, device=dist_utils.get_device())
            dist.broadcast(do_train_flag, src=0)
            do_train_flag = bool(do_train_flag.item())
            if not do_train_flag:
                self.logger.warning("no valid training samples, skipping this batch")
                continue
            
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                batch_experience = self.dispatch_batch_experience_before_making_experience(batch_experience)
            
            
            n_micro_batches = None
            if dist.get_rank() == 0:
                n_micro_batches = batch_experience.input_ids.shape[0] // (self.args.micro_batch_size * mpu.get_data_parallel_world_size())
                n_micro_batches_tensor = torch.tensor([n_micro_batches], dtype=torch.int64, device=dist_utils.get_device())
            else:
                n_micro_batches_tensor = torch.zeros((1,), dtype=torch.int64, device=dist_utils.get_device())
            dist.broadcast(n_micro_batches_tensor, src=0)
            n_micro_batches = n_micro_batches_tensor.item()

            if dist.get_rank() == PROFILE_RANK and self.do_profile and n_micro_batches >= 8:
                self.logger.warning("nsys profiling start")
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push("Stage_Prepare")

            if self.args.distillation_enabled:
                self.logger.info("computing teacher logits")
                # st = time.time()
                self.compute_teacher_logits([batch_experience])
                # 蒸馏：在创建 DataLoader 之前附加 distill_segments，确保 Dataset 切分与 teacher_logits 对齐
                self._attach_distill_segments([batch_experience])
                # et = time.time()
                # if "compute_teacher_logits" not in time_costs:
                #     time_costs["compute_teacher_logits"] = [et - st]
                # else:
                #     time_costs["compute_teacher_logits"].append(et - st)
            
            if self.args.init_kl_coef > 1e-8:
                self.logger.info("sending inputs to ref model")
                st = time.time()
                if dist.get_rank() == 0:
                    self.rank_0_send_input_to_ref([batch_experience])
                dist.barrier()
                et = time.time()
                if "send_to_ref" not in time_costs:
                    time_costs["send_to_ref"] = [et - st]
                else:
                    time_costs["send_to_ref"].append(et - st)
            
            if self.args.init_kl_coef > 1e-8:
                self.logger.info("receiving and syncing ref log probs")
                st = time.time()
                self.receive_and_sync_output_from_ref([batch_experience])
                et = time.time()
                if "receive_from_ref" not in time_costs:
                    time_costs["receive_from_ref"] = [et - st]
                else:
                    time_costs["receive_from_ref"].append(et - st)

            data_iter = None
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                dataloader = create_distributed_dataloader([batch_experience], self.args.micro_batch_size)
                data_iter = iter(dataloader)
                
            
            if dist.get_rank() == PROFILE_RANK and self.do_profile and n_micro_batches >= 8:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("Stage_Train")
            if self.args.distillation_enabled:
                self.distillation_step(
                    data_iter,
                    num_microbatches=n_micro_batches,
                    train_metrics=train_metrics,
                    iter_num=rollout_step,
                    batch_num=batch_num,
                )
            else:
                self.train_step(data_iter, num_microbatches=n_micro_batches, train_metrics=train_metrics, iter_num=rollout_step, batch_num=batch_num)
            
            if dist.get_rank() == PROFILE_RANK and self.do_profile and n_micro_batches >= 8:
                self.logger.warning("nsys profiling stop")
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
                self.do_profile = False

                # if self.args.dump_experience and dist.get_rank() == 0:
                #     self.dump_experience(passed_iters + 1, batch_experience, dump_path=self.args.dump_path, dump_tensors=self.args.dump_tensors)

        self.optim_and_update(train_metrics)
        self.update_params_to_sglang()
        iter_time = time.time() - start_time

        if dist.get_rank() == 0:
            self.log_rollout_metrics_and_throughput(per_type_rollout_metrics, per_type_avg_metrics, iter_time, rollout_step)
        self.log_train_metrics(train_metrics, rollout_step)

    def train(self):
        """Training main function.
        """
        assert self.init_success, "Trainer must be initialized before training."
        args = self.args
        logger = self.logger
        passed_iters = self.passed_iters

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
        pbar = tqdm(desc="Training", initial=passed_iters, total=args.train_iters)

        rollout_output_queue: MpQueue = self.output_queue
        while self.passed_iters_this_run < args.train_iters:
            self.passed_iters_this_run = self.passed_iters_this_run + 1
            if self.input_queue is not None:
                self.input_queue.put(self.passed_iters_this_run)

            self.logger.info("train steps")
            self.async_train_batch(self.passed_iters_this_run, rollout_output_queue)
            dist.barrier()
            # if passed_iters % args.save_interval == 0:
            #     self.save_logs_and_checkpoints(passed_iters)
            # dist.barrier()
            self._maybe_save_checkpoint()
            dist.barrier()

            pbar.update(1)

            if self.passed_iters_this_run == self.args.early_stop_steps:
                self.logger.warning(f"reached early stop steps: {self.passed_iters_this_run}")
                break
            
            stop_signal = self.stop_event.is_set() if self.stop_event is not None else False
            stop_signal = torch.tensor([int(stop_signal)], dtype=torch.int32, device=dist_utils.get_device())
            dist.broadcast(stop_signal, src=0)
            if bool(stop_signal.item()):
                self.logger.warning("received stop signal from rollout process, stopping training")
                break
        
        pbar.close()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

        self.stop_event.set()

        if dist.get_rank() == 0:
            if hasattr(self.input_queue, "close"):
                self.input_queue.close()
            if hasattr(self.output_queue, "close"):
                self.output_queue.close()
            self.saver_thread_pool.shutdown(wait=True)
            self.rollout_process.join()

    def all_reduce_in_pp_group(self):
        # very import and necesarry
        tensor = torch.zeros((1024,), dtype=torch.bfloat16, device=dist_utils.get_device())
        dist.all_reduce(tensor, group=mpu.get_pipeline_model_parallel_group())

    def dump_rollout_fn(self, iter: int, samples: List[Sample], dump_path: str):
        with open(f"{dump_path}/iter_{iter}_samples.jsonl", mode='a', encoding='utf-8') as f:
            for sample in samples:
                raw_line = json.dumps(self._sample_to_dump_dict(sample), ensure_ascii=False)
                f.write(raw_line)
                f.write('\n')
            f.flush()
    
    def dump_rollout(self, iter: int, samples: List[Sample], dump_path: str):
        if self.prev_save_future is not None:
            self.prev_save_future.result()
        self.prev_save_future = self.saver_thread_pool.submit(self.dump_rollout_fn, iter, samples, dump_path)
    
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
                raw_line = json.dumps(self._sample_to_dump_dict(sample), ensure_ascii=False)
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
        teacher_logps = None
        outcome_rewards = None
        need_teacher_logps = False

        if dist.get_rank() == 0:
            # keep the gpu tensor
            input_ids = batch_experience.input_ids
            labels = batch_experience.labels
            loss_mask = batch_experience.loss_mask
            teacher_logps = batch_experience.teacher_logps
            outcome_rewards = batch_experience.outcome_rewards
            need_teacher_logps = teacher_logps is not None
        
        if batch_experience is None:
            batch_experience = BatchExperience()

        batch_experience.input_ids = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
        batch_experience.labels = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
        batch_experience.loss_mask = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        need_teacher_logps_tensor = dist_utils._sync_2D_input_data(
            torch.tensor([[int(need_teacher_logps)]], dtype=torch.int32, device=dist_utils.get_device()) if dist.get_rank() == 0 else None,
            torch.int32,
            shape_tensor=torch.LongTensor([1, 1]),
        )
        need_teacher_logps = bool(need_teacher_logps_tensor[0, 0].item())
        if need_teacher_logps:
            batch_experience.teacher_logps = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        else:
            batch_experience.teacher_logps = None
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
        teacher_logps = dist_utils._sync_2D_input_data(teacher_logps, torch.float32, shape_tensor)
        outcome_rewards = dist_utils._sync_2D_input_data(
            outcome_rewards.unsqueeze(dim=0) if outcome_rewards is not None else None, 
            torch.float32, torch.LongTensor([1, shape_tensor[0]])
        ).squeeze(dim=0)

        with torch.cuda.stream(self.s1):
            batch_experience.input_ids.copy_(input_ids, non_blocking=True)
            batch_experience.labels.copy_(labels, non_blocking=True)
        with torch.cuda.stream(self.s2):
            batch_experience.loss_mask.copy_(loss_mask, non_blocking=True)
            if batch_experience.teacher_logps is not None:
                batch_experience.teacher_logps.copy_(teacher_logps, non_blocking=True)
            batch_experience.outcome_rewards.copy_(outcome_rewards, non_blocking=True)
        self.s1.synchronize()
        self.s2.synchronize()

        return batch_experience

    def unfold_samples(self, samples: List[MultiResponseSample]) -> List[Sample]:
        """Unfold and shuffle samples.
        """
        all_samples = []
        for sample in samples:
            prompt = sample.prompt
            responses = sample.responses or []
            rewards = sample.rewards or []
            normed_rewards = sample.normed_rewards or []
            group_messages = sample.group_messages or [
                self._build_legacy_messages(prompt, response)
                for response in responses
            ]
            group_content_ids = sample.group_content_ids or []
            group_loss_mask = sample.group_loss_mask or []
            group_teacher_log_probs = sample.group_teacher_log_probs or []
            completion_tokens = sample.completion_tokens
            reward_metrics_list = sample.reward_metrics_list or []
            sample_id = sample.sample_id

            use_teacher_logps = any(
                item is not None
                for item in (sample.group_content_ids, sample.group_loss_mask, sample.group_teacher_log_probs)
            )
            if use_teacher_logps:
                expected = len(responses)
                if not (
                    len(group_content_ids) == expected
                    and len(group_loss_mask) == expected
                    and len(group_teacher_log_probs) == expected
                ):
                    raise ValueError(
                        f"teacher distillation fields length mismatch for sample_id={sample_id}: "
                        f"responses={expected}, content_ids={len(group_content_ids)}, "
                        f"loss_mask={len(group_loss_mask)}, teacher_log_probs={len(group_teacher_log_probs)}"
                    )
            all_samples += [
                Sample(
                    prompt=prompt,
                    response=response,
                    messages=messages,
                    content_ids=(group_content_ids[idx] if use_teacher_logps else None),
                    loss_mask=(group_loss_mask[idx] if use_teacher_logps else None),
                    teacher_log_probs=(group_teacher_log_probs[idx] if use_teacher_logps else None),
                    reward=reward,
                    normed_reward=normed_reward,
                    reward_metrics=reward_metrics,
                    completion_tokens=completion_tokens / len(responses),
                    ground_truth=sample.ground_truth,
                    dataset_type=sample.dataset_type,
                    sample_id=sample_id,
                )
                for idx, (response, messages, reward, normed_reward, reward_metrics) in enumerate(
                    zip(responses, group_messages, rewards, normed_rewards, reward_metrics_list)
                )
            ]
        random.shuffle(all_samples)
        return all_samples
    
    def collate_fn_optimized(self, raw_samples: List[Sample]) -> BatchExperience:
        # if self.args.skip_zero_reward_sample:
        #     raise NotImplementedError
        #     # samples: List[Sample] = self.filter_out_zero_reward_sample(raw_samples)
        # else:
        #     samples = raw_samples
        samples = raw_samples
        self.logger.debug(f"drop {len(raw_samples) - len(samples)} samples")
        assert len(samples) > 0
        outcome_rewards = torch.tensor([s.normed_reward for s in samples], dtype=torch.float32, device=dist_utils.get_device())
        use_teacher_logps = [self._sample_uses_teacher_logps(sample) for sample in samples]
        if any(use_teacher_logps) and not all(use_teacher_logps):
            raise ValueError("collate_fn_optimized 当前不支持 teacher OPD 样本与普通 PPO 样本混跑")
        use_opd_path = all(use_teacher_logps)

        batch_input_ids_list = []
        batch_loss_mask_list = []
        batch_teacher_logps_list = [] if use_opd_path else None

        for sample in samples:
            if use_opd_path:
                if sample.content_ids is None or sample.loss_mask is None or sample.teacher_log_probs is None:
                    raise ValueError("OPD 样本缺少 content_ids/loss_mask/teacher_log_probs")
                input_ids = np.asarray(sample.content_ids)
                loss_mask = np.asarray(sample.loss_mask)
                teacher_logps = np.asarray(sample.teacher_log_probs)
                if not (len(input_ids) == len(loss_mask) == len(teacher_logps)):
                    raise ValueError("OPD 样本的 content_ids/loss_mask/teacher_log_probs 长度不一致")
                batch_input_ids_list.append(torch.tensor(input_ids, dtype=torch.int64, pin_memory=True))
                batch_loss_mask_list.append(torch.tensor(loss_mask, dtype=torch.float32, pin_memory=True))
                batch_teacher_logps_list.append(torch.tensor(teacher_logps, dtype=torch.float32, pin_memory=True))
            else:
                if sample.messages is None:
                    raise ValueError("Sample.messages must be populated before collation")
                input_ids, loss_mask = self._messages_to_token_ids_and_loss_mask(sample.messages)
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
        processed_teacher_logps = [] if use_opd_path else None

        for idx, (ids, mask) in enumerate(zip(batch_input_ids_list, batch_loss_mask_list)):
            ids = ids[:max_len_for_padding]
            mask = mask[:max_len_for_padding]
            processed_input_ids.append(ids)
            processed_loss_masks.append(mask)
            if use_opd_path:
                processed_teacher_logps.append(batch_teacher_logps_list[idx][:max_len_for_padding])
        
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
        if use_opd_path:
            padded_teacher_logps_tensor = torch.nn.utils.rnn.pad_sequence(
                processed_teacher_logps,
                batch_first=True,
                padding_value=0.0
            ).to(dist_utils.get_device(), non_blocking=True)
        else:
            padded_teacher_logps_tensor = None

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
            if use_opd_path:
                pad_tensor_teacher = torch.full(
                    (padded_teacher_logps_tensor.size(0), padding_needed),
                    0.0,
                    dtype=torch.float32,
                    device=dist_utils.get_device(),
                )
                padded_teacher_logps_tensor = torch.cat([padded_teacher_logps_tensor, pad_tensor_teacher], dim=1)
        
        final_input_ids = padded_input_ids_tensor[:, :-1].contiguous()
        final_labels = padded_input_ids_tensor[:, 1:].contiguous()
        final_loss_mask = padded_loss_mask_tensor[:, 1:].contiguous()
        final_teacher_logps = None if padded_teacher_logps_tensor is None else padded_teacher_logps_tensor[:, 1:].contiguous()

        return BatchExperience(
            input_ids=final_input_ids,
            labels=final_labels,
            loss_mask=final_loss_mask,
            teacher_logps=final_teacher_logps,
            outcome_rewards=outcome_rewards,
            batch_samples=samples
        )

    def compute_actor_ref_logps(self, batch_experience: BatchExperience) -> BatchExperience:
        if self.args.init_kl_coef > 1e-8:
            if dist.get_rank() == 0:
                self.rank_0_send_input_to_ref(batch_experience)
            dist.barrier()
        # actor forward
        batch_experience = self.actor_forward_only(batch_experience)

        if self.args.init_kl_coef > 1e-8:
            if dist.get_rank() == 0:
                batch_experience = self.rank_0_receive_output_from_ref(batch_experience)
            dist.barrier()
            if dist.get_rank() == 0:
                batch_experience.ref_logps = dist_utils._sync_2D_input_data(batch_experience.ref_logps, dtype=torch.float32).cpu()
            elif mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                batch_experience.ref_logps = dist_utils._sync_2D_input_data(batch_experience.ref_logps, dtype=torch.float32).cpu()
            dist.barrier()
            self.logger.debug("ref logp synced")

    def prepare_logits_cache(self, num_batches: int):
        if getattr(self.args, "gkd_sparse_topk_enabled", False):
            if self.topk_indices_cache is not None and self.topk_values_cache is not None:
                return
        else:
            if self.logits_cache is not None:
                return
        if getattr(self.args, "gkd_sparse_topk_enabled", False):
            topKp = int(self.args.gkd_topk) + 1
            max_possible_numel = (self.args.global_batch_size // mpu.get_data_parallel_world_size()) * self.args.seq_length * topKp
            self.logger.info(f"prepare_topk_cache, max_possible_numel: {max_possible_numel}")
            self.topk_indices_cache = [
                torch.empty((max_possible_numel,), pin_memory=False, device='cpu', dtype=torch.long)
                for _ in range(num_batches)
            ]
            self.topk_values_cache = [
                torch.empty((max_possible_numel,), pin_memory=False, device='cpu', dtype=torch.float32)
                for _ in range(num_batches)
            ]
            self.logger.info("topk cache created")
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
            local_bs = batch_experience.input_ids.shape[0] // mpu.get_data_parallel_world_size()
            seq_len = batch_experience.input_ids.shape[1]
            if getattr(self.args, "gkd_sparse_topk_enabled", False):
                topKp = int(self.args.gkd_topk) + 1
                logits_shape = (local_bs, seq_len, topKp)
                cur_numel = local_bs * seq_len * topKp
            else:
                logits_shape = (local_bs, seq_len, self.args.padded_vocab_size // mpu.get_tensor_model_parallel_world_size())
                cur_numel = local_bs * seq_len * (self.args.padded_vocab_size // mpu.get_tensor_model_parallel_world_size())
        if mpu.is_pipeline_last_stage():
            if getattr(self.args, "gkd_sparse_topk_enabled", False):
                batch_experience.teacher_topk_indices = self.topk_indices_cache[batch_id][:cur_numel].view(*logits_shape)
                batch_experience.teacher_topk_values = self.topk_values_cache[batch_id][:cur_numel].view(*logits_shape)
            else:
                batch_experience.teacher_logits = self.logits_cache[batch_id][:cur_numel].view(*logits_shape)

        if mpu.is_pipeline_last_stage():
            for i in range(num_iters):
                offset = i * logits_transfer_batch_size
                if getattr(self.args, "gkd_sparse_topk_enabled", False):
                    received_indices, received_values = self.logits_express.teacher_send_student_receive_topk()
                    dst_idx = batch_experience.teacher_topk_indices[offset:offset + logits_transfer_batch_size]
                    dst_val = batch_experience.teacher_topk_values[offset:offset + logits_transfer_batch_size]
                    dst_idx.copy_(received_indices.to(dtype=torch.long), non_blocking=True)
                    dst_val.copy_(received_values.to(dtype=torch.float32), non_blocking=True)
                else:
                    # 接收logits
                    received_logits = self.logits_express.teacher_send_student_receive()
                    dst = batch_experience.teacher_logits[offset:offset + logits_transfer_batch_size]
                    dst.copy_(received_logits, non_blocking=True)
                if int(os.environ.get("LIGHT_SCALE_DUMP_FLAG", os.environ.get("MRL_DUMP_FLAG", 0))) == 1:
                    dump_path = os.environ.get("LIGHT_SCALE_DUMP_PATH", os.environ.get("MRL_DUMP_PATH", None))
                    if dump_path is None:
                        raise RuntimeError("LIGHT_SCALE_DUMP_PATH is None")
                    if getattr(self.args, "gkd_sparse_topk_enabled", False):
                        np.save(f"{dump_path}/student_recv_topk_indices_iter_{i}_tp_{mpu.get_tensor_model_parallel_rank()}_dp_{mpu.get_data_parallel_rank()}.npy", received_indices.detach().cpu().numpy())
                        np.save(f"{dump_path}/student_recv_topk_values_iter_{i}_tp_{mpu.get_tensor_model_parallel_rank()}_dp_{mpu.get_data_parallel_rank()}.npy", received_values.detach().cpu().numpy())
                    else:
                        np.save(f"{dump_path}/student_recv_logits_iter_{i}_tp_{mpu.get_tensor_model_parallel_rank()}_dp_{mpu.get_data_parallel_rank()}.npy", received_logits.detach().cpu().numpy())
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

    def make_experience(self, batch_experience: BatchExperience) -> BatchExperience:
        # critic forward
        if self.args.init_kl_coef > 1e-8:
            if dist.get_rank() == 0:
                self.rank_0_send_input_to_ref(batch_experience)
            dist.barrier()
        # actor forward
        batch_experience = self.actor_forward_only(batch_experience)

        if self.args.init_kl_coef > 1e-8:
            if dist.get_rank() == 0:
                batch_experience = self.rank_0_receive_output_from_ref(batch_experience)
            dist.barrier()
            if dist.get_rank() == 0:
                batch_experience.ref_logps = dist_utils._sync_2D_input_data(batch_experience.ref_logps, dtype=torch.float32).cpu()
            elif mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                batch_experience.ref_logps = dist_utils._sync_2D_input_data(batch_experience.ref_logps, dtype=torch.float32).cpu()
            dist.barrier()
            self.logger.debug("ref logp synced")
            
        # compute kl
        if (self.args.init_kl_coef > 1e-8) and (not self.args.use_kl_loss) and dist.get_rank() == 0:
            batch_experience.kls = compute_approx_kl(
                batch_experience.old_actor_logps.to(device=dist_utils.get_device()),
                batch_experience.ref_logps.to(device=dist_utils.get_device()),
                batch_experience.loss_mask.to(device=dist_utils.get_device()),
                kl_estimator=self.args.kl_estimator,
            ).detach().cpu()
        else:
            if dist.get_rank() == 0 or mpu.is_pipeline_last_stage():
                batch_experience.kls = torch.zeros_like(
                    batch_experience.input_ids, 
                    dtype=torch.float32, 
                    device="cpu"
                )

        # compute advantages
        self.logger.debug("computing advantages")
        batch_experience = self.compute_advantages(batch_experience)
        return batch_experience
        
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
                # with torch.cuda.stream(self.s1):
                #     batch_experience.ref_logps.copy_(ref_logps, non_blocking=True)
                # self.s1.synchronize()
                batch_experience.ref_logps.copy_(ref_logps, non_blocking=True)
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

    def zero_grad(self):
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optim.zero_grad()

    def train_step(self, data_iterator: Optional[Iterator], num_microbatches: int, train_metrics: Dict[str, list], iter_num: int, batch_num: int):
        self.logger.debug("train step")
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        # self.optim.zero_grad()

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
            if dist.get_rank() == PROFILE_RANK and self.do_profile and num_microbatches >= 8:
                torch.cuda.nvtx.range_push(f"Stage_LossFunc")
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
                # if kl_loss is not None:
                #     if self.args.negate_kl_loss:
                #         loss -= self.args.init_kl_coef * kl_loss
                #     else:
                #         loss += self.args.init_kl_coef * kl_loss
                # if entropy_loss is not None and entropy_loss < self.args.entropy_loss_threshold:
                #     loss -= self.args.entropy_loss_coef * entropy_loss
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

            if dist.get_rank() == PROFILE_RANK and self.do_profile and num_microbatches >= 8:
                torch.cuda.nvtx.range_pop()

            return loss, {"losses": losses_reduced}

        def opd_loss_func(
            output_tensor: torch.Tensor,
            labels: torch.Tensor,
            loss_mask: torch.Tensor,
            teacher_logps: torch.Tensor,
            non_loss_data: bool = False,
        ):
            assert non_loss_data == False
            student_logps = from_parallel_logits_to_logprobs(output_tensor, labels, higher_stability=True, ignore_last=False)
            with torch.no_grad():
                advantages = teacher_logps - student_logps.detach()

            logp_diff = student_logps - student_logps.detach()
            ratio = logp_diff.exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2)
            actor_loss = masked_mean(actor_loss, loss_mask, dim=-1)
            if mpu.get_context_parallel_world_size() > 1:
                dist.all_reduce(actor_loss, op=ReduceOp.AVG, group=mpu.get_context_parallel_group())
            policy_loss = actor_loss.mean()

            entropy_loss = None
            if self.args.entropy_loss_coef > -1:
                unmasked_entropy_loss = vocab_parallel_entropy(output_tensor)
                entropy_loss = masked_mean(unmasked_entropy_loss, loss_mask, dim=-1)
                if mpu.get_context_parallel_world_size() > 1:
                    dist.all_reduce(entropy_loss, op=ReduceOp.AVG, group=mpu.get_context_parallel_group())
                entropy_loss = entropy_loss.mean()

            loss = self.args.policy_loss_coef * policy_loss
            losses_reduced = torch.cat(
                [
                    loss.clone().detach().unsqueeze(dim=0),
                    policy_loss.clone().detach().unsqueeze(dim=0),
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
                "teacher_logps": None,
                "loss_mask": None,
                "outcome_rewards": None,
                "debug_i": None
            }
            selected_loss_func = loss_func
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                assert data_iterator is not None
                batched_item: Dict[str, torch.Tensor] = next(data_iterator)
                if mpu.is_pipeline_first_stage():
                    forward_args["input_ids"] = batched_item["input_ids"].to(device=dist_utils.get_device(), non_blocking=True)
                if mpu.is_pipeline_last_stage():
                    loss_func_args["labels"] = batched_item["labels"].to(device=dist_utils.get_device(), non_blocking=True)
                    loss_func_args["loss_mask"] = batched_item["loss_mask"].to(device=dist_utils.get_device(), non_blocking=True)
                    loss_func_args["debug_i"] = batched_item["idx"]
                    if batched_item.get("teacher_logps") is not None:
                        selected_loss_func = opd_loss_func
                        loss_func_args["teacher_logps"] = batched_item["teacher_logps"].to(device=dist_utils.get_device(), non_blocking=True)
                    else:
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

            return output_tensor, partial(selected_loss_func, **loss_func_args)

        reduced_losses = get_forward_backward_func()(
            forward_step_func=megatron_forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=False,
            seq_length=self.args.seq_length,
            micro_batch_size=self.args.micro_batch_size,
            global_num_microbatches=self.args.global_batch_size // (self.args.micro_batch_size * mpu.get_data_parallel_world_size()),
        )

        self.logger.debug("received reduced losses")

        # update_successful, grad_norm, num_zeros_in_grad = self.optim.step()
        # if update_successful:
        #     self.scheduler.step(increment=self.args.global_batch_size)
        # else:
        #     self.logger.warning("optim update unsuccessful")
        self.optim.accumulate_grad_step()

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
        # lr = None
        # for param_group in self.optim.param_groups:
        #     if param_group['is_decoupled_lr']:
        #         lr = param_group['lr']
        #     else:
        #         lr = param_group['lr']
        self.logger.info(f"iter: {iter_num}, batch: {batch_num}, loss: {loss_item}, policy_loss: {policy_loss_item}, kl_loss: {kl_loss_item:.2e}, entropy_loss: {entropy_loss_item}")
        train_metrics["loss"].append(loss_item)
        train_metrics["policy_loss"].append(policy_loss_item)
        train_metrics["kl_loss"].append(kl_loss_item)
        train_metrics["entropy_loss"].append(entropy_loss_item)

        
        # if dist.get_rank() == 0:
        #     writer = self.tensorboard_writer
        #     writer.add_scalar("train/loss", loss_item, iter_num)
        #     writer.add_scalar("train/policy_loss", policy_loss_item, iter_num)
        #     writer.add_scalar("train/kl_loss", kl_loss_item, iter_num)
        #     writer.add_scalar("train/grad_norm", grad_norm, iter_num)
        #     writer.add_scalar("train/learning_rate", lr, iter_num)
        #     writer.add_scalar("train/entropy_loss", entropy_loss_item, iter_num)
        #     writer.flush()
        dist.barrier()

    def optim_and_update(self, train_metrics):
        update_successful, grad_norm, num_zeros_in_grad = self.optim.step_with_accumulated_grads()
        # update_successful, grad_norm, num_zeros_in_grad = self.optim.step()
        if update_successful:
            self.scheduler.step(increment=self.args.global_batch_size)
        else:
            self.logger.warning("optim update unsuccessful")
        
        train_metrics["grad_norm"] = grad_norm
        lr = None
        for param_group in self.optim.param_groups:
            if param_group['is_decoupled_lr']:
                lr = param_group['lr']
            else:
                lr = param_group['lr']
        train_metrics["learning_rate"] = lr

    def log_train_metrics(self, train_metrics, iter_num):
        def _safe_avg(values: List[float], default: float = 0.0):
            return (sum(values) / len(values)) if len(values) > 0 else default

        loss_item = _safe_avg(train_metrics.get("loss", []), 0.0)
        policy_loss_item = _safe_avg(train_metrics.get("policy_loss", []), 0.0)
        kl_loss_item = _safe_avg(train_metrics.get("kl_loss", []), 0.0)
        entropy_loss_item = _safe_avg(train_metrics.get("entropy_loss", []), 0.0)
        distill_loss_item = _safe_avg(train_metrics.get("distill_loss", []), 0.0)
        grad_norm = train_metrics["grad_norm"]
        lr = train_metrics["learning_rate"]
        if self.args.distillation_enabled:
            self.logger.info(f"iter: {iter_num}, loss: {loss_item:.2e}, distill_loss: {distill_loss_item:.2e}, grad_norm: {grad_norm:.2e}, learning_rate: {lr:.2e}")
        else:
            self.logger.info(f"iter: {iter_num}, loss: {loss_item:.2e}, policy_loss: {policy_loss_item:.2e}, kl_loss: {kl_loss_item:.2e}, entropy_loss: {entropy_loss_item:.2e}, grad_norm: {grad_norm:.2e}, learning_rate: {lr:.2e}")

        writer = self.tensorboard_writer
        if dist.get_rank() == 0:
            writer = self.tensorboard_writer
            writer.add_scalar("train/loss", loss_item, iter_num)
            if self.args.distillation_enabled:
                writer.add_scalar("train/distill_loss", distill_loss_item, iter_num)
            else:
                writer.add_scalar("train/policy_loss", policy_loss_item, iter_num)
                writer.add_scalar("train/kl_loss", kl_loss_item, iter_num)
                writer.add_scalar("train/entropy_loss", entropy_loss_item, iter_num)
            writer.add_scalar("train/grad_norm", grad_norm, iter_num)
            writer.add_scalar("train/learning_rate", lr, iter_num)
            writer.flush()

    def log_rollout_metrics_and_throughput(self, per_type_rollout_metrics, per_type_avg_metrics, iter_time, iter_num):
        for data_type, rollout_metrics in per_type_rollout_metrics.items():
            metrics_for_log = dict()
            avg_metrics = per_type_avg_metrics[data_type]
            for name, values in rollout_metrics.items():
                numeric_values = [value for value in values if isinstance(value, Number)]
                if not numeric_values:
                    continue
                if name in self.AVG_METRICS or name in avg_metrics:
                    avg_value = sum(numeric_values) / len(numeric_values)
                else:
                    avg_value = sum(numeric_values)
                metrics_for_log[name] = avg_value

            throughput = (metrics_for_log["total_tokens"] * self.args.rollout_batch_size) / (iter_time * self.args.total_world_size)
            metrics_for_log['completion_tokens'] = int(metrics_for_log['completion_tokens'] / self.args.n_samples)
            metrics_for_log["total_tokens"] = int(metrics_for_log["total_tokens"] / self.args.n_samples)

            writer = self.tensorboard_writer
            writer.add_scalar(f"throughput", throughput, iter_num)
            for name, avg_value in metrics_for_log.items():
                writer.add_scalar(f"rollout/{data_type}/{name}", avg_value, iter_num)
            writer.flush()
            part_1 = f"iter: {iter_num}, data_type: {data_type}, reward: {metrics_for_log['reward']:.4f}, completion tokens: {metrics_for_log['completion_tokens']}, invalid_samples: {metrics_for_log['invalid_samples']}, throughput: {int(throughput)} tokens/s/p, "
            metrics_for_log.pop("reward")
            metrics_for_log.pop("completion_tokens")
            metrics_for_log.pop("invalid_samples")
            metrics_for_log.pop("total_tokens")
            part_2 = ", ".join([f"{k}: {v}" for k, v in metrics_for_log.items()])
            self.logger.info(part_1 + part_2)

    def distillation_step(self, data_iterator: Optional[Iterator], num_microbatches: int, train_metrics: Dict[str, list], iter_num: int = None, batch_num: int = None):
        self.logger.debug("distillation train step")
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        # self.optim.zero_grad()

        def loss_func(
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor = None,
            teacher_topk_indices: torch.Tensor = None,
            teacher_topk_values: torch.Tensor = None,
            loss_mask: torch.Tensor = None,
            beta=0.5, 
            student_temperature=1.0,
            teacher_temperature=1.0,
            reduction="batchmean",
            non_loss_data: bool = False,
            debug_i: Optional[torch.Tensor] = None
        ):
            assert non_loss_data == False

            assert loss_mask is not None
            if getattr(self.args, "gkd_sparse_topk_enabled", False):
                assert teacher_topk_indices is not None and teacher_topk_values is not None
            else:
                assert teacher_logits is not None

            if getattr(self.args, "gkd_sparse_topk_enabled", False):
                tp_rank = int(mpu.get_tensor_model_parallel_rank())
                tp_world_size = int(mpu.get_tensor_model_parallel_world_size())
                vocab_start, vocab_end = get_tp_vocab_range(
                    padded_vocab_size=int(self.args.padded_vocab_size),
                    tp_rank=tp_rank,
                    tp_world_size=tp_world_size,
                )
                sentinel_value = float(self.args.topk_sentinel_value)

                # teacher 侧已构造“全局 topK + label 槽”（固定形状 K+1），
                # 这里直接使用，不再在 student loss 里做 TP all_gather/topK。
                final_indices, final_values = teacher_topk_indices, teacher_topk_values

                # teacher_topk_values 已经是 teacher 侧预先计算并传输的 log_probs（含 teacher_temperature）。
                # 转为本 rank 的 local-view sparse 表示：非本地 shard 位置填 sentinel。
                in_shard = (final_indices >= int(vocab_start)) & (final_indices < int(vocab_end))
                teacher_log_probs = torch.where(
                    in_shard,
                    final_values,
                    torch.full_like(final_values, sentinel_value),
                )

                # student 直接在 topK+1 全局索引上计算 local-view sparse log_probs，
                # 避免构造 full [B, L, V_local] 的 log_probs。
                student_log_probs = distributed_sparse_log_softmax(
                    vocab_parallel_logits=student_logits / student_temperature,
                    indices_global=final_indices,
                    vocab_start=int(vocab_start),
                    vocab_end=int(vocab_end),
                    fill_value=sentinel_value,
                    group=mpu.get_tensor_model_parallel_group(),
                )
            else:
                teacher_logits = teacher_logits / teacher_temperature
                student_logits = student_logits / student_temperature

                teacher_log_probs = distributed_log_softmax(teacher_logits)
                student_log_probs = distributed_log_softmax(student_logits)

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
            global_jsd = tp_sum_forward_identity_backward(local_jsd, group=mpu.get_tensor_model_parallel_group())

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
                "teacher_topk_indices": None,
                "teacher_topk_values": None,
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
                    if getattr(self.args, "gkd_sparse_topk_enabled", False):
                        loss_func_args["teacher_topk_indices"] = batched_item["teacher_topk_indices"].to(device=dist_utils.get_device(), non_blocking=True)
                        loss_func_args["teacher_topk_values"] = batched_item["teacher_topk_values"].to(device=dist_utils.get_device(), non_blocking=True)
                    else:
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
            micro_batch_size=self.args.micro_batch_size,
            global_num_microbatches=self.args.global_batch_size // (self.args.micro_batch_size * mpu.get_data_parallel_world_size()),
        )

        self.logger.debug("received reduced losses")

        # async 训练中仅累计梯度，统一在 optim_and_update() 中完成真正参数更新
        self.optim.accumulate_grad_step()

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
        self.logger.info(f"iter: {iter_num}, batch: {batch_num}, distill_loss: {loss_item:.2e}")
        train_metrics["loss"].append(loss_item)
        train_metrics["distill_loss"].append(loss_item)
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

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _maybe_save_checkpoint(self) -> None:
        args = self.args
        iteration = self.passed_iters_this_run
        if args.save is not None and args.save_interval and iteration % args.save_interval == 0:
            self._save_checkpoint(iteration, non_persistent=False)
            return
        if (
            args.non_persistent_save_interval > 0
            and iteration % args.non_persistent_save_interval == 0
        ):
            self._save_checkpoint(iteration, non_persistent=True)
            return
        if self.args.save_test_step > 0 and self.passed_iters_this_run - self.passed_iters == self.args.save_test_step:
            self._save_checkpoint(iteration, non_persistent=True)
            return

    def _save_checkpoint(self, iteration: int, non_persistent: bool) -> None:
        self.logger.info(
            f"Saving {'non-persistent' if non_persistent else 'persistent'} checkpoint at iteration {iteration} ..."
        )
        if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
            disable_forward_pre_hook(self.model)
        save_megatron_checkpoint(iteration, self.model, self.optim, self.scheduler, num_floating_point_operations_so_far=0, non_persistent_ckpt=non_persistent)
        if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
            enable_forward_pre_hook(self.model)

    def debug_hook(self, grad, name):
        if grad is not None:
            print(f"{name} grad is zero: {torch.allclose(grad, torch.zeros_like(grad), atol=1e-8)}", flush=True)
            norm = grad.norm().item()
            # if norm > 0:  # 您可以根据需要取消注释这个过滤条件
            print(f"[DEBUG HOOK] {name} received grad. Norm: {norm:.6e}", flush=True)