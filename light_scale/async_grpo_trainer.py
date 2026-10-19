from typing import List, Optional, Tuple, Union, Iterator, Dict
import os
from numbers import Number
import heapq

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
    get_model_config,
    unwrap_model,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

from light_scale.dataset import PlannedBatchIterator, build_microbatch_plan, create_distributed_dataloader
from light_scale.data import BatchExperience, Message, MultiResponseSample, Sample
from light_scale.logits_express import LogitsExpress
from light_scale.gkd_utils import distributed_log_softmax, safe_all_reduce, tp_sum_forward_identity_backward, _TPSumForwardIdentityBackward

from light_scale.gkd_utils import (
    distributed_log_softmax,
    distributed_sparse_log_softmax,
    safe_all_reduce,
    tp_sum_forward_identity_backward,
    _TPSumForwardIdentityBackward,
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
    def _to_pinned_cpu_tensor(data, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(data, dtype=dtype)
        return tensor.pin_memory()

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

    def _group_messages_to_content_ids_and_loss_mask(self, batch_sample: MultiResponseSample) -> None:
        if batch_sample.group_content_ids is not None and batch_sample.group_loss_mask is not None:
            return

        responses = batch_sample.responses or []
        group_messages = batch_sample.group_messages or [
            self._build_legacy_messages(batch_sample.prompt, response)
            for response in responses
        ]
        batch_sample.group_messages = group_messages

        flat_contents: List[str] = []
        flat_is_masked: List[bool] = []
        group_message_counts: List[int] = []
        for messages in group_messages:
            group_message_counts.append(len(messages))
            for message in messages:
                flat_contents.append(message.content or "")
                flat_is_masked.append(message.is_masked)

        if not flat_contents:
            batch_sample.group_content_ids = []
            batch_sample.group_loss_mask = []
            return

        if callable(self.tokenizer):
            encodings = self.tokenizer(
                flat_contents,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            encoded_messages = encodings["input_ids"] if isinstance(encodings, dict) else encodings.input_ids
        else:
            encoded_messages = [
                self.tokenizer.encode(content, add_special_tokens=False)
                for content in flat_contents
            ]

        group_content_ids: List[np.ndarray] = []
        group_loss_mask: List[np.ndarray] = []
        cursor = 0
        for message_count in group_message_counts:
            sample_input_ids: List[int] = []
            sample_loss_mask: List[float] = []
            for _ in range(message_count):
                message_ids = encoded_messages[cursor]
                is_masked = flat_is_masked[cursor]
                sample_input_ids.extend(message_ids)
                sample_loss_mask.extend([0.0 if is_masked else 1.0] * len(message_ids))
                cursor += 1
            group_content_ids.append(np.asarray(sample_input_ids, dtype=np.int64))
            group_loss_mask.append(np.asarray(sample_loss_mask, dtype=np.float32))

        batch_sample.group_content_ids = group_content_ids
        batch_sample.group_loss_mask = group_loss_mask

    @staticmethod
    def _sample_uses_teacher_logps(sample: Sample) -> bool:
        return sample.teacher_log_probs is not None

    def _external_expert_ids_spec(self) -> Optional[Tuple[int, int, int]]:
        config = get_model_config(self.model[0])
        if not config.enable_external_expert_ids:
            return None

        moe_layer_freq = config.moe_layer_freq
        if isinstance(moe_layer_freq, int):
            if moe_layer_freq <= 0:
                raise ValueError("moe_layer_freq must be positive for external expert IDs")
            num_moe_layers = sum(
                layer_idx % moe_layer_freq == 0 for layer_idx in range(config.num_layers)
            )
        elif isinstance(moe_layer_freq, list):
            if len(moe_layer_freq) != config.num_layers or any(value not in (0, 1) for value in moe_layer_freq):
                raise ValueError(
                    "external expert IDs require moe_layer_freq to be a binary pattern "
                    "with one entry per transformer layer"
                )
            num_moe_layers = sum(moe_layer_freq)
        else:
            raise ValueError(f"Unsupported moe_layer_freq for external expert IDs: {moe_layer_freq!r}")

        if config.num_moe_experts is None or config.num_moe_experts <= 0 or num_moe_layers <= 0:
            raise ValueError("external expert IDs require at least one MoE layer and expert")
        return num_moe_layers, config.moe_router_topk, config.num_moe_experts

    def _prepare_sample_expert_ids(
        self,
        sample: Sample,
        token_count: int,
        expected_spec: Optional[Tuple[int, int, int]],
    ) -> Optional[np.ndarray]:
        if expected_spec is None:
            return None
        if sample.expert_ids is None:
            raise ValueError(f"external expert IDs are missing for sample_id={sample.sample_id}")

        num_moe_layers, topk, num_experts = expected_spec
        expert_ids = np.asarray(sample.expert_ids)
        if expert_ids.dtype != np.int32:
            raise ValueError(
                f"external expert IDs for sample_id={sample.sample_id} must have dtype int32, "
                f"got {expert_ids.dtype}"
            )
        expected_shape = (num_moe_layers, token_count, topk)
        if expert_ids.shape != expected_shape:
            raise ValueError(
                f"external expert IDs for sample_id={sample.sample_id} must have shape "
                f"{expected_shape}, got {expert_ids.shape}"
            )
        if np.any((expert_ids < 0) | (expert_ids >= num_experts)):
            raise ValueError(
                f"external expert IDs for sample_id={sample.sample_id} must be in "
                f"[0, {num_experts}) before collation"
            )
        return expert_ids

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
            "expert_ids_shape": None if sample.expert_ids is None else list(sample.expert_ids.shape),
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
            assert tokenizer.pad_id is not None
            tokenizer = tokenizer.tokenizer
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

    def _get_max_response_length(self) -> Optional[int]:
        raw_max = getattr(self.args, "max_response_length", None)
        if raw_max is None:
            return None

        max_response_length = int(raw_max)
        if max_response_length <= 0:
            raise ValueError("max_response_length must be a positive integer or None")
        return max_response_length

    def _get_max_pad_token_ratio(self) -> Optional[float]:
        raw_ratio = getattr(self.args, "max_pad_token_ratio", None)
        if raw_ratio is None:
            return None

        max_pad_token_ratio = float(raw_ratio)
        if not 0.0 <= max_pad_token_ratio <= 1.0:
            raise ValueError("max_pad_token_ratio must be in the closed interval [0, 1]")
        return max_pad_token_ratio

    def _get_opd_clip_coef(self) -> Optional[float]:
        raw_coef = getattr(self.args, "opd_clip_coef", None)
        if raw_coef is None:
            return None

        opd_clip_coef = float(raw_coef)
        if not 0.0 < opd_clip_coef < 1.0:
            raise ValueError("opd_clip_coef must be in the open interval (0, 1)")
        return opd_clip_coef

    def _truncate_response_supervision(self, input_ids, loss_mask, teacher_logps=None):
        max_response_length = self._get_max_response_length()
        if max_response_length is None:
            return input_ids, loss_mask, teacher_logps

        response_token_indices = np.flatnonzero(np.asarray(loss_mask))
        if response_token_indices.size <= max_response_length:
            return input_ids, loss_mask, teacher_logps

        truncate_end = int(response_token_indices[max_response_length - 1]) + 1
        truncated_input_ids = input_ids[:truncate_end]
        truncated_loss_mask = loss_mask[:truncate_end]
        truncated_teacher_logps = None if teacher_logps is None else teacher_logps[:truncate_end]
        return truncated_input_ids, truncated_loss_mask, truncated_teacher_logps

    @staticmethod
    def _estimate_pad_token_ratio(lengths: List[int]) -> float:
        if not lengths:
            return 0.0
        max_length = max(lengths)
        if max_length <= 0:
            return 0.0
        return (len(lengths) * max_length - sum(lengths)) / (len(lengths) * max_length)

    def _pop_batch_samples_from_heap(self, sample_heap) -> List[Sample]:
        available_samples = len(sample_heap)
        if available_samples < self.minimum_train_batch_size:
            return []

        max_batch_size = available_samples
        if self.max_train_batch_size > 0:
            max_batch_size = min(max_batch_size, self.max_train_batch_size)
        max_batch_size = (max_batch_size // self.minimum_train_batch_size) * self.minimum_train_batch_size
        if max_batch_size < self.minimum_train_batch_size:
            return []

        popped_items = [heapq.heappop(sample_heap) for _ in range(max_batch_size)]
        best_batch_size = self.minimum_train_batch_size
        max_pad_token_ratio = self._get_max_pad_token_ratio()

        if max_pad_token_ratio is None:
            best_batch_size = max_batch_size
        else:
            prefix_lengths: List[int] = []
            for index, (sample_length, _, _) in enumerate(popped_items, start=1):
                prefix_lengths.append(sample_length)
                if index % self.minimum_train_batch_size != 0:
                    continue
                pad_token_ratio = self._estimate_pad_token_ratio(prefix_lengths)
                if pad_token_ratio <= max_pad_token_ratio:
                    best_batch_size = index

        selected_samples = [item[2] for item in popped_items[:best_batch_size]]
        for item in popped_items[best_batch_size:]:
            heapq.heappush(sample_heap, item)
        return selected_samples
    
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
        actor_service_names = async_rollout_cfg["actor_service_names"]
        assert isinstance(actor_service_names, list) and len(actor_service_names) > 0, (
            "resolved async_rollout config must contain a non-empty actor_service_names list"
        )

        service_by_name = {service["name"]: service for service in async_rollout_cfg["services"]}
        resources = []
        for actor_service_name in actor_service_names:
            service = service_by_name.get(actor_service_name)
            assert service is not None, f"Cannot find actor service config for {actor_service_name}"
            service_resources = service.get("resources")
            assert service_resources is not None, f"Actor service {actor_service_name} is missing resources"
            resources.extend(service_resources)

        assert len(resources) > 0, f"Cannot find any actor resources for {actor_service_names}"
        base_url_list = [resource["base_url"] for resource in resources]
        world_size = sum(resource["num_gpus"] for resource in resources)
        return base_url_list, world_size
        

    def async_train_batch(self, rollout_step: int, rollout_output_queue: MpQueue):
        # 进行一次rollout batch的异步多次训练
        self.zero_grad()
        num_received_batch_samples: int = 0 # 从rollout队列中获取到的样本数
        current_samples = [] if dist.get_rank() == 0 else None # 当前正在等待训练的样本最小堆
        current_invalid_samples: deque = deque() if dist.get_rank() == 0 else None # 当前正在等待训练的无效样本列表
        time_costs = dict()
        train_metrics = {"loss": [], "policy_loss": [], "kl_loss": [], "entropy_loss": [], "distill_loss": []}
        per_type_rollout_metrics: Dict[str, Dict[str, List[float]]] = dict()
        per_type_avg_metrics: Dict[str, List[str]] = dict()
        batch_num = 0
        start_time = time.time()
        sample_arrival_index = 0

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
                        self._group_messages_to_content_ids_and_loss_mask(batch_sample)
                        samples = self.unfold_samples([batch_sample])
                        self.dump_rollout(rollout_step, samples, dump_path=self.args.dump_path) # TODO: make it async
                        for sample in samples:
                            if sample.content_ids is None or sample.loss_mask is None:
                                raise ValueError("Sample.content_ids and Sample.loss_mask must be populated before batching")
                            if sample.normed_reward == 0.0 and self.args.skip_zero_reward_sample:
                                # 需要避免一批rollout出来全是invalid
                                current_invalid_samples.append(sample)
                            else:
                                heapq.heappush(current_samples, (len(sample.content_ids), sample_arrival_index, sample))
                                sample_arrival_index += 1
                    except QueueEmpty:
                        if len(current_samples) >= self.minimum_train_batch_size:
                            self.logger.debug("break case 1")
                            batch_samples = self._pop_batch_samples_from_heap(current_samples)
                            break
                        elif num_received_batch_samples == self.args.rollout_batch_size and len(current_samples) > 0:
                            # 已经全推完了，但剩余样本不够一个train batch，需要和无效样本拼车
                            self.logger.debug("break case 2")
                            assert len(current_samples) + len(current_invalid_samples) >= self.minimum_train_batch_size, "something must be wrong"
                            for _ in range(len(current_samples)):
                                batch_samples.append(heapq.heappop(current_samples)[2])
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
                    if all(sample.teacher_log_probs is None for sample in batch_samples):
                        avg_normed_reward = sum(sample.normed_reward for sample in batch_samples) / len(batch_samples)
                        self.logger.info(
                            f"iter {rollout_step} batch {batch_num} created, samples: {len(batch_samples)}, "
                            f"{num_received_batch_samples} / {self.args.rollout_batch_size} samples received from rollout, "
                            f"avg_normed_reward: {avg_normed_reward:.6f}"
                        )
                    else:
                        self.logger.info(
                            f"iter {rollout_step} batch {batch_num} created, samples: {len(batch_samples)}, "
                            f"{num_received_batch_samples} / {self.args.rollout_batch_size} samples received from rollout"
                        )
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

            n_micro_batches = 0

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
            is_packed_path_local = False
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                if batch_experience.is_thd_packed:
                    if self.args.distillation_enabled:
                        raise RuntimeError(
                            "distillation_enabled is not supported with async THD packed OPD path; "
                            "please disable distillation or disable sequence_packing for OPD."
                        )
                    is_packed_path_local = True
                    data_iter = PlannedBatchIterator(batch_experience)
                    n_micro_batches = len(data_iter)
                else:
                    dataloader = create_distributed_dataloader([batch_experience], self.args.micro_batch_size)
                    data_iter = iter(dataloader)
                    n_micro_batches = len(dataloader)
            active_stage_tensor = torch.tensor(
                [
                    int(mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()),
                    int(is_packed_path_local),
                ],
                dtype=torch.int64,
                device=dist_utils.get_device(),
            )
            dist.all_reduce(active_stage_tensor, op=ReduceOp.SUM)
            active_stage_world_count = int(active_stage_tensor[0].item())
            packed_stage_world_count = int(active_stage_tensor[1].item())

            # Propagate scalar values to all ranks while also checking active-stage consistency.
            huge_int = torch.iinfo(torch.int64).max
            local_n_micro_batches = int(n_micro_batches)
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                n_micro_min = torch.tensor([local_n_micro_batches], dtype=torch.int64, device=dist_utils.get_device())
                n_micro_max = torch.tensor([local_n_micro_batches], dtype=torch.int64, device=dist_utils.get_device())
            else:
                n_micro_min = torch.tensor([huge_int], dtype=torch.int64, device=dist_utils.get_device())
                n_micro_max = torch.tensor([0], dtype=torch.int64, device=dist_utils.get_device())

            dist.all_reduce(n_micro_min, op=ReduceOp.MIN)
            dist.all_reduce(n_micro_max, op=ReduceOp.MAX)

            if active_stage_world_count > 0:
                if int(n_micro_min.item()) != int(n_micro_max.item()):
                    raise RuntimeError(
                        f"Inconsistent num_microbatches across active pipeline stages: "
                        f"min={int(n_micro_min.item())}, max={int(n_micro_max.item())}."
                    )

            # Planned packed path should run on every active first/last stage rank.
            if packed_stage_world_count > 0 and packed_stage_world_count != active_stage_world_count:
                raise RuntimeError(
                    "Packed async path was enabled on only a subset of active pipeline stages; "
                    "all active first/last stages must use the same path."
                )

            n_micro_batches = int(n_micro_max.item())
                
            
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
                self.train_step(
                    data_iter,
                    num_microbatches=n_micro_batches,
                    train_metrics=train_metrics,
                    iter_num=rollout_step,
                    batch_num=batch_num,
                    is_thd_packed=packed_stage_world_count > 0,
                )
            
            if dist.get_rank() == PROFILE_RANK and self.do_profile and n_micro_batches >= 8:
                self.logger.warning("nsys profiling stop")
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
                self.do_profile = False

                # if self.args.dump_experience and dist.get_rank() == 0:
                #     self.dump_experience(passed_iters + 1, batch_experience, dump_path=self.args.dump_path, dump_tensors=self.args.dump_tensors)

        self.optim_and_update(train_metrics)
        self.track_auxiliary_losses(rollout_step)
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
        expert_ids = None
        outcome_rewards = None
        need_teacher_logps = False
        is_thd_packed = False
        is_opd = False
        sample_seq_lens = None
        microbatch_offsets = None
        external_expert_ids_spec = self._external_expert_ids_spec()

        if dist.get_rank() == 0:
            # keep the gpu tensor
            input_ids = batch_experience.input_ids
            labels = batch_experience.labels
            loss_mask = batch_experience.loss_mask
            teacher_logps = batch_experience.teacher_logps
            expert_ids = batch_experience.expert_ids
            outcome_rewards = batch_experience.outcome_rewards
            need_teacher_logps = teacher_logps is not None
            is_thd_packed = batch_experience.is_thd_packed
            is_opd = batch_experience.is_opd
            sample_seq_lens = getattr(batch_experience, 'sample_seq_lens', None)
            microbatch_offsets = getattr(batch_experience, 'microbatch_offsets', None)
            assert is_opd == need_teacher_logps
            assert is_thd_packed == (sample_seq_lens is not None)
            assert is_thd_packed == (microbatch_offsets is not None)
            if external_expert_ids_spec is not None:
                num_moe_layers, topk, num_experts = external_expert_ids_spec
                expected_shape = (batch_size, num_moe_layers, seq_length, topk)
                if expert_ids is None or expert_ids.dtype != torch.int32 or tuple(expert_ids.shape) != expected_shape:
                    raise ValueError(
                        "external expert IDs must be an int32 tensor with shape "
                        f"{expected_shape}, got "
                        f"{None if expert_ids is None else (expert_ids.dtype, tuple(expert_ids.shape))}"
                    )
                if torch.any((expert_ids < -1) | (expert_ids >= num_experts)):
                    raise ValueError(
                        f"external expert IDs must contain either -1 or IDs in [0, {num_experts})"
                    )
        
        if batch_experience is None:
            batch_experience = BatchExperience()

        # This method is invoked only by the first and last pipeline stages.
        # Do not use the default world-group broadcast here: intermediate PP
        # stages do not enter this method and would deadlock the collective.
        thd_meta_tensor = dist_utils._sync_2D_input_data(
            torch.tensor(
                [
                    [int(is_thd_packed)],
                    [int(is_opd)],
                    [0 if microbatch_offsets is None else int(microbatch_offsets.shape[1] - 1)],
                ],
                dtype=torch.int64,
                device=dist_utils.get_device(),
            ) if dist.get_rank() == 0 else None,
            torch.int64,
            shape_tensor=torch.LongTensor([3, 1]),
        ).squeeze(dim=1)
        is_thd_packed = bool(thd_meta_tensor[0].item())
        is_opd = bool(thd_meta_tensor[1].item())
        planned_num_microbatches = int(thd_meta_tensor[2].item())

        batch_experience.input_ids = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
        batch_experience.labels = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.int64, device="cpu", pin_memory=True)
        batch_experience.loss_mask = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        if external_expert_ids_spec is not None:
            num_moe_layers, topk, _ = external_expert_ids_spec
            batch_experience.expert_ids = torch.zeros(
                (shape_tensor[0], num_moe_layers, shape_tensor[1], topk),
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
        else:
            batch_experience.expert_ids = None
        if is_opd:
            batch_experience.teacher_logps = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=torch.float32, device="cpu", pin_memory=True)
        else:
            batch_experience.teacher_logps = None
        batch_experience.outcome_rewards = torch.zeros((shape_tensor[0],), dtype=torch.float32, device="cpu", pin_memory=True)
        batch_experience.is_thd_packed = is_thd_packed
        batch_experience.is_opd = is_opd
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
        if external_expert_ids_spec is not None:
            expert_ids = dist_utils._sync_2D_input_data(
                expert_ids.reshape(shape_tensor[0], -1) if expert_ids is not None else None,
                torch.int32,
                torch.LongTensor(
                    [shape_tensor[0], external_expert_ids_spec[0] * shape_tensor[1] * external_expert_ids_spec[1]]
                ),
            ).reshape(
                shape_tensor[0],
                external_expert_ids_spec[0],
                shape_tensor[1],
                external_expert_ids_spec[1],
            )
        if is_opd:
            teacher_logps = dist_utils._sync_2D_input_data(teacher_logps, torch.float32, shape_tensor)
        else:
            teacher_logps = None
        outcome_rewards = dist_utils._sync_2D_input_data(
            outcome_rewards.unsqueeze(dim=0) if outcome_rewards is not None else None, 
            torch.float32, torch.LongTensor([1, shape_tensor[0]])
        ).squeeze(dim=0)
        if is_thd_packed:
            assert planned_num_microbatches > 0
            sample_seq_lens = dist_utils._sync_2D_input_data(
                sample_seq_lens.unsqueeze(dim=0) if sample_seq_lens is not None else None,
                torch.int64,
                torch.LongTensor([1, shape_tensor[0]]),
            ).squeeze(dim=0)
            microbatch_offsets = dist_utils._sync_2D_input_data(
                microbatch_offsets,
                torch.int64,
                torch.LongTensor([
                    mpu.get_data_parallel_world_size(),
                    planned_num_microbatches + 1,
                ]),
            )
            batch_experience.sample_seq_lens = torch.zeros(
                (shape_tensor[0],), dtype=torch.int64, device="cpu", pin_memory=True
            )
            batch_experience.microbatch_offsets = torch.zeros(
                microbatch_offsets.shape, dtype=torch.int64, device="cpu", pin_memory=True
            )
        else:
            sample_seq_lens = None
            microbatch_offsets = None
            batch_experience.sample_seq_lens = None
            batch_experience.microbatch_offsets = None

        with torch.cuda.stream(self.s1):
            batch_experience.input_ids.copy_(input_ids, non_blocking=True)
            batch_experience.labels.copy_(labels, non_blocking=True)
            if batch_experience.sample_seq_lens is not None:
                batch_experience.sample_seq_lens.copy_(sample_seq_lens, non_blocking=True)
            if batch_experience.microbatch_offsets is not None:
                batch_experience.microbatch_offsets.copy_(microbatch_offsets, non_blocking=True)
            if batch_experience.expert_ids is not None:
                batch_experience.expert_ids.copy_(expert_ids, non_blocking=True)
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
        external_expert_ids_enabled = self._external_expert_ids_spec() is not None
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
            group_expert_ids = sample.group_expert_ids or []
            completion_tokens = sample.completion_tokens
            reward_metrics_list = sample.reward_metrics_list or []
            sample_id = sample.sample_id

            has_content_ids = sample.group_content_ids is not None
            has_loss_mask = sample.group_loss_mask is not None
            use_teacher_logps = sample.group_teacher_log_probs is not None
            has_expert_ids = sample.group_expert_ids is not None
            if has_content_ids != has_loss_mask:
                raise ValueError(
                    f"content_ids/loss_mask presence mismatch for sample_id={sample_id}: "
                    f"content_ids={has_content_ids}, loss_mask={has_loss_mask}"
                )
            if has_content_ids:
                expected = len(responses)
                if not (
                    len(group_content_ids) == expected
                    and len(group_loss_mask) == expected
                ):
                    raise ValueError(
                        f"content_ids/loss_mask length mismatch for sample_id={sample_id}: "
                        f"responses={expected}, content_ids={len(group_content_ids)}, "
                        f"loss_mask={len(group_loss_mask)}"
                    )
            if use_teacher_logps:
                expected = len(responses)
                if not (
                    len(group_teacher_log_probs) == expected
                ):
                    raise ValueError(
                        f"teacher distillation fields length mismatch for sample_id={sample_id}: "
                        f"responses={expected}, teacher_log_probs={len(group_teacher_log_probs)}"
                    )
            if external_expert_ids_enabled:
                if not has_expert_ids or len(group_expert_ids) != len(responses):
                    raise ValueError(
                        f"external expert IDs length mismatch for sample_id={sample_id}: "
                        f"responses={len(responses)}, expert_ids={len(group_expert_ids)}"
                    )
            all_samples += [
                Sample(
                    prompt=prompt,
                    response=response,
                    messages=messages,
                    content_ids=(group_content_ids[idx] if has_content_ids else None),
                    loss_mask=(group_loss_mask[idx] if has_loss_mask else None),
                    teacher_log_probs=(group_teacher_log_probs[idx] if use_teacher_logps else None),
                    expert_ids=(group_expert_ids[idx] if has_expert_ids else None),
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

    def _collate_fn_thd_packing(self, samples: List[Sample], *, is_opd: bool) -> BatchExperience:
        assert self.args.sequence_packing, "THD path requires sequence_packing=True"
        assert self.args.micro_batch_size == 1, "THD path requires micro_batch_size == 1"

        # THD OPD 的 collate 阶段只做“样本对齐”的准备，不在 rank0 直接 pack。
        # 这里的目标是先把每个真实样本保留为独立的一行，并把长度补齐到 CP 可切分的粒度，
        # 后续再由 PlannedBatchIterator 在各个 rank 上执行：
        # 1. DP shard 切分；
        # 2. 对每个 sample 做 CP head-tail slicing；
        # 3. 在本 rank 上按切分后的 local length 做 sequence packing。
        # 这样才能同时满足“每个 DP rank 拿到相同真实样本数”和“当前 Megatron 版本的 THD+CP
        # 需要先对 sample 做 CP slicing，再 pack”这两个约束。
        cp_world_size = mpu.get_context_parallel_world_size()
        cp_multiple = 1 if cp_world_size == 1 else cp_world_size * 2
        pad_token_id = self.tokenizer.pad_token_id

        processed_input_ids = []
        processed_loss_masks = []
        processed_teacher_logps = [] if is_opd else None
        processed_expert_ids = [] if self._external_expert_ids_spec() is not None else None
        processed_outcome_rewards = []
        processed_samples = []
        sample_seq_lens = []

        for sample in samples:
            if sample.content_ids is None or sample.loss_mask is None:
                raise ValueError("THD 样本缺少 content_ids/loss_mask")

            input_ids = sample.content_ids
            loss_mask = sample.loss_mask
            teacher_logps = sample.teacher_log_probs if is_opd else None
            expert_ids = self._prepare_sample_expert_ids(
                sample, len(input_ids), self._external_expert_ids_spec()
            )
            if is_opd and teacher_logps is None:
                raise ValueError("THD OPD 样本缺少 teacher_log_probs")
            if is_opd and not (len(input_ids) == len(loss_mask) == len(teacher_logps)):
                raise ValueError("THD OPD 样本的 content_ids/loss_mask/teacher_log_probs 长度不一致")

            input_ids, loss_mask, teacher_logps = self._truncate_response_supervision(
                input_ids,
                loss_mask,
                teacher_logps,
            )
            if expert_ids is not None:
                expert_ids = expert_ids[:, :len(input_ids), :]

            # 这里记录的是 shift 之前的 sample 存储长度：
            # - 训练时最终会做 input_ids[:, :-1] / labels[:, 1:]，所以真实参与 loss 的长度是 sample_seq_len；
            # - 仅当 cp_size > 1 时，才需要为了后续 CP slicing 的“拿头拿尾”规则向 2 * cp_size 对齐；
            #   若 cp_size == 1，则这里不做额外对齐，避免引入不必要的 padding；
            # - 同时仍受 seq_length 上限约束。
            sample_seq_len = min(
                _ceil_to_nearest(max(len(input_ids) - 1, 1), cp_multiple),
                self.args.seq_length,
            )
            sample_storage_len = sample_seq_len + 1

            # 每个 sample 在 collate 阶段都单独 pad 到 sample_storage_len，
            # 这样 rank0 广播出去的是“样本边界稳定”的二维张量。
            # 后续 iterator 只需要再根据 sample_seq_lens 裁出真实有效区间即可，
            # 不需要重新推断每个样本的有效 token 边界。
            padded_input_ids = np.full((sample_storage_len,), pad_token_id, dtype=np.int64)
            padded_loss_mask = np.zeros((sample_storage_len,), dtype=np.float32)
            padded_teacher_logps = np.zeros((sample_storage_len,), dtype=np.float32) if is_opd else None
            padded_expert_ids = (
                np.full(
                    (expert_ids.shape[0], sample_storage_len, expert_ids.shape[2]),
                    -1,
                    dtype=np.int32,
                )
                if expert_ids is not None
                else None
            )

            valid_len = min(len(input_ids), sample_storage_len)
            padded_input_ids[:valid_len] = input_ids[:valid_len]
            padded_loss_mask[:valid_len] = loss_mask[:valid_len]
            if is_opd:
                padded_teacher_logps[:valid_len] = teacher_logps[:valid_len]
            if padded_expert_ids is not None:
                padded_expert_ids[:, :valid_len, :] = expert_ids[:, :valid_len, :]

            processed_input_ids.append(self._to_pinned_cpu_tensor(padded_input_ids, dtype=torch.int64))
            processed_loss_masks.append(self._to_pinned_cpu_tensor(padded_loss_mask, dtype=torch.float32))
            if is_opd:
                processed_teacher_logps.append(self._to_pinned_cpu_tensor(padded_teacher_logps, dtype=torch.float32))
            if padded_expert_ids is not None:
                processed_expert_ids.append(self._to_pinned_cpu_tensor(padded_expert_ids, dtype=torch.int32))
            processed_outcome_rewards.append(sample.normed_reward)
            processed_samples.append(sample)
            sample_seq_lens.append(sample_seq_len)

        # 在 rank 0 上统一规划 DP shard 和所有 THD microbatch 的样本归属。
        # 规划结果保证每个 DP shard 都有相同数量的非空真实 microbatch；较少的
        # shard 会拆分已有 real bucket，而不是补一个 whole-empty alignment bucket。
        microbatch_plan = build_microbatch_plan(
            sample_seq_lens,
            dp_world_size=mpu.get_data_parallel_world_size(),
            packed_seq_length=self.args.seq_length,
        )
        ordered_indices = microbatch_plan.ordered_indices

        ordered_input_ids = [processed_input_ids[idx] for idx in ordered_indices]
        ordered_loss_masks = [processed_loss_masks[idx] for idx in ordered_indices]
        ordered_teacher_logps = [processed_teacher_logps[idx] for idx in ordered_indices] if is_opd else None
        ordered_expert_ids = [processed_expert_ids[idx] for idx in ordered_indices] if processed_expert_ids is not None else None
        ordered_samples = [processed_samples[idx] for idx in ordered_indices]
        ordered_outcome_rewards = [processed_outcome_rewards[idx] for idx in ordered_indices]
        ordered_sample_seq_lens = [sample_seq_lens[idx] for idx in ordered_indices]

        padded_input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
            ordered_input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        ).to(dist_utils.get_device(), non_blocking=True)
        padded_loss_mask_tensor = torch.nn.utils.rnn.pad_sequence(
            ordered_loss_masks,
            batch_first=True,
            padding_value=0.0,
        ).to(dist_utils.get_device(), non_blocking=True)
        padded_teacher_logps_tensor = None
        if is_opd:
            padded_teacher_logps_tensor = torch.nn.utils.rnn.pad_sequence(
                ordered_teacher_logps,
                batch_first=True,
                padding_value=0.0,
            ).to(dist_utils.get_device(), non_blocking=True)
        padded_expert_ids_tensor = None
        if ordered_expert_ids is not None:
            padded_expert_ids_tensor = torch.nn.utils.rnn.pad_sequence(
                [tensor.permute(1, 0, 2) for tensor in ordered_expert_ids],
                batch_first=True,
                padding_value=-1,
            ).permute(0, 2, 1, 3).to(dist_utils.get_device(), non_blocking=True)

        # 这里仍然保持普通自回归训练的 shift 约定。
        # 注意：sample_seq_lens 表示的是 shift 之后每个样本的 token 数，
        # iterator 会用它来裁切出每个 sample 的真实 supervision 区间，再做 CP slice 和 packing。
        final_input_ids = padded_input_ids_tensor[:, :-1].contiguous()
        final_labels = padded_input_ids_tensor[:, 1:].contiguous()
        final_loss_mask = padded_loss_mask_tensor[:, 1:].contiguous()
        final_teacher_logps = None if padded_teacher_logps_tensor is None else padded_teacher_logps_tensor[:, 1:].contiguous()
        final_expert_ids = (
            None
            if padded_expert_ids_tensor is None
            else padded_expert_ids_tensor[:, :, :-1, :].contiguous()
        )

        return BatchExperience(
            input_ids=final_input_ids,
            labels=final_labels,
            loss_mask=final_loss_mask,
            teacher_logps=final_teacher_logps,
            expert_ids=final_expert_ids,
            sample_seq_lens=torch.tensor(
                ordered_sample_seq_lens, dtype=torch.int64, device=dist_utils.get_device()
            ),
            microbatch_offsets=torch.tensor(
                microbatch_plan.microbatch_offsets,
                dtype=torch.int64,
                device=dist_utils.get_device(),
            ),
            is_thd_packed=True,
            is_opd=is_opd,
            outcome_rewards=torch.tensor(
                ordered_outcome_rewards, dtype=torch.float32, device=dist_utils.get_device()
            ),
            batch_samples=ordered_samples,
        )

    def _collate_fn_opd_packing(self, samples: List[Sample]) -> BatchExperience:
        return self._collate_fn_thd_packing(samples, is_opd=True)

    def _collate_fn_grpo_packing(self, samples: List[Sample]) -> BatchExperience:
        assert self.args.use_outcome_rewards_as_advantages
        assert self.args.init_kl_coef <= 1e-8
        assert not self.args.distillation_enabled
        assert not self.is_old_actor_logp_required
        return self._collate_fn_thd_packing(samples, is_opd=False)
    
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

        if self.args.sequence_packing:
            return self._collate_fn_opd_packing(samples) if use_opd_path else self._collate_fn_grpo_packing(samples)

        batch_input_ids_list = []
        batch_loss_mask_list = []
        batch_teacher_logps_list = [] if use_opd_path else None
        batch_expert_ids_list = [] if self._external_expert_ids_spec() is not None else None

        for sample in samples:
            if use_opd_path:
                if sample.content_ids is None or sample.loss_mask is None or sample.teacher_log_probs is None:
                    raise ValueError("OPD 样本缺少 content_ids/loss_mask/teacher_log_probs")
                input_ids = sample.content_ids
                loss_mask = sample.loss_mask
                teacher_logps = sample.teacher_log_probs
                expert_ids = self._prepare_sample_expert_ids(
                    sample, len(input_ids), self._external_expert_ids_spec()
                )
                if not (len(input_ids) == len(loss_mask) == len(teacher_logps)):
                    raise ValueError("OPD 样本的 content_ids/loss_mask/teacher_log_probs 长度不一致")
                input_ids, loss_mask, teacher_logps = self._truncate_response_supervision(
                    input_ids,
                    loss_mask,
                    teacher_logps,
                )
                if expert_ids is not None:
                    expert_ids = expert_ids[:, :len(input_ids), :]
                batch_input_ids_list.append(self._to_pinned_cpu_tensor(input_ids, dtype=torch.int64))
                batch_loss_mask_list.append(self._to_pinned_cpu_tensor(loss_mask, dtype=torch.float32))
                batch_teacher_logps_list.append(self._to_pinned_cpu_tensor(teacher_logps, dtype=torch.float32))
                if expert_ids is not None:
                    batch_expert_ids_list.append(self._to_pinned_cpu_tensor(expert_ids, dtype=torch.int32))
            else:
                if sample.content_ids is None or sample.loss_mask is None:
                    raise ValueError("PPO 样本缺少 content_ids/loss_mask 预编码结果")
                input_ids = sample.content_ids
                loss_mask = sample.loss_mask
                expert_ids = self._prepare_sample_expert_ids(
                    sample, len(input_ids), self._external_expert_ids_spec()
                )
                input_ids, loss_mask, _ = self._truncate_response_supervision(input_ids, loss_mask)
                if expert_ids is not None:
                    expert_ids = expert_ids[:, :len(input_ids), :]
                batch_input_ids_list.append(self._to_pinned_cpu_tensor(input_ids, dtype=torch.int64)) # 先转tensor，方便pad_sequence
                batch_loss_mask_list.append(self._to_pinned_cpu_tensor(loss_mask, dtype=torch.float32))
                if expert_ids is not None:
                    batch_expert_ids_list.append(self._to_pinned_cpu_tensor(expert_ids, dtype=torch.int32))
        
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
        processed_expert_ids = [] if batch_expert_ids_list is not None else None

        for idx, (ids, mask) in enumerate(zip(batch_input_ids_list, batch_loss_mask_list)):
            ids = ids[:max_len_for_padding]
            mask = mask[:max_len_for_padding]
            processed_input_ids.append(ids)
            processed_loss_masks.append(mask)
            if use_opd_path:
                processed_teacher_logps.append(batch_teacher_logps_list[idx][:max_len_for_padding])
            if processed_expert_ids is not None:
                processed_expert_ids.append(batch_expert_ids_list[idx][:, :max_len_for_padding, :])
        
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
        padded_expert_ids_tensor = None
        if processed_expert_ids is not None:
            padded_expert_ids_tensor = torch.nn.utils.rnn.pad_sequence(
                [tensor.permute(1, 0, 2) for tensor in processed_expert_ids],
                batch_first=True,
                padding_value=-1,
            ).permute(0, 2, 1, 3).to(dist_utils.get_device(), non_blocking=True)

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
            if padded_expert_ids_tensor is not None:
                padded_expert_ids_tensor = torch.cat(
                    [
                        padded_expert_ids_tensor,
                        torch.full(
                            (
                                padded_expert_ids_tensor.size(0),
                                padded_expert_ids_tensor.size(1),
                                padding_needed,
                                padded_expert_ids_tensor.size(3),
                            ),
                            -1,
                            dtype=torch.int32,
                            device=dist_utils.get_device(),
                        ),
                    ],
                    dim=2,
                )
        
        final_input_ids = padded_input_ids_tensor[:, :-1].contiguous()
        final_labels = padded_input_ids_tensor[:, 1:].contiguous()
        final_loss_mask = padded_loss_mask_tensor[:, 1:].contiguous()
        final_teacher_logps = None if padded_teacher_logps_tensor is None else padded_teacher_logps_tensor[:, 1:].contiguous()
        final_expert_ids = (
            None
            if padded_expert_ids_tensor is None
            else padded_expert_ids_tensor[:, :, :-1, :].contiguous()
        )

        return BatchExperience(
            input_ids=final_input_ids,
            labels=final_labels,
            loss_mask=final_loss_mask,
            teacher_logps=final_teacher_logps,
            expert_ids=final_expert_ids,
            is_opd=use_opd_path,
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

    def train_step(
        self,
        data_iterator: Optional[Iterator],
        num_microbatches: int,
        train_metrics: Dict[str, list],
        iter_num: int,
        batch_num: int,
        is_thd_packed: bool = False,
    ):
        self.logger.debug("train step")
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        # self.optim.zero_grad()
        model_config = get_model_config(self.model[0])
        external_expert_ids_enabled = model_config.enable_external_expert_ids
        external_expert_ids_spec = self._external_expert_ids_spec()
        if external_expert_ids_enabled and not is_thd_packed:
            raise RuntimeError("external expert IDs currently require THD sequence packing")
        if (
            external_expert_ids_enabled
            and model_config.virtual_pipeline_model_parallel_size is not None
        ):
            raise RuntimeError("external expert IDs do not support virtual pipeline parallelism")

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

            def _cp_global_batch_mean(per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                loss_sum = (per_token_loss * mask).sum(dim=-1)
                valid_tokens = mask.sum(dim=-1)
                stats = torch.stack((loss_sum, valid_tokens), dim=0)
                if mpu.get_context_parallel_world_size() > 1:
                    stats = _TPSumForwardIdentityBackward.apply(stats, mpu.get_context_parallel_group())
                return stats[0] / (stats[1] + 1e-8)

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
            else:
                assert advantages is not None
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2)
            actor_loss = _cp_global_batch_mean(actor_loss, loss_mask) # (batch,)
            policy_loss = actor_loss.mean()

            # kl loss
            kl_loss = None
            if (self.args.init_kl_coef > 1e-8) and self.args.use_kl_loss:
                kl = compute_approx_kl(
                    actor_logps, 
                    ref_logps, 
                    kl_estimator=self.args.kl_estimator
                )
                kl_loss = _cp_global_batch_mean(kl, loss_mask) # (batch,)
                kl_loss = kl_loss.mean()

            # entropy loss
            entropy_loss = None
            if self.args.entropy_loss_coef > -1:
                unmasked_entropy_loss = vocab_parallel_entropy(output_tensor) # (batch, length)
                entropy_loss = _cp_global_batch_mean(unmasked_entropy_loss, loss_mask) # (batch,)
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

            def _cp_global_batch_mean(per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                loss_sum = (per_token_loss * mask).sum(dim=-1)
                valid_tokens = mask.sum(dim=-1)
                stats = torch.stack((loss_sum, valid_tokens), dim=0)
                if mpu.get_context_parallel_world_size() > 1:
                    stats = _TPSumForwardIdentityBackward.apply(stats, mpu.get_context_parallel_group())
                return stats[0] / (stats[1] + 1e-8)

            student_logps = from_parallel_logits_to_logprobs(output_tensor, labels, higher_stability=True, ignore_last=False)
            opd_clip_ratio = None
            with torch.no_grad():
                advantages = teacher_logps - student_logps.detach()
                opd_clip_coef = self._get_opd_clip_coef()
                if opd_clip_coef is not None:
                    clip_floor = math.log(opd_clip_coef) / (1.0 - opd_clip_coef)
                    clipped_tokens = (advantages < clip_floor).to(loss_mask.dtype)
                    advantages = torch.clamp_min(advantages, clip_floor)
                    opd_clip_ratio = _cp_global_batch_mean(clipped_tokens, loss_mask).mean()

            logp_diff = student_logps - student_logps.detach()
            ratio = logp_diff.exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2)
            actor_loss = _cp_global_batch_mean(actor_loss, loss_mask)
            policy_loss = actor_loss.mean()

            entropy_loss = None
            if self.args.entropy_loss_coef > -1:
                unmasked_entropy_loss = vocab_parallel_entropy(output_tensor)
                entropy_loss = _cp_global_batch_mean(unmasked_entropy_loss, loss_mask)
                entropy_loss = entropy_loss.mean()

            loss = self.args.policy_loss_coef * policy_loss
            losses_reduced = torch.cat(
                [
                    loss.clone().detach().unsqueeze(dim=0),
                    policy_loss.clone().detach().unsqueeze(dim=0),
                    torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                    entropy_loss.clone().detach().unsqueeze(dim=0) if entropy_loss is not None else \
                        torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                    opd_clip_ratio.clone().detach().unsqueeze(dim=0) if opd_clip_ratio is not None else \
                        torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                ], dim=0
            )
            dist.all_reduce(losses_reduced, op=ReduceOp.AVG, group=mpu.get_data_parallel_group())
            return loss, {"losses": losses_reduced}

        def opd_packing_loss_func(
            output_tensor: torch.Tensor,
            labels: torch.Tensor,
            loss_mask: torch.Tensor,
            teacher_logps: torch.Tensor,
            cu_seqlens: torch.Tensor,
            num_real_samples_in_microbatch: torch.Tensor,
            packed_microbatch_index: int,
            non_loss_data: bool = False,
        ):
            assert non_loss_data == False
            num_real_samples = int(num_real_samples_in_microbatch.item())

            def _cp_sample_means(per_token_loss: torch.Tensor) -> torch.Tensor:
                if num_real_samples == 0:
                    return per_token_loss.new_zeros((0,), dtype=torch.float32)

                cu = cu_seqlens[: num_real_samples + 1].to(dtype=torch.long)
                flat_loss = (per_token_loss.float() * loss_mask.float()).reshape(-1)
                flat_mask = loss_mask.float().reshape(-1)
                prefix_loss = torch.cat(
                    [flat_loss.new_zeros((1,)), torch.cumsum(flat_loss, dim=0)], dim=0
                )
                prefix_mask = torch.cat(
                    [flat_mask.new_zeros((1,)), torch.cumsum(flat_mask, dim=0)], dim=0
                )
                sample_loss_sum = prefix_loss[cu[1:]] - prefix_loss[cu[:-1]]
                sample_valid_tokens = prefix_mask[cu[1:]] - prefix_mask[cu[:-1]]
                stats = torch.stack((sample_loss_sum, sample_valid_tokens), dim=0)
                if mpu.get_context_parallel_world_size() > 1:
                    stats = _TPSumForwardIdentityBackward.apply(stats, mpu.get_context_parallel_group())
                return stats[0] / (stats[1] + 1e-8)

            student_logps = from_parallel_logits_to_logprobs(
                output_tensor, labels, higher_stability=True, ignore_last=False
            )
            opd_clip_ratio = None
            clipped_tokens = None
            clipped_sample_ratio = None
            logp_diff = None
            ratio = None
            surr1 = None
            surr2 = None
            with torch.no_grad():
                advantages = teacher_logps - student_logps.detach()
                opd_clip_coef = self._get_opd_clip_coef()
                if opd_clip_coef is not None:
                    clip_floor = math.log(opd_clip_coef) / (1.0 - opd_clip_coef)
                    clipped_tokens = (advantages < clip_floor).to(loss_mask.dtype)
                    advantages = torch.clamp_min(advantages, clip_floor)
                    clipped_sample_ratio = _cp_sample_means(clipped_tokens)
                    if clipped_sample_ratio.numel() > 0:
                        opd_clip_ratio = clipped_sample_ratio.mean()

            if num_real_samples == 0:
                policy_loss = output_tensor.float().sum() * 0.0
                entropy_loss = output_tensor.float().sum() * 0.0
                loss = output_tensor.float().sum() * 0.0
                actor_loss = None
                sample_actor_loss = None
                microbatch_sample_mean_sum = None
                unmasked_entropy_loss = None
                sample_entropy = None
            else:
                logp_diff = student_logps - student_logps.detach()
                ratio = logp_diff.exp()
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2)
                sample_actor_loss = _cp_sample_means(actor_loss)
                policy_loss = sample_actor_loss.mean()
                microbatch_sample_mean_sum = sample_actor_loss.sum()

                entropy_loss = None
                unmasked_entropy_loss = None
                sample_entropy = None
                if self.args.entropy_loss_coef > -1:
                    unmasked_entropy_loss = vocab_parallel_entropy(output_tensor)
                    sample_entropy = _cp_sample_means(unmasked_entropy_loss)
                    entropy_loss = sample_entropy.mean()

                loss = self.args.policy_loss_coef * microbatch_sample_mean_sum

            if os.environ.get("LS_PACKING_DUMP") == "1":
                dump_path = os.environ.get("LS_PACKING_DUMP_PATH")
                if not dump_path:
                    raise RuntimeError(
                        "LS_PACKING_DUMP_PATH must be set when LS_PACKING_DUMP=1."
                    )
                os.makedirs(dump_path, exist_ok=True)
                prefix = (
                    f"{dump_path}/iter_{iter_num}_batch_{batch_num}_"
                    f"tp_{mpu.get_tensor_model_parallel_rank()}_"
                    f"pp_{mpu.get_pipeline_model_parallel_rank()}_"
                    f"dp_{mpu.get_data_parallel_rank()}_"
                    f"microbatch_{packed_microbatch_index}"
                )
                output_tensors = {
                    "output_tensor": output_tensor,
                    "student_logps": student_logps,
                    "advantages": advantages,
                    "clipped_tokens": clipped_tokens,
                    "clipped_sample_ratio": clipped_sample_ratio,
                    "opd_clip_ratio": opd_clip_ratio,
                    "logp_diff": logp_diff,
                    "ratio": ratio,
                    "surr1": surr1,
                    "surr2": surr2,
                    "actor_loss": actor_loss,
                    "sample_actor_loss": sample_actor_loss,
                    "microbatch_sample_mean_sum": microbatch_sample_mean_sum,
                    "policy_loss": policy_loss,
                    "unmasked_entropy_loss": unmasked_entropy_loss,
                    "sample_entropy": sample_entropy,
                    "entropy_loss": entropy_loss,
                    "loss": loss,
                }
                for name, tensor in output_tensors.items():
                    if tensor is not None:
                        tensor_to_dump = tensor.detach()
                        if tensor_to_dump.dtype == torch.bfloat16:
                            tensor_to_dump = tensor_to_dump.float()
                        np.save(f"{prefix}_{name}.npy", tensor_to_dump.cpu().numpy())

            losses_reduced = torch.cat(
                [
                    (self.args.policy_loss_coef * policy_loss).clone().detach().unsqueeze(dim=0),
                    policy_loss.clone().detach().unsqueeze(dim=0),
                    torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                    entropy_loss.clone().detach().unsqueeze(dim=0) if entropy_loss is not None else \
                        torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                    opd_clip_ratio.clone().detach().unsqueeze(dim=0) if opd_clip_ratio is not None else \
                        torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                ], dim=0
            )
            dist.all_reduce(losses_reduced, op=ReduceOp.AVG, group=mpu.get_data_parallel_group())
            return loss, {"losses": losses_reduced}

        def grpo_packing_loss_func(
            output_tensor: torch.Tensor,
            labels: torch.Tensor,
            loss_mask: torch.Tensor,
            sample_outcome_rewards: torch.Tensor,
            cu_seqlens: torch.Tensor,
            num_real_samples_in_microbatch: torch.Tensor,
            non_loss_data: bool = False,
        ):
            assert non_loss_data == False
            assert self.args.use_outcome_rewards_as_advantages
            assert not self.is_old_actor_logp_required
            assert self.args.init_kl_coef <= 1e-8

            num_real_samples = int(num_real_samples_in_microbatch.item())

            def _cp_sample_means(per_token_loss: torch.Tensor) -> torch.Tensor:
                if num_real_samples == 0:
                    return per_token_loss.new_zeros((0,), dtype=torch.float32)

                cu = cu_seqlens[: num_real_samples + 1].to(dtype=torch.long)
                flat_loss = (per_token_loss.float() * loss_mask.float()).reshape(-1)
                flat_mask = loss_mask.float().reshape(-1)
                prefix_loss = torch.cat(
                    [flat_loss.new_zeros((1,)), torch.cumsum(flat_loss, dim=0)], dim=0
                )
                prefix_mask = torch.cat(
                    [flat_mask.new_zeros((1,)), torch.cumsum(flat_mask, dim=0)], dim=0
                )
                sample_loss_sum = prefix_loss[cu[1:]] - prefix_loss[cu[:-1]]
                sample_valid_tokens = prefix_mask[cu[1:]] - prefix_mask[cu[:-1]]
                stats = torch.stack((sample_loss_sum, sample_valid_tokens), dim=0)
                if mpu.get_context_parallel_world_size() > 1:
                    stats = _TPSumForwardIdentityBackward.apply(stats, mpu.get_context_parallel_group())
                return stats[0] / (stats[1] + 1e-8)

            if num_real_samples == 0:
                policy_loss = output_tensor.float().sum() * 0.0
                entropy_loss = None
                loss = output_tensor.float().sum() * 0.0
            else:
                cu = cu_seqlens[: num_real_samples + 1].to(dtype=torch.long)
                sample_outcome_rewards = sample_outcome_rewards.reshape(-1)
                assert sample_outcome_rewards.numel() == num_real_samples

                local_sample_lengths = cu[1:] - cu[:-1]
                packed_token_count = loss_mask.numel()
                assert int(cu[-1].item()) <= packed_token_count
                token_outcome_rewards = torch.repeat_interleave(
                    sample_outcome_rewards.to(dtype=output_tensor.dtype), local_sample_lengths
                )
                if token_outcome_rewards.numel() < packed_token_count:
                    token_outcome_rewards = torch.nn.functional.pad(
                        token_outcome_rewards,
                        (0, packed_token_count - token_outcome_rewards.numel()),
                    )
                token_outcome_rewards = token_outcome_rewards.reshape_as(loss_mask)

                actor_logps = from_parallel_logits_to_logprobs(
                    output_tensor, labels, higher_stability=True, ignore_last=False
                )
                logp_diff = actor_logps - actor_logps.detach()
                ratio = logp_diff.exp()
                surr1 = ratio * token_outcome_rewards
                surr2 = ratio.clamp(1 - self.args.clip_eps, 1 + self.args.clip_eps) * token_outcome_rewards
                actor_loss = -torch.min(surr1, surr2)
                sample_actor_loss = _cp_sample_means(actor_loss)
                policy_loss = sample_actor_loss.mean()
                loss = self.args.policy_loss_coef * sample_actor_loss.sum()

                entropy_loss = None
                if self.args.entropy_loss_coef > -1:
                    sample_entropy = _cp_sample_means(vocab_parallel_entropy(output_tensor))
                    entropy_loss = sample_entropy.mean()

            losses_reduced = torch.cat(
                [
                    (self.args.policy_loss_coef * policy_loss).clone().detach().unsqueeze(dim=0),
                    policy_loss.clone().detach().unsqueeze(dim=0),
                    torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                    entropy_loss.clone().detach().unsqueeze(dim=0) if entropy_loss is not None else \
                        torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                    torch.tensor(0.0, dtype=torch.float32, device=dist_utils.get_device()).unsqueeze(dim=0),
                ], dim=0
            )
            dist.all_reduce(losses_reduced, op=ReduceOp.AVG, group=mpu.get_data_parallel_group())
            return loss, {"losses": losses_reduced}

        def _sync_packed_seq_params_to_pipeline(
            global_cu_seqlens: Optional[torch.Tensor],
            global_cu_seqlens_padded: Optional[torch.Tensor],
            global_max_seqlen: Optional[torch.Tensor],
        ) -> Optional[PackedSeqParams]:
            """Pipeline-sync TE global THD metadata prepared by the dataset.

            Local cumulative boundaries remain exclusive to packed losses. TE needs
            global regular and padded boundaries, which must have matching interval
            counts even when a packed bucket has a physical tail-padding segment.
            """
            cu_seqlens = global_cu_seqlens
            cu_seqlens_padded = global_cu_seqlens_padded
            max_seqlen = global_max_seqlen

            if cu_seqlens is not None:
                assert cu_seqlens_padded is not None and max_seqlen is not None
                assert cu_seqlens.ndim == cu_seqlens_padded.ndim == 1
                assert len(cu_seqlens) == len(cu_seqlens_padded)
                assert int(cu_seqlens[0].item()) == int(cu_seqlens_padded[0].item()) == 0
                assert int(cu_seqlens[-1].item()) <= int(cu_seqlens_padded[-1].item())

            pipeline_group = mpu.get_pipeline_model_parallel_group()
            if pipeline_group is None or mpu.get_pipeline_model_parallel_world_size() == 1:
                if cu_seqlens is None or max_seqlen is None:
                    return None
                return PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    cu_seqlens_q_padded=cu_seqlens_padded,
                    cu_seqlens_kv_padded=cu_seqlens_padded,
                    max_seqlen_q=int(max_seqlen.item()),
                    max_seqlen_kv=int(max_seqlen.item()),
                )

            pipeline_src_rank = mpu.get_pipeline_model_parallel_first_rank()
            if dist.get_rank() == pipeline_src_rank:
                assert cu_seqlens is not None and max_seqlen is not None
                length_tensor = torch.tensor(
                    [cu_seqlens.shape[0], 0 if cu_seqlens_padded is None else cu_seqlens_padded.shape[0]],
                    dtype=torch.int32,
                    device=dist_utils.get_device(),
                )
            else:
                length_tensor = torch.empty((2,), dtype=torch.int32, device=dist_utils.get_device())
            dist.broadcast(length_tensor, src=pipeline_src_rank, group=pipeline_group)
            cu_len = int(length_tensor[0].item())
            padded_cu_len = int(length_tensor[1].item())
            if dist.get_rank() != pipeline_src_rank:
                cu_seqlens = torch.empty((cu_len,), dtype=torch.int32, device=dist_utils.get_device())
                max_seqlen = torch.empty((1,), dtype=torch.int32, device=dist_utils.get_device())
                cu_seqlens_padded = (
                    torch.empty((padded_cu_len,), dtype=torch.int32, device=dist_utils.get_device())
                    if padded_cu_len > 0
                    else None
                )
            dist.broadcast(cu_seqlens, src=pipeline_src_rank, group=pipeline_group)
            if padded_cu_len > 0:
                dist.broadcast(cu_seqlens_padded, src=pipeline_src_rank, group=pipeline_group)
            dist.broadcast(max_seqlen, src=pipeline_src_rank, group=pipeline_group)
            return PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                max_seqlen_q=int(max_seqlen.item()),
                max_seqlen_kv=int(max_seqlen.item()),
            )

        def _stage_packed_seq_params(
            packed_data_iterator: Optional[Iterator],
        ) -> Tuple[
            Optional[Iterator],
            List[PackedSeqParams],
            Optional[List[Dict[str, torch.Tensor]]],
        ]:
            """Materialize endpoint batches and stage THD metadata before PP P2P.

            A non-first PP stage receives its activation before its scheduled
            forward callback runs. A PP broadcast inside that callback therefore
            forms a cycle with activation P2P. All ranks must complete the
            metadata broadcasts before the schedule starts.
            """
            if dist.get_rank() == 0:
                self.logger.info(
                    f"iter {iter_num} step {batch_num} syncing packed seq params"
                )

            prepared_microbatches: Optional[List[Dict[str, torch.Tensor]]] = None
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                assert packed_data_iterator is not None
                prepared_microbatches = list(packed_data_iterator)
                if len(prepared_microbatches) != num_microbatches:
                    raise RuntimeError(
                        "Prepared packed microbatch count does not match the pipeline schedule: "
                        f"prepared={len(prepared_microbatches)}, scheduled={num_microbatches}."
                    )
                packed_data_iterator = iter(prepared_microbatches)

            packed_seq_params_cache: List[PackedSeqParams] = []
            for microbatch_idx in range(num_microbatches):
                cu_src = None
                cu_padded_src = None
                max_src = None
                if mpu.is_pipeline_first_stage():
                    assert prepared_microbatches is not None
                    batched_item = prepared_microbatches[microbatch_idx]
                    cu_src = batched_item["cu_seqlens_global"][0].to(
                        device=dist_utils.get_device(), non_blocking=True
                    )
                    cu_padded_src = batched_item["cu_seqlens_global_padded"][0].to(
                        device=dist_utils.get_device(), non_blocking=True
                    )
                    max_src = batched_item["max_seqlen_global"][0].to(
                        device=dist_utils.get_device(), non_blocking=True
                    )
                packed_seq_params = _sync_packed_seq_params_to_pipeline(
                    cu_src, cu_padded_src, max_src
                )
                assert packed_seq_params is not None
                packed_seq_params_cache.append(packed_seq_params)
            return packed_data_iterator, packed_seq_params_cache, prepared_microbatches

        def _get_local_global_moe_ordinals() -> torch.Tensor:
            """Map this stage's actual MoE layers to global logical MoE ordinals."""
            global_moe_layer_numbers = []
            for layer_index in range(model_config.num_layers):
                if isinstance(model_config.moe_layer_freq, int):
                    is_moe_layer = layer_index % model_config.moe_layer_freq == 0
                else:
                    is_moe_layer = bool(model_config.moe_layer_freq[layer_index])
                if is_moe_layer:
                    global_moe_layer_numbers.append(layer_index + 1)
            ordinal_by_layer_number = {
                layer_number: ordinal
                for ordinal, layer_number in enumerate(global_moe_layer_numbers)
            }
            core_model = unwrap_model(self.model)[0]
            local_layer_numbers = [
                layer.layer_number
                for layer in core_model.decoder.layers
                if getattr(layer, "is_moe_layer", False)
            ]
            try:
                ordinals = [ordinal_by_layer_number[layer_number] for layer_number in local_layer_numbers]
            except KeyError as error:
                raise RuntimeError(
                    f"Local MoE layer {error.args[0]} is absent from the configured global MoE layout"
                ) from error
            return torch.tensor(ordinals, dtype=torch.int32, device=dist_utils.get_device())

        def _stage_external_routing_metadata(
            prepared_microbatches: Optional[List[Dict[str, torch.Tensor]]],
        ) -> List[Tuple[Optional[torch.Tensor], torch.Tensor]]:
            """Synchronize route metadata before pipeline activation P2P starts."""
            pipeline_group = mpu.get_pipeline_model_parallel_group()
            pipeline_world_size = mpu.get_pipeline_model_parallel_world_size()
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
            pipeline_src_rank = mpu.get_pipeline_model_parallel_first_rank()

            local_ordinals = _get_local_global_moe_ordinals()
            local_count = torch.tensor([local_ordinals.numel()], dtype=torch.int32, device=local_ordinals.device)
            ordinal_counts = [torch.empty_like(local_count) for _ in range(pipeline_world_size)]
            dist.all_gather(ordinal_counts, local_count, group=pipeline_group)
            max_ordinal_count = max(int(count.item()) for count in ordinal_counts)
            padded_ordinals = torch.full(
                (max_ordinal_count,), -1, dtype=torch.int32, device=local_ordinals.device
            )
            padded_ordinals[: local_ordinals.numel()] = local_ordinals
            all_ordinals = [torch.empty_like(padded_ordinals) for _ in range(pipeline_world_size)]
            dist.all_gather(all_ordinals, padded_ordinals, group=pipeline_group)
            all_active_ordinals = torch.cat(
                [ordinals[: int(count.item())] for ordinals, count in zip(all_ordinals, ordinal_counts)]
            )
            expected_ordinals = torch.arange(
                external_expert_ids_spec[0], dtype=torch.int32, device=local_ordinals.device
            )
            if not torch.equal(torch.sort(all_active_ordinals).values, expected_ordinals):
                raise RuntimeError(
                    "Pipeline MoE layers do not map one-to-one to global external expert-ID ordinals."
                )

            if mpu.is_pipeline_first_stage():
                assert prepared_microbatches is not None
                assert len(prepared_microbatches) == num_microbatches

            metadata_cache = []
            for microbatch_idx in range(num_microbatches):
                if dist.get_rank() == pipeline_src_rank:
                    source_microbatch = prepared_microbatches[microbatch_idx]
                    source_expert_ids = source_microbatch["expert_ids"]
                    source_router_padding_mask = source_microbatch["router_padding_mask"]
                    if source_expert_ids is None or source_router_padding_mask is None:
                        raise RuntimeError("Packed external routing metadata is missing.")
                    router_padding_mask = source_router_padding_mask.to(
                        device=dist_utils.get_device(), dtype=torch.bool, non_blocking=True
                    )
                else:
                    router_padding_mask = torch.empty(
                        (self.args.seq_length // mpu.get_context_parallel_world_size(),),
                        dtype=torch.bool,
                        device=dist_utils.get_device(),
                    )
                dist.broadcast(router_padding_mask, src=pipeline_src_rank, group=pipeline_group)

                local_expert_ids = None
                for stage_rank, (ordinal_count, stage_ordinals) in enumerate(
                    zip(ordinal_counts, all_ordinals)
                ):
                    stage_count = int(ordinal_count.item())
                    if stage_count == 0:
                        continue
                    if dist.get_rank() == pipeline_src_rank:
                        source_indices = stage_ordinals[:stage_count].to(dtype=torch.long)
                        stage_expert_ids = source_expert_ids.index_select(0, source_indices).to(
                            device=dist_utils.get_device(), dtype=torch.int32, non_blocking=True
                        )
                    else:
                        stage_expert_ids = torch.empty(
                            (stage_count, router_padding_mask.numel(), model_config.moe_router_topk),
                            dtype=torch.int32,
                            device=dist_utils.get_device(),
                        )
                    dist.broadcast(stage_expert_ids, src=pipeline_src_rank, group=pipeline_group)
                    if pipeline_rank == stage_rank:
                        local_expert_ids = stage_expert_ids
                metadata_cache.append((local_expert_ids, router_padding_mask))
            return metadata_cache

        def _dump_staged_packing_inputs(
            prepared_microbatches: Optional[List[Dict[str, torch.Tensor]]],
            packed_seq_params_cache: List[PackedSeqParams],
        ) -> None:
            """Optionally dump the staged THD inputs without adding collectives.

            Every rank enters this function. Pipeline endpoints own the complete
            planned microbatch dictionaries, while intermediate stages only own
            the pipeline-synchronized TE packed-sequence metadata.
            """
            if os.environ.get("LS_PACKING_DUMP") != "1":
                return

            dump_path = os.environ.get("LS_PACKING_DUMP_PATH")
            if not dump_path:
                raise RuntimeError(
                    "LS_PACKING_DUMP_PATH must be set when LS_PACKING_DUMP=1."
                )

            os.makedirs(dump_path, exist_ok=True)
            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            dp_rank = mpu.get_data_parallel_rank()

            def _save_array(prefix: str, name: str, value) -> None:
                if isinstance(value, torch.Tensor):
                    array = value.detach().cpu().numpy()
                else:
                    array = np.asarray(value)
                np.save(f"{prefix}_{name}.npy", array)

            for microbatch_idx, packed_seq_params in enumerate(packed_seq_params_cache):
                prefix = (
                    f"{dump_path}/iter_{iter_num}_batch_{batch_num}_tp_{tp_rank}_"
                    f"pp_{pp_rank}_dp_{dp_rank}_microbatch_{microbatch_idx}"
                )
                if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                    assert prepared_microbatches is not None
                    for name, value in prepared_microbatches[microbatch_idx].items():
                        if isinstance(value, torch.Tensor):
                            _save_array(prefix, name, value)
                else:
                    _save_array(prefix, "cu_seqlens_global", packed_seq_params.cu_seqlens_q)
                    _save_array(
                        prefix,
                        "cu_seqlens_global_padded",
                        packed_seq_params.cu_seqlens_q_padded,
                    )
                    _save_array(prefix, "max_seqlen_global", packed_seq_params.max_seqlen_q)

        packed_seq_params_cache: Optional[List[PackedSeqParams]] = None
        prepared_packed_microbatches: Optional[List[Dict[str, torch.Tensor]]] = None
        external_routing_metadata_cache: Optional[List[Tuple[Optional[torch.Tensor], torch.Tensor]]] = None
        if is_thd_packed:
            (
                data_iterator,
                packed_seq_params_cache,
                prepared_packed_microbatches,
            ) = _stage_packed_seq_params(data_iterator)
            _dump_staged_packing_inputs(
                prepared_packed_microbatches,
                packed_seq_params_cache,
            )
            if external_expert_ids_enabled:
                external_routing_metadata_cache = _stage_external_routing_metadata(
                    prepared_packed_microbatches
                )

        packed_forward_index = 0

        def megatron_forward_step(data_iterator, model):
            nonlocal packed_forward_index
            forward_args = {
                "input_ids": None,
                "position_ids": None,
                "attention_mask": None,
                "packed_seq_params": None,
            }
            loss_func_args = {}
            selected_loss_func = loss_func
            packed_seq_params = None
            packed_microbatch_index = None
            if is_thd_packed:
                assert packed_seq_params_cache is not None
                if packed_forward_index >= len(packed_seq_params_cache):
                    raise RuntimeError("Pipeline schedule requested more packed microbatches than prepared.")
                packed_microbatch_index = packed_forward_index
                packed_seq_params = packed_seq_params_cache[packed_forward_index]
                if external_expert_ids_enabled:
                    assert external_routing_metadata_cache is not None
                    expert_ids, router_padding_mask = external_routing_metadata_cache[
                        packed_forward_index
                    ]
                    forward_args["expert_ids"] = expert_ids
                    forward_args["padding_mask"] = router_padding_mask
                packed_forward_index += 1
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                assert data_iterator is not None
                batched_item: Dict[str, torch.Tensor] = next(data_iterator)
                batch_is_thd_packed = bool(
                    batched_item.get("is_thd_packed", torch.tensor(0)).item()
                )
                batch_is_opd = bool(batched_item.get("is_opd", torch.tensor(0)).item())
                if batch_is_thd_packed != is_thd_packed:
                    raise RuntimeError(
                        "Packed-path consistency check failed between the planned iterator "
                        "and the training-step schedule."
                    )
                if mpu.is_pipeline_first_stage():
                    forward_args["input_ids"] = batched_item["input_ids"].to(device=dist_utils.get_device(), non_blocking=True)
                if mpu.is_pipeline_last_stage():
                    labels = batched_item["labels"].to(device=dist_utils.get_device(), non_blocking=True)
                    loss_mask = batched_item["loss_mask"].to(device=dist_utils.get_device(), non_blocking=True)
                    if batch_is_opd:
                        if batched_item.get("teacher_logps") is None:
                            raise RuntimeError("OPD batch is missing teacher_logps.")
                        teacher_logps = batched_item["teacher_logps"].to(device=dist_utils.get_device(), non_blocking=True)
                        if batch_is_thd_packed:
                            selected_loss_func = opd_packing_loss_func
                            loss_func_args = {
                                "labels": labels,
                                "loss_mask": loss_mask,
                                "teacher_logps": teacher_logps,
                                "cu_seqlens": batched_item["cu_seqlens"][0].to(device=dist_utils.get_device(), non_blocking=True),
                                "num_real_samples_in_microbatch": batched_item["num_real_samples_in_microbatch"][0].to(device=dist_utils.get_device(), non_blocking=True),
                                "packed_microbatch_index": packed_microbatch_index,
                            }
                        else:
                            selected_loss_func = opd_loss_func
                            loss_func_args = {
                                "labels": labels,
                                "loss_mask": loss_mask,
                                "teacher_logps": teacher_logps,
                            }
                    else:
                        if batch_is_thd_packed:
                            selected_loss_func = grpo_packing_loss_func
                            loss_func_args = {
                                "labels": labels,
                                "loss_mask": loss_mask,
                                "sample_outcome_rewards": batched_item["sample_outcome_rewards"].to(device=dist_utils.get_device(), non_blocking=True),
                                "cu_seqlens": batched_item["cu_seqlens"][0].to(device=dist_utils.get_device(), non_blocking=True),
                                "num_real_samples_in_microbatch": batched_item["num_real_samples_in_microbatch"][0].to(device=dist_utils.get_device(), non_blocking=True),
                            }
                        else:
                            loss_func_args = {
                                "labels": labels,
                                "loss_mask": loss_mask,
                                "debug_i": batched_item["idx"],
                                "outcome_rewards": batched_item["outcome_rewards"].to(device=dist_utils.get_device(), non_blocking=True),
                            }
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

            forward_args["packed_seq_params"] = packed_seq_params

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
            losses_for_report = torch.zeros((5,), dtype=torch.float32, device=dist_utils.get_device())

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
        opd_clip_ratio_item = losses_for_report[4].item()
        # lr = None
        # for param_group in self.optim.param_groups:
        #     if param_group['is_decoupled_lr']:
        #         lr = param_group['lr']
        #     else:
        #         lr = param_group['lr']
        self.logger.info(f"iter: {iter_num}, batch: {batch_num}, loss: {loss_item}, policy_loss: {policy_loss_item}, kl_loss: {kl_loss_item:.2e}, entropy_loss: {entropy_loss_item}, opd_clip_ratio: {opd_clip_ratio_item}")
        train_metrics["loss"].append(loss_item)
        train_metrics["policy_loss"].append(policy_loss_item)
        train_metrics["kl_loss"].append(kl_loss_item)
        train_metrics["entropy_loss"].append(entropy_loss_item)
        train_metrics["opd_clip_ratio"].append(opd_clip_ratio_item)

        
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

    def track_auxiliary_losses(self, iter_num: int) -> None:
        """Reduce and log MoE/MTP losses accumulated across one async update."""
        global_num_microbatches = self.args.global_batch_size // (
            self.args.micro_batch_size * mpu.get_data_parallel_world_size()
        )
        loss_scale = 1 / global_num_microbatches
        auxiliary_metrics = {}

        if self.args.num_experts is not None:
            track_names = []
            routing_types = self.args.moe_router_load_balancing_type
            if isinstance(routing_types, str):
                routing_types = [routing_types]
            if "aux_loss" in routing_types:
                track_names.append("load_balancing_loss")
            if "seq_aux_loss" in routing_types:
                track_names.append("seq_load_balancing_loss")
            if "global_aux_loss" in routing_types:
                track_names.append("global_load_balancing_loss")
            if self.args.moe_z_loss_coeff is not None:
                track_names.append("z_loss")
            track_moe_metrics(
                loss_scale=loss_scale,
                iteration=iter_num,
                writer=self.tensorboard_writer,
                wandb_writer=None,
                total_loss_dict=auxiliary_metrics,
                per_layer_logging=self.args.moe_per_layer_logging,
                force_initialize=True,
                track_names=track_names,
                num_layers=self.args.num_layers,
                moe_layer_freq=self.args.moe_layer_freq,
                mtp_num_layers=self.args.mtp_num_layers,
            )

        if self.args.mtp_num_layers:
            mtp_metrics = {}
            if mpu.is_pipeline_last_stage():
                MTPLossLoggingHelper.track_mtp_metrics(
                    loss_scale=loss_scale,
                    iteration=iter_num,
                    writer=(
                        self.tensorboard_writer
                        if mpu.get_pipeline_model_parallel_world_size() == 1
                        else None
                    ),
                    wandb_writer=None,
                    total_loss_dict=mtp_metrics,
                )

            if mpu.get_pipeline_model_parallel_world_size() == 1:
                auxiliary_metrics.update(mtp_metrics)
            elif mpu.get_tensor_and_context_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():
                    mtp_losses = torch.empty(
                        self.args.mtp_num_layers, device=dist_utils.get_device()
                    )
                    dist.recv(mtp_losses, src=mpu.get_pipeline_model_parallel_last_rank())
                    for mtp_layer_id, mtp_loss in enumerate(mtp_losses, start=1):
                        name = f"mtp_{mtp_layer_id} loss"
                        auxiliary_metrics[name] = mtp_loss
                        if self.tensorboard_writer is not None:
                            self.tensorboard_writer.add_scalar(name, mtp_loss, iter_num)
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.flush()
                elif mpu.is_pipeline_last_stage():
                    mtp_losses = torch.stack(
                        [
                            mtp_metrics[f"mtp_{mtp_layer_id} loss"]
                            for mtp_layer_id in range(1, self.args.mtp_num_layers + 1)
                        ]
                    )
                    dist.send(mtp_losses, dst=mpu.get_pipeline_model_parallel_first_rank())

        if auxiliary_metrics and dist.get_rank() == 0:
            metric_fields = []
            for name, value in auxiliary_metrics.items():
                scalar_value = (
                    value.detach().float().mean().item()
                    if isinstance(value, torch.Tensor)
                    else float(value)
                )
                metric_fields.append(f"{name.replace(' ', '.')}={scalar_value:.6e}")
            self.logger.info(
                "iter %d / %d | auxiliary metrics | %s",
                iter_num,
                self.args.train_iters,
                " | ".join(metric_fields),
            )

    def log_train_metrics(self, train_metrics, iter_num):
        def _safe_avg(values: List[float], default: float = 0.0):
            return (sum(values) / len(values)) if len(values) > 0 else default

        loss_item = _safe_avg(train_metrics.get("loss", []), 0.0)
        policy_loss_item = _safe_avg(train_metrics.get("policy_loss", []), 0.0)
        kl_loss_item = _safe_avg(train_metrics.get("kl_loss", []), 0.0)
        entropy_loss_item = _safe_avg(train_metrics.get("entropy_loss", []), 0.0)
        opd_clip_ratio_item = _safe_avg(train_metrics.get("opd_clip_ratio", []), 0.0)
        distill_loss_item = _safe_avg(train_metrics.get("distill_loss", []), 0.0)
        grad_norm = train_metrics["grad_norm"]
        lr = train_metrics["learning_rate"]
        if self.args.distillation_enabled:
            self.logger.info(f"iter: {iter_num}, loss: {loss_item:.2e}, distill_loss: {distill_loss_item:.2e}, grad_norm: {grad_norm:.2e}, learning_rate: {lr:.2e}")
        else:
            self.logger.info(f"iter: {iter_num}, loss: {loss_item:.2e}, policy_loss: {policy_loss_item:.2e}, kl_loss: {kl_loss_item:.2e}, entropy_loss: {entropy_loss_item:.2e}, opd_clip_ratio: {opd_clip_ratio_item:.2e}, grad_norm: {grad_norm:.2e}, learning_rate: {lr:.2e}")

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
                writer.add_scalar("train/opd_clip_ratio", opd_clip_ratio_item, iter_num)
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
            part_1 = f"iter: {iter_num}, data_type: {data_type}, reward: {metrics_for_log['reward']:.4f}, completion tokens: {metrics_for_log['completion_tokens']}, total tokens: {metrics_for_log['total_tokens']}, invalid_samples: {metrics_for_log['invalid_samples']}, throughput: {int(throughput)} tokens/s/p, "
            metrics_for_log.pop("reward")
            metrics_for_log.pop("completion_tokens")
            metrics_for_log.pop("total_tokens")
            metrics_for_log.pop("invalid_samples")
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