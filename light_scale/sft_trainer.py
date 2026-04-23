from functools import partial
from typing import Any, Dict, Iterator, List, Optional
import random
import time
import math

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP, finalize_model_grads
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import save_checkpoint as save_megatron_checkpoint
from megatron.training.global_vars import get_args, get_tensorboard_writer
from megatron.training.training import disable_forward_pre_hook, enable_forward_pre_hook
from light_scale import dist_utils
from light_scale.logger_utils import setup_logger
from megatron.training.global_vars import get_args, get_tokenizer
from megatron.core.transformer.moe.moe_utils import track_moe_metrics


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler with resume support based on global iteration offset."""

    def __init__(
        self,
        dataset,
        *,
        resume_step: int,
        micro_batch_size: int,
        num_microbatches: int,
        **kwargs,
    ) -> None:
        super().__init__(dataset, **kwargs)

        per_rank_samples = micro_batch_size * num_microbatches
        consumed_samples_rank = resume_step * per_rank_samples

        if per_rank_samples == 0 or self.num_samples == 0:
            self._epoch_offset = 0
            self._initial_skip = 0
        else:
            self._epoch_offset = consumed_samples_rank // self.num_samples
            self._initial_skip = consumed_samples_rank % self.num_samples

        self._skip_pending = self._initial_skip > 0
        self._logical_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._logical_epoch = epoch
        super().set_epoch(epoch + self._epoch_offset)

    def __iter__(self):
        base_iter = super().__iter__()
        if not self._skip_pending:
            return base_iter

        indices = list(base_iter)
        skip = min(self._initial_skip, len(indices))
        indices = indices[skip:]
        self._skip_pending = False
        self._initial_skip = 0
        return iter(indices)


class _InfiniteDataIterator:
    """Wrap a DataLoader to provide an infinite stream with synchronized epochs."""

    def __init__(self, dataloader: DataLoader, sampler: Optional[DistributedSampler]) -> None:
        self.dataloader = dataloader
        self.sampler = sampler
        self._epoch = 0
        self._iterator: Optional[Iterator] = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._reset()
        try:
            return next(self._iterator)
        except StopIteration:
            self._reset()
            return next(self._iterator)

    def _reset(self):
        if self.sampler is not None:
            self.sampler.set_epoch(self._epoch)
        self._iterator = iter(self.dataloader)
        self._epoch += 1

# copy from NeMo
def _ceil_to_nearest(n, m, ceil_to_power_2=False):
    if ceil_to_power_2:
        # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
        return 2 ** math.ceil(math.log2(n))
    else:
        return (n + m - 1) // m * m

class SFTTrainer:
    def __init__(self, passed_iters: int, model, optim, scheduler) -> None:
        args = get_args()
        logger = setup_logger("light_scale")
        logger.warning(f"seed: {args.seed}")
        random.seed(args.seed)
        self.args = args
        self.logger = logger

        tokenizer = get_tokenizer()._tokenizer
        assert tokenizer.pad_token_id is not None
        logger.warning(f"pad_token_id: {tokenizer.pad_token_id}")
        self.tokenizer = tokenizer

        tensorboard_writer = None
        if dist.get_rank() == 0:
            assert args.tensorboard_dir is not None
            tensorboard_writer = SummaryWriter(
                log_dir=args.tensorboard_dir,
                filename_suffix="light_scale"
            )

        self.tensorboard_writer = tensorboard_writer

        data_sanity_check_passed = True
        if self._require_dataloader():
            logger.info(f"Loading SFT dataset from {args.data_path[0]}")
            dataset = load_from_disk(args.data_path[0])
            logger.info(f"{len(dataset)} samples loaded")

            data_sanity_check_passed = self.__dataset_sainity_check(dataset)
        dist.barrier()
        logger.info(f"Data sanity check passed: {data_sanity_check_passed}")
        data_sanity_check_passed_tensor = torch.tensor(
            1 if data_sanity_check_passed else 0, device=dist_utils.get_device()
        )
        dist.all_reduce(
            data_sanity_check_passed_tensor,
            op=ReduceOp.MIN
        )
        if data_sanity_check_passed_tensor.item() == 0:
            raise RuntimeError("Dataset sanity check failed. See logs for details.")

        dataloader = None
        data_iterator = None
        num_microbatches = (
            args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())
        )
        self.num_microbatches = num_microbatches
        if self._require_dataloader():
            logger.info("creating dataloader")
            sampler = ResumableDistributedSampler(
                dataset,
                num_replicas=mpu.get_data_parallel_world_size(),
                rank=mpu.get_data_parallel_rank(),
                shuffle=args.sft_data_shuffle,
                resume_step=passed_iters,
                micro_batch_size=args.micro_batch_size,
                num_microbatches=num_microbatches,
            )
            dataloader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=args.micro_batch_size,
                drop_last=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=self._collate_packing if args.sequence_packing else self._collate_non_packing,
            )
            logger.info("Dataloader created")
            data_iterator = _InfiniteDataIterator(dataloader, sampler)
        dist.barrier()
        self.dataloader = dataloader
        self.data_iterator = data_iterator
        
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.passed_iters = passed_iters
        self.passed_iters_this_run = 0

        self._num_flos = 0
        self.s1 = torch.cuda.Stream()

    def _require_dataloader(self):
        res = False
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            if mpu.get_tensor_and_context_parallel_rank() == 0:
                res = True
        return res

    def __dataset_sainity_check(self, dataset):
        # 要求input_ids和labels错1位
        # 若启用packing，数据集应包含input_ids,labels,cu_seqlens
        # 若不启用，数据集应包含input_ids,labels
        # 允许数据集有其他字段，框架不使用

        ds_column_names = set(dataset.column_names)
        required_columns = None
        if self.args.sequence_packing:
            required_columns = {"input_ids", "labels", "cu_seqlens"}
        else:
            required_columns = {"input_ids", "labels"}
        if not required_columns.issubset(ds_column_names):
            missing = required_columns - ds_column_names
            self.logger.error(f"Missing required columns in dataset: {missing}")
            return False
        return True

    def all_reduce_in_pp_group(self):
        # very import and necesarry
        tensor = torch.zeros((1024,), dtype=torch.bfloat16, device=dist_utils.get_device())
        dist.all_reduce(tensor, group=mpu.get_pipeline_model_parallel_group())

    def train(self) -> None:
        args = self.args
        config = get_model_config(self.model[0])
        config.grad_scale_func = self.optim.scale_loss
        config.finalize_model_grads_func = finalize_model_grads
        if isinstance(self.model[0], DDP) and args.overlap_grad_reduce:
            config.no_sync_func = [chunk.no_sync for chunk in self.model]
            if len(self.model) == 1:
                config.no_sync_func = config.no_sync_func[0]
        if args.overlap_param_gather and args.align_param_gather:
            config.param_sync_func = [chunk.start_param_sync for chunk in self.model]
            if len(self.model) == 1:
                config.param_sync_func = config.param_sync_func[0]

        if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
            enable_forward_pre_hook(self.model)
        dist.barrier()
        iteration = self.passed_iters
        self.logger.info("warming up pp group comm")
        self.all_reduce_in_pp_group()
        dist.barrier()

        self.logger.info("start training")
        try:
            while iteration < args.train_iters:
                self._train_step(self.data_iterator, self.num_microbatches, iter_num=iteration + 1)
                iteration += 1
                self.passed_iters_this_run += 1
                self._maybe_save_checkpoint(iteration)
                if self.args.early_stop_steps is not None and self.args.early_stop_steps > 0:
                    if self.passed_iters_this_run == self.args.early_stop_steps:
                        break
        finally:
            if self.args.use_distributed_optimizer and self.args.overlap_param_gather:
                disable_forward_pre_hook(self.model)
        dist.barrier()

    # ---------------------------------------------------------------------
    # Data utilities
    # ---------------------------------------------------------------------
    def _validate_dataset_schema(self) -> None:
        column_names = set(self.dataset.column_names)
        required = {self.input_key, self.label_key, self.loss_mask_key}
        missing = required - column_names
        if missing:
            raise RuntimeError(f"Dataset missing required columns: {sorted(missing)}")
        if self.enable_packing:
            if self.packed_param_column is None:
                packed_missing = [col for col in self.required_packed_cols if col not in column_names]
                if packed_missing:
                    raise RuntimeError(
                        "Packed mode requires either --sft-packed-param-column or the following columns: "
                        + ", ".join(packed_missing)
                    )
            elif self.packed_param_column not in column_names:
                raise RuntimeError(
                    f"Packed metadata column '{self.packed_param_column}' not found in dataset columns"
                )

    def _collate_non_packing(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 先pad input_ids和labels
        pad_token_id = self.tokenizer.pad_token_id
        max_length = max(len(s["input_ids"]) for s in samples)
        target_length = max_length
        max_allowed_length = self.args.seq_length

        if max_length > max_allowed_length:
            target_length = max_allowed_length

        def _lcm(a: int, b: int) -> int:
            if a == 1:
                return b
            if b == 1:
                return a
            return abs(a * b) // math.gcd(a, b)

        required_multiple = 1
        if self.args.sequence_parallel:
            tp_world_size = mpu.get_tensor_model_parallel_world_size()
            if tp_world_size > 1:
                required_multiple = _lcm(required_multiple, tp_world_size)

        context_parallel_size = self.args.context_parallel_size
        if context_parallel_size > 1:
            # Megatron要求序列长度是2*CP大小的倍数，见training/arguments.py。
            required_multiple = _lcm(required_multiple, context_parallel_size * 2)

        if required_multiple > 1:
            target_length = _ceil_to_nearest(target_length, required_multiple)

        assert target_length <= max_allowed_length, \
            f"Cannot pad to target length {target_length} which exceeds max allowed {max_allowed_length}"

        max_length = target_length
        input_ids_list = [s["input_ids"][:max_length] for s in samples]
        for i, input_ids in enumerate(input_ids_list):
            padding_length = max_length - len(input_ids)
            input_ids.extend([pad_token_id] * padding_length)
        labels_list = [s["labels"][:max_length] for s in samples]
        for i, labels in enumerate(labels_list):
            padding_length = max_length - len(labels)
            labels.extend([self.args.ignore_token_id] * padding_length)

        # 构建batch张量
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.int64)
        labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

        # 构建loss_mask张量
        loss_mask_tensor = labels_tensor.ne(self.args.ignore_token_id).float()

        # 复原labels
        labels_tensor[labels_tensor == self.args.ignore_token_id] = pad_token_id

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "loss_mask": loss_mask_tensor,
        }

    def _collate_packing(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_tensor = torch.stack(
            [torch.tensor(s["input_ids"], dtype=torch.int64) for s in samples], dim=0
        )
        labels_tensor = torch.stack(
            [torch.tensor(s["labels"], dtype=torch.int64) for s in samples], dim=0
        )

        # 构建loss_mask张量
        loss_mask_tensor = labels_tensor.ne(self.args.ignore_token_id).float()

        # 复原labels
        labels_tensor[labels_tensor == self.args.ignore_token_id] = self.tokenizer.pad_token_id

        cu_seqlens_tensor = torch.tensor(
            samples[0]["cu_seqlens"], dtype=torch.int32
        )

        max_seqlen = 0
        for i in range(cu_seqlens_tensor.shape[0] - 1):
            if cu_seqlens_tensor[i + 1] - cu_seqlens_tensor[i] > max_seqlen:
                max_seqlen = cu_seqlens_tensor[i + 1] - cu_seqlens_tensor[i]
        # max_seqlen = self.args.seq_length
        max_seqlen_tensor = max_seqlen.int()
        # max_seqlen_tensor = torch.tensor(self.args.seq_length, dtype=torch.int32)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "loss_mask": loss_mask_tensor,
            "cu_seqlens": cu_seqlens_tensor,
            "max_seqlen": max_seqlen_tensor,
        }
    
    def _sync_input_data(self, data: torch.Tensor, shape, dtype, group, src_rank) -> torch.Tensor:
        # 当前数据只在src_rank上有，其他rank需要同步
        if dist.get_rank() != src_rank:
            assert data is None, "data must be None on non-src ranks"
            data = torch.empty(shape, dtype=dtype, device=dist_utils.get_device())
        else:
            assert data is not None, "data must not be None on src rank"
        dist.broadcast(data, src=src_rank, group=group)
        return data
    
    def _sync_packed_seq_params(self, cu_seqlens: Optional[torch.Tensor], max_seqlen: Optional[torch.Tensor], group, src_rank) -> PackedSeqParams:
        if dist.get_rank() != src_rank:
            assert cu_seqlens is None, "cu_seqlens must be None on non-src ranks"
            assert max_seqlen is None, "max_seqlen must be None on non-src ranks"
            length_tensor = torch.empty(1, dtype=torch.int32, device=dist_utils.get_device())
        else:
            assert cu_seqlens is not None, "cu_seqlens must not be None on src rank"
            assert max_seqlen is not None, "max_seqlen must not be None on src rank"
            length_tensor = torch.tensor(cu_seqlens.shape[0], dtype=torch.int32, device=dist_utils.get_device())
        dist.broadcast(length_tensor, src=src_rank, group=group)
        length = length_tensor.item()
        if dist.get_rank() != src_rank:
            cu_seqlens = torch.empty(length, dtype=torch.int32, device=dist_utils.get_device())
            max_seqlen = torch.empty(1, dtype=torch.int32, device=dist_utils.get_device())
        dist.broadcast(cu_seqlens, src=src_rank, group=group)
        dist.broadcast(max_seqlen, src=src_rank, group=group)
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen.item(),
            max_seqlen_kv=max_seqlen.item(),
        )

    def _broadcast_packed_seq_params_to_pipeline(
        self, packed_seq_params: Optional[PackedSeqParams]
    ) -> Optional[PackedSeqParams]:
        if not self.args.sequence_packing:
            return None
        pipeline_group = mpu.get_pipeline_model_parallel_group()
        if pipeline_group is None or mpu.get_pipeline_model_parallel_world_size() == 1:
            return packed_seq_params
        pipeline_src_rank = mpu.get_pipeline_model_parallel_first_rank()
        cu_src = None
        max_src = None
        if dist.get_rank() == pipeline_src_rank and packed_seq_params is not None:
            cu_src = packed_seq_params.cu_seqlens_q
            max_src = torch.tensor(packed_seq_params.max_seqlen_q, dtype=torch.int32, device=dist_utils.get_device())
        return self._sync_packed_seq_params(cu_src, max_src, pipeline_group, pipeline_src_rank)

    def _prepare_microbatch_payload(
        self,
        data_iter: Optional[Iterator],
        iter_num: int,
        microbatch_idx: int,
    ) -> Dict[str, Any]:
        input_ids = None
        labels = None
        loss_mask = None
        cu_seqlens = None
        max_seqlen = None
        packed_seq_params = None
        shape = None

        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            with torch.no_grad():
                if data_iter is not None:
                    batch = next(data_iter)
                    device = dist_utils.get_device()
                    # with torch.cuda.stream(self.s1):
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    loss_mask = batch["loss_mask"].to(device, non_blocking=True)
                    if self.args.sequence_packing:
                        cu_seqlens = batch["cu_seqlens"].to(device, non_blocking=True)
                        max_seqlen = batch["max_seqlen"].to(device, non_blocking=True)
                    shape = input_ids.shape if input_ids is not None else None

                shape_tensor = torch.empty((2,), dtype=torch.int32, device=dist_utils.get_device())
                if shape is not None:
                    shape_tensor[0] = shape[0]
                    shape_tensor[1] = shape[1]

                if mpu.get_context_parallel_rank() == 0:
                    dist.broadcast(
                        shape_tensor,
                        src=mpu.get_tensor_model_parallel_src_rank(),
                        group=mpu.get_tensor_model_parallel_group(),
                    )
                    shape = (shape_tensor[0].item(), shape_tensor[1].item())
                    self.logger.debug(
                        f"iter {iter_num} | microbatch {microbatch_idx} | synchronized input shape: {shape}"
                    )

                    input_ids = self._sync_input_data(
                        input_ids,
                        shape,
                        torch.int64,
                        mpu.get_tensor_model_parallel_group(),
                        mpu.get_tensor_model_parallel_src_rank(),
                    )
                    labels = self._sync_input_data(
                        labels,
                        shape,
                        torch.int64,
                        mpu.get_tensor_model_parallel_group(),
                        mpu.get_tensor_model_parallel_src_rank(),
                    )
                    loss_mask = self._sync_input_data(
                        loss_mask,
                        shape,
                        torch.float32,
                        mpu.get_tensor_model_parallel_group(),
                        mpu.get_tensor_model_parallel_src_rank(),
                    )
                    if self.args.sequence_packing:
                        packed_seq_params = self._sync_packed_seq_params(
                            cu_seqlens,
                            max_seqlen,
                            mpu.get_tensor_model_parallel_group(),
                            mpu.get_tensor_model_parallel_src_rank(),
                        )

            if mpu.get_context_parallel_world_size() > 1:
                raise NotImplementedError("context parallelism is not supported yet")

        if self.args.sequence_packing:
            packed_seq_params = self._broadcast_packed_seq_params_to_pipeline(packed_seq_params)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "packed_seq_params": packed_seq_params,
        }

    def _prepare_microbatch_payloads(
        self, data_iterator: Optional[Iterator], num_microbatches: int, iter_num: int
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for microbatch_idx in range(num_microbatches):
            payloads.append(self._prepare_microbatch_payload(data_iterator, iter_num, microbatch_idx))
        return payloads

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _train_step(self, data_iterator: Iterator, num_microbatches: int, iter_num: int) -> None:
        for chunk in self.model:
            chunk.zero_grad_buffer()
        self.optim.zero_grad()

        prepared_microbatches = self._prepare_microbatch_payloads(data_iterator, num_microbatches, iter_num)
        prepared_microbatch_iter = iter(prepared_microbatches)

        def loss_func(output_tensor: torch.Tensor, loss_mask: torch.Tensor):
            mask = loss_mask.float().view(-1)
            per_token = output_tensor.float().view(-1)
            num_valid_tokens = mask.sum()
            loss = torch.sum(per_token * mask)
            if self.args.context_parallel_size > 1:
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            
            # handle loss report
            loss_report = torch.cat([loss.clone().detach().unsqueeze(0), num_valid_tokens.clone().detach().unsqueeze(0)], dim=0)
            dist.all_reduce(loss_report, group=mpu.get_data_parallel_group())
            return loss * mpu.get_context_parallel_world_size(), num_valid_tokens.clone().detach().to(torch.int), {"lm_loss": loss_report}

        def loss_func_batch_mean(output_tensor: torch.Tensor, loss_mask: torch.Tensor):
            mask = loss_mask.float().view(-1)
            per_token = output_tensor.float().view(-1)
            num_valid_tokens = mask.sum()
            # assert num_valid_tokens.item() > 0, "num_valid_tokens is 0"
            loss = torch.sum(per_token * mask)
            if self.args.context_parallel_size > 1:
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss = loss / (num_valid_tokens + 1e-8)
            
            # handle loss report
            loss_report = torch.cat([loss.clone().detach().unsqueeze(0),], dim=0)
            dist.all_reduce(loss_report, group=mpu.get_data_parallel_group(), op=ReduceOp.AVG)
            return loss * mpu.get_context_parallel_world_size(), {"lm_loss": loss_report}

        def forward_step(data_iter, model):
            input_ids = None
            labels = None
            loss_mask = None
            cu_seqlens = None
            max_seqlen = None
            packed_seq_params = None

            shape = None

            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                with torch.no_grad():
                    if data_iter is not None:
                        h2d_time1 = time.time()
                        batch = next(data_iter)
                        # async h2d transfer
                        device = dist_utils.get_device()
                        with torch.cuda.stream(self.s1):
                            input_ids = batch["input_ids"].to(device, non_blocking=True)
                            labels = batch["labels"].to(device, non_blocking=True)
                            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
                            if self.args.sequence_packing:
                                cu_seqlens = batch["cu_seqlens"].to(device, non_blocking=True)
                                max_seqlen = batch["max_seqlen"].to(device, non_blocking=True)
                        h2d_time2 = time.time()
                        self.logger.debug(f"h2d time: {h2d_time2 - h2d_time1}")
                        shape = input_ids.shape

                    data_broadcast_time1 = time.time()
                    shape_tensor = torch.empty((2,), dtype=torch.int32, device=dist_utils.get_device())
                    if shape is not None:
                        shape_tensor[0] = shape[0]
                        shape_tensor[1] = shape[1]            
                    
                    if mpu.get_context_parallel_rank() == 0:
                        dist.broadcast(shape_tensor, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
                        shape = (shape_tensor[0].item(), shape_tensor[1].item())
                        self.logger.debug(f"iter {iter_num} | synchronized input shape: {shape}")

                        input_ids = self._sync_input_data(input_ids, shape, torch.int64, mpu.get_tensor_model_parallel_group(), mpu.get_tensor_model_parallel_src_rank())
                        labels = self._sync_input_data(labels, shape, torch.int64, mpu.get_tensor_model_parallel_group(), mpu.get_tensor_model_parallel_src_rank())
                        loss_mask = self._sync_input_data(loss_mask, shape, torch.float32, mpu.get_tensor_model_parallel_group(), mpu.get_tensor_model_parallel_src_rank())
                        if self.args.sequence_packing:
                            packed_seq_params = self._sync_packed_seq_params(
                                cu_seqlens, max_seqlen, mpu.get_tensor_model_parallel_group(), mpu.get_tensor_model_parallel_src_rank()
                            )
                    data_broadcast_time2 = time.time()
                    self.logger.debug(f"data_broadcast_time: {data_broadcast_time2 - data_broadcast_time1}")
                if mpu.get_context_parallel_world_size() > 1:
                    raise NotImplementedError("context parallelism is not supported yet")

            forward_args = {
                "input_ids": input_ids,
                "labels": labels, 
                "position_ids": None,
                "attention_mask": None,
                "packed_seq_params": packed_seq_params,
            }
            output_tensor = model(**forward_args)
            return output_tensor, partial(loss_func_batch_mean, loss_mask=loss_mask)

        def forward_step_v2(data_iter, model):
            try:
                batch = next(data_iter)
            except StopIteration:
                raise RuntimeError("prepared microbatch iterator exhausted early")

            forward_args = {
                "input_ids": batch["input_ids"],
                "labels": batch["labels"],
                "position_ids": None,
                "attention_mask": None,
                "packed_seq_params": batch["packed_seq_params"],
            }
            output_tensor = model(**forward_args)
            return output_tensor, partial(loss_func_batch_mean, loss_mask=batch["loss_mask"])

        step_time1 = time.time()
        reduced_losses = get_forward_backward_func()(
            forward_step_func=forward_step_v2,
            data_iterator=prepared_microbatch_iter,
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=False,
            seq_length=self.args.seq_length,
            micro_batch_size=self.args.micro_batch_size,
        )

        update_successful, grad_norm, _ = self.optim.step()
        if update_successful:
            self.scheduler.step(increment=self.args.global_batch_size)
        else:
            self.logger.warning("optimizer step skipped due to overflow")
        lr = self._current_lr()

        if reduced_losses:
            stacked = torch.stack([
                item["lm_loss"] for item in reduced_losses
            ], dim=0)
            # stacked = stacked.sum(dim=0)
            # loss_value = (stacked[0] / stacked[1]).item()

            # batch mean
            loss_value = stacked.mean(dim=0)[0].item()
        else:
            loss_value = 0.0

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # send loss to rank 0
            if mpu.get_tensor_and_context_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():
                    loss_buffer = torch.empty(1, device=dist_utils.get_device())
                    dist.recv(loss_buffer, src=mpu.get_pipeline_model_parallel_last_rank())
                    loss_value = loss_buffer.item()
                elif mpu.is_pipeline_last_stage():
                    loss_buffer = torch.tensor(loss_value, device=dist_utils.get_device())
                    dist.send(loss_buffer, dst=mpu.get_pipeline_model_parallel_first_rank())
        dist.barrier()
        step_time2 = time.time()

        if self.args.num_experts is not None:
            moe_loss_scale = 1 / self.num_microbatches
            track_names = []
            if self.args.moe_router_load_balancing_type in ["aux_loss", "seq_aux_loss"]:
                track_names.append("load_balancing_loss")
            if self.args.moe_z_loss_coeff is not None:
                track_names.append("z_loss")
            total_loss_dict = {k: torch.tensor([0.0], dtype=torch.float, device=dist_utils.get_device()) for k in track_names}
            track_moe_metrics(
                loss_scale=moe_loss_scale,
                iteration=iter_num,
                writer=self.tensorboard_writer,
                wandb_writer=None,
                total_loss_dict=total_loss_dict,
                per_layer_logging=self.args.moe_per_layer_logging,
                force_initialize=True,
                track_names=track_names,
                num_layers=self.args.num_layers,
                moe_layer_freq=self.args.moe_layer_freq,
                mtp_num_layers=self.args.mtp_num_layers,
            )
            self.logger.info(f"iter {iter_num} | moe metrics: {total_loss_dict}")

        self.logger.info(
            "iter %d / %d | loss %.6f | grad_norm %.4f | lr %.6e | step time(s) %d",
            iter_num,
            self.args.train_iters,
            loss_value,
            grad_norm,
            lr,
            step_time2 - step_time1
        )
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("train/loss", loss_value, iter_num)
            self.tensorboard_writer.add_scalar("train/grad_norm", grad_norm, iter_num)
            self.tensorboard_writer.add_scalar("train/lr", lr, iter_num)
            self.tensorboard_writer.flush()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _maybe_save_checkpoint(self, iteration: int) -> None:
        args = self.args
        if args.save is not None and args.save_interval and iteration % args.save_interval == 0:
            self._save_checkpoint(iteration, non_persistent=False)
            return
        if (
            args.non_persistent_save_interval
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

    def _maybe_get_tensorboard_writer(self):
        try:
            return get_tensorboard_writer()
        except Exception:
            return None

    def _current_lr(self) -> float:
        lr = None
        for param_group in self.optim.param_groups:
            if param_group['is_decoupled_lr']:
                lr = param_group['lr']
            else:
                lr = param_group['lr']
        return lr
