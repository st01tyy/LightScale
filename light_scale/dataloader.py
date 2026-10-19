# 每一个进程都应该创建并执行dataloader，与进程相关的判断逻辑在内部执行，外部无需处理
# 并非torch Dataloader，实现python iterator即可

import torch.distributed as dist
from megatron.core import mpu
from datasets import load_from_disk
from light_scale.config import ActorInferenceServiceConfig, ReferenceModelConfig
from light_scale.llm_caller import ActorInferenceServiceCaller, ReferenceModelCaller
from typing import List, Tuple, Optional
from light_scale.collator import Collator
import light_scale.dist_utils as dist_utils
import torch
from light_scale.fastapi_server import FastAPIServer
from light_scale.logger_utils import setup_logger

INPUT_KEYS = [
    "input_ids",
    "labels",
    "loss_mask"
]

def _sync_input_data(input_data: Optional[torch.Tensor], dtype: torch.dtype):
    # sync the 2-D input data, shape should be: batch_size, seq_length
    # only the first and last pp stage enter this function
    # TODO: what if only one pp stage?
    if dist.get_rank() == 0:
        assert len(input_data.shape) == 2
    
    # first, we need to get the shape
    shape_tensor = torch.zeros((2,), dtype=torch.int32, device=dist_utils.get_device())

    if dist.get_rank() == 0:
        shape_tensor[0] = input_data.shape[0]
        shape_tensor[1] = input_data.shape[1]
    
    # rank 0 sends shape to pp[-1]dp[0]cp[0]tp[0]
    if mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
        if mpu.is_pipeline_first_stage():
            dist.send(shape_tensor, dst=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        elif mpu.is_pipeline_last_stage():
            dist.recv(shape_tensor, src=mpu.get_pipeline_model_parallel_first_rank(), group=mpu.get_pipeline_model_parallel_group())
    dist_utils.wait_for_dp_and_cp_and_tp_neighbors()

    # broadcast shape in first and last pp stage
    if mpu.is_pipeline_first_stage():
        dist_utils.broadcast_in_pp_stage(shape_tensor)
    if mpu.is_pipeline_last_stage():
        dist_utils.broadcast_in_pp_stage(shape_tensor)
    dist_utils.wait_for_dp_and_cp_and_tp_neighbors()

    if mpu.is_pipeline_first_stage() and mpu.is_pipeline_last_stage():
        # make sure everyone got the shape
        assert shape_tensor.sum().item() > 0
    
    if input_data is None:
        # every rank except for rank 0 should be none
        input_data = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=dtype, device=dist_utils.get_device())

    # rank 0 sends data to pp[-1]dp[0]cp[0]tp[0]
    if mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
        if mpu.is_pipeline_first_stage():
            dist.send(input_data, dst=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        elif mpu.is_pipeline_last_stage():
            dist.recv(input_data, src=mpu.get_pipeline_model_parallel_first_rank(), group=mpu.get_pipeline_model_parallel_group())
    dist_utils.wait_for_dp_and_cp_and_tp_neighbors()

    # broadcast data in first and last pp stage
    if mpu.is_pipeline_first_stage():
        dist_utils.broadcast_in_pp_stage(input_data)
    if mpu.is_pipeline_last_stage():
        dist_utils.broadcast_in_pp_stage(input_data)
    dist_utils.wait_for_dp_and_cp_and_tp_neighbors()

    return input_data

class RolloutDataloader:
    def __init__(self, hf_dataset_path: str, rollout_batch_size: int, passed_iters: int):
        logger = setup_logger("light_scale")
        self.rollout_batch_size = rollout_batch_size
        self.passed_iters = passed_iters

        logger.info(f"loading from {hf_dataset_path}")
        self.samples = load_from_disk(hf_dataset_path)
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
        dataset_type = "default" # for debug
        return {
            "prompt": sample["prompt"],
            "ground_truth": sample["ground_truth"],
            "dataset_type": dataset_type
        }

    def __next__(self):
        # 读取rollout batch个sample
        samples = [self._read_single_sample() for _ in range(self.rollout_batch_size)]

        # 预处理得到prompts
        prompts = [self._process_single_sample(sample) for sample in samples]

        return prompts

class DistributedDataloader:
    def __init__(self):
        assert dist.is_initialized() and mpu.is_initialized()

class ActorDataloader(DistributedDataloader):
    # actor模型的dataloader
    # pp[0]tp[0]负责读取数据，预处理，调用推理服务
    # pp[0]tp[0]负责获取推理结果，结果发送至pp[-1]tp[0]，处理模型输入
    # pp[-1]tp[0]将推理结果异步发送至ref model，异步获取结果
    # pp[0|-1]tp[0]广播至pp[0|-1][1:]
    def __init__(self, 
                 hf_dataset_path: str,
                 micro_batch_size: int, 
                 global_batch_size: int,
                 max_iters: int, 
                 passed_iters: int,
                 inference_service_config: Optional[ActorInferenceServiceConfig] = None,
                 reference_model_config: Optional[ReferenceModelConfig] = None,
                 collator: Optional[Collator] = None):
        super().__init__()
        assert passed_iters < max_iters

        self.samples = None
        self.cur_id = None
        self.passed_iters = passed_iters
        self.max_iters = max_iters
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.mbs_dp = micro_batch_size * mpu.get_data_parallel_world_size()
        assert self.global_batch_size % self.mbs_dp == 0
        self.inference_service_config = inference_service_config

        self.actor_inference_caller: ActorInferenceServiceCaller = None
        self.collator = None

        self.reference_model_caller: ReferenceModelCaller = None
        

        if dist.get_rank() == 0:
            # 加载数据集
            print(f"loading from {hf_dataset_path}")
            self.samples = load_from_disk(hf_dataset_path)
            assert len(self.samples) > 0
            self.cur_id = (self.passed_iters * self.mbs_dp + len(self.samples)) % len(self.samples)

            # 初始化推理服务调用
            self.actor_inference_caller = ActorInferenceServiceCaller(
                inference_service_config.actor_service_url,
                inference_service_config.model_name,
                inference_service_config.num_workers
            )

            assert collator is not None
            self.collator = collator

        if mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
            assert reference_model_config is not None
            self.reference_model_caller = ReferenceModelCaller(reference_model_config.reference_service_url)
        
        self.logger = setup_logger("light_scale")
        dist.barrier()

    def _read_single_sample(self):
        # 只有rank0进入这个函数
        # 读取一个sample，更新cur_id
        if self.cur_id == len(self.samples):
            self.cur_id = 0
        sample = self.samples[self.cur_id]
        self.cur_id += 1
        return sample
    
    def _process_single_sample(self, sample):
        # 只有rank0进入这个函数
        # 预处理一条sample
        # for debug: 直接读取sample中的prompt字段
        assert "prompt" in sample
        return sample["prompt"]
    
    def _read_preprocess_call(self) -> Tuple[List[List[str]], List[int]]:
        # 只有rank0进入这个函数
        # 读取、预处理、调用推理服务

        # 读取global batch个sample
        samples = [self._read_single_sample() for _ in range(self.global_batch_size)]

        # 预处理得到prompts
        prompts = [self._process_single_sample(sample) for sample in samples]
        results = self.actor_inference_caller.batch_completions(
            prompts=prompts,
            n_samples=self.inference_service_config.n_samples,
            max_tokens=self.inference_service_config.max_tokens,
            temperature=self.inference_service_config.temperature
        )

        responses_list = [result[0] for result in results]
        completion_tokens = [result[1] for result in results]

        processed_samples = []
        for prompt, n_responses in zip(prompts, responses_list):
            processed_samples.append((prompt, n_responses))
        
        assert len(processed_samples) == self.global_batch_size, f"{len(processed_samples)}"
        assert len(completion_tokens) == self.global_batch_size, f"{len(completion_tokens)}"
        
        return processed_samples, completion_tokens

    def __next__(self):
        if self.passed_iters == self.max_iters:
            raise StopIteration
        # 读取global batch个sample
        processed_samples = None
        completion_tokens = None
        if dist.get_rank() == 0:
            processed_samples, completion_tokens = self._read_preprocess_call()
            self.logger.info(completion_tokens)
        dist.barrier()
        if dist.get_rank() == 0:
            self.logger.info("sending processed_samples")
            dist.send_object_list([processed_samples, completion_tokens], mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        elif mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
            objects = [None, None]
            self.logger.info("receiving processed_samples")
            dist.recv_object_list(objects, src=0, group=mpu.get_pipeline_model_parallel_group())
            processed_samples, completion_tokens = objects
        dist.barrier()

        # 将每个micro batch分批给ref logp发送请求
        ref_logp_futures = []
        # num_micro_batches = self.global_batch_size // self.mbs_dp
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
            self.logger.info("sending ref logp request")
            for i in range(self.global_batch_size // self.mbs_dp):
                start = i * self.mbs_dp
                end = start + self.mbs_dp
                micro_batch_dp_samples = processed_samples[start:end]
                ref_logp_future = self.reference_model_caller.async_batch_logp(micro_batch_dp_samples)
                ref_logp_futures.append(ref_logp_future)
        dist.barrier()

        batch_inputs = None
        if dist.get_rank() == 0:
            self.logger.info("calling collate_fn")
            batch_inputs = self.collator.collate_fn(processed_samples)
            self.logger.info(batch_inputs.keys())
        
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            if batch_inputs is None:
                batch_inputs = dict()
            for input_key in INPUT_KEYS:
                if dist.get_rank() == 0:
                    self.logger.info(f"sending input data: {input_key}")
                    _sync_input_data(batch_inputs[input_key], dtype=torch.int64)
                else:
                    self.logger.info(f"receiving input data {input_key}")
                    batch_inputs[input_key] = _sync_input_data(None, torch.int64)
            batch_inputs["ref_logp_futures"] = ref_logp_futures

        dist.barrier()
        return batch_inputs
    
    def __iter__(self):
        return self

class ReferenceServingDataloader(DistributedDataloader):
    def __init__(self, server: Optional[FastAPIServer] = None, collator: Optional[Collator] = None):
        super().__init__()

        self.server = None
        self.collator = None
        if dist.get_rank() == 0:
            assert server is not None
            assert collator is not None

            self.server = server
            self.collator = collator
        self.logger = setup_logger("light_scale")
    
    def _wait_and_read_input_samples(self) -> list:
        # 只有rank 0进入该函数，等待request数据传入并处理
        samples = self.server.read_message()  # actor pp[-1]tp[0]发送的processed_samples，should be same as #198
        return samples
    
    def __next__(self):
        # 返回一个List[dict]作为ref global step，计算ref_ga(micro batch numbers)
        # reference model的每一步负责计算mbs*n*dp个样本，此处有多种方法
        # ref_ga如何决定？N = mbs*n*dp, N % ref_dp == 0; ref_Nd = N // ref_dp; ref_mbs * ref_ga = ref_Nd；
        # TODO: 如何动态决定ref_mbs，ref_ga？
        # TODO: 如何决定动态长度？
        # TODO: 如何动态放弃被截断的样本？全-100？额外的valid_mask？
        # 目前方案：固定ref_mbs为1，ref_dp已知，ref_ga = N // ref_dp，不涉及padding，超长样本截断不废弃
        samples = None
        num_micro_batches = None
        if dist.get_rank() == 0:
            self.logger.info("rank 0 waiting for input")
            samples = self._wait_and_read_input_samples()
            self.logger.info("rank 0 received input")
            self.logger.info(samples)
            N = len(samples) * len(samples[0][1])
            assert N % mpu.get_data_parallel_world_size() == 0
            num_micro_batches = N // mpu.get_data_parallel_world_size()
            self.logger.info(f"num_micro_batches: {num_micro_batches}")
        self.logger.info("waiting for rank 0 receive input")
        dist.barrier()
        
        batch_inputs = None
        if dist.get_rank() == 0:
            self.logger.info("rank 0 running collate_fn")
            batch_inputs = self.collator.collate_fn(samples)
            batch_inputs["num_micro_batches"] = torch.tensor([[num_micro_batches]], dtype=torch.int64, device=dist_utils.get_device())
            self.logger.info("rank 0 batch_inputs keys:")
            self.logger.info(batch_inputs.keys())
        
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            if batch_inputs is None:
                batch_inputs = dict()
            for input_key in INPUT_KEYS + ["num_micro_batches"]:
                self.logger.info(f"syncthing input data: {input_key}")
                if dist.get_rank() == 0:
                    _sync_input_data(batch_inputs[input_key], dtype=torch.int64)
                else:
                    batch_inputs[input_key] = _sync_input_data(None, torch.int64)

        dist.barrier()
        self.logger.info("dataloader next() done")
        return batch_inputs
    
    def __iter__(self):
        return self
