from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from light_scale import dist_utils
import requests
from concurrent.futures import ThreadPoolExecutor
import os
import datetime
import torch
import torch.distributed as dist
from light_scale.logger_utils import setup_logger
from typing import List, Optional

from light_scale.distributed_lock import DistributedLock

class SafetensorsSaver:
    def __init__(self, file_name: str, dist_lock: DistributedLock):
        self.logger = setup_logger("light_scale")
        self.state_dict = dict()
        self.file_name = file_name
        self.part = 1
        self.dist_lock = dist_lock

    def add(self, weight_name: str, weight: Tensor):
        self.state_dict[weight_name] = weight.detach().cpu()
        self.logger.debug(f"added {weight_name}")
    
    def commit(self):
        with self.dist_lock:
            save_file(self.state_dict, f"{self.file_name}_part_{self.part:03d}.safetensors", metadata={"format": "pt"})
        self.part += 1
        self.state_dict = dict()

class SGLangSaver:
    def __init__(self, 
                 server_base_url_list: List[str],
                 server_world_size: int,
                 pp_rank: int,
                 dist_lock: DistributedLock,
                 update_group_port: int = 65500):
        self.logger = setup_logger("light_scale")
        self.pp_rank = pp_rank
        self.dist_lock = dist_lock
        self.group_name = f"light_scale_parameter_update_group_pp_{pp_rank}"
        self.logger.info(f"SGLangSaver is initialized on device {torch.cuda.current_device()}")
        self.state_dict = dict()
        self.state_keys = []
        self.state_dtypes = []
        self.state_shapes = []
        self.dtype_str_dict = {
            torch.float32: "float32",
            torch.bfloat16: "bfloat16",
            torch.float16: "float16"
        }
        self.server_base_url_list = server_base_url_list
        self.server_world_size = server_world_size
        assert server_world_size % len(server_base_url_list) == 0
        self.num_gpus_per_service = server_world_size // len(server_base_url_list)
        self.update_group_port = update_group_port
        master_addr = dist_utils.get_hostname()
        if master_addr is None:
            raise RuntimeError("master addr not found")
        self.logger.debug(f"master addr: {master_addr}")
        master_port = os.environ.get("MASTER_PORT")
        if master_port is None:
            raise RuntimeError("master port not found")
        self.master_port = int(master_port)
        self.logger.debug(f"master port: {self.master_port}")
        self.master_addr = master_addr
        self.thread_pool = ThreadPoolExecutor(len(server_base_url_list))
        with self.dist_lock:
            self.update_group = self._init_update_group()

    def _init_update_group(self):
        futures = []
        for i, server_base_url in enumerate(self.server_base_url_list):
            future = self.thread_pool.submit(self._init_update_group_remote, server_base_url, i)
            futures.append(future)
        group = self._init_update_group_local()
        self.logger.info("local initialized, waiting for remote")
        dist.barrier(group)
        for future in futures:
            response = future.result()
            self.logger.debug(response.text)
            if response.status_code != 200:
                self.close()
                raise RuntimeError(f"status code: {response.status_code}")
        return group

    def _init_update_group_local(self):
        self.logger.info("initializing update group from local")
        group = dist_utils.init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_addr}:{self.update_group_port}",
            world_size=self.server_world_size + 1,
            rank=0,
            group_name=self.group_name,
            timeout=datetime.timedelta(seconds=300)
        )
        return group
    
    def _init_update_group_remote(self, server_base_url: str, service_offset: int):
        self.logger.info("initializing update group from remote")
        response = requests.post(
            f"{server_base_url}/init_weights_update_group",
            json={
                "master_address": self.master_addr,
                "master_port": self.update_group_port,
                "rank_offset": 1 + service_offset * self.num_gpus_per_service,
                "world_size": self.server_world_size + 1,
                "group_name": self.group_name,
                "backend": "nccl",
            },
            timeout=310
        )
        return response
    
    def close(self):
        self.thread_pool.shutdown()

    def add(self, weight_name: str, weight: Tensor):
        self.state_dict[weight_name] = weight
        self.state_keys.append(weight_name)
        self.state_dtypes.append(self.dtype_str_dict[weight.dtype])
        self.state_shapes.append(weight.shape)
        self.logger.debug(f"added {weight_name}")
    
    def _broadcast_weight_local(self, weight: Tensor):
        torch.distributed.broadcast(
            weight,
            src=0,
            group=self.update_group,
        )

    def _broadcast_weights_local(self):
        for weight_name in self.state_keys:
            weight = self.state_dict[weight_name]
            torch.distributed.broadcast(
                weight,
                src=0,
                group=self.update_group,
            )

    def _broadcast_weight_remote(self, weight_name: str, weight: Tensor, server_base_url: str):
        self.logger.debug(f"remote broadcast {weight_name}")
        # refer to https://github.com/sgl-project/sglang/blob/8cd344586e09669432d35c678b7cc208c0a6f47e/test/srt/test_update_weights_from_distributed.py#L310
        response = requests.post(
            f"{server_base_url}/update_weights_from_distributed",
            json={
                "names": [weight_name],
                "dtypes": [self.dtype_str_dict[weight.dtype]],
                "shapes": [weight.shape],
                "group_name": self.group_name
            },
        )
        return response

    def _broadcast_weights_remote(self, server_base_url: str):
        response = requests.post(
            f"{server_base_url}/update_weights_from_distributed",
            json={
                "names": self.state_keys,
                "dtypes": self.state_dtypes,
                "shapes": self.state_shapes,
                "group_name": self.group_name
            },
        )
        return response

    def _broadcast_weight(self, weight_name: str, weight: Tensor):
        self.logger.debug(f"broadcast {weight_name}")
        futures = []
        for server_base_url in self.server_base_url_list:
            future = self.thread_pool.submit(self._broadcast_weight_remote, weight_name, weight, server_base_url)
            futures.append(future)
        self._broadcast_weight_local(weight)
        self.logger.debug("waiting for remote")
        for future in futures:
            response = future.result()
            self.logger.debug(response.text)
            if response.status_code != 200:
                self.close()
                raise RuntimeError(f"status code: {response.status_code}")

    def _broadcast_weights(self):
        self.logger.debug(f"broadcast_weights, {len(self.state_keys)} weights")
        futures = []
        for server_base_url in self.server_base_url_list:
            future = self.thread_pool.submit(self._broadcast_weights_remote, server_base_url)
            futures.append(future)
        self._broadcast_weights_local()
        self.logger.debug("waiting for remote")
        for future in futures:
            response = future.result()
            self.logger.debug(response.text)
            if response.status_code != 200:
                self.close()
                raise RuntimeError(f"status code: {response.status_code}")
    
    def commit(self):
        with self.dist_lock:
            # for key, weight in self.state_dict.items():
            #     self._broadcast_weight(key, weight)
            self._broadcast_weights()
        self.state_dict = dict()
        self.state_keys = []
        self.state_dtypes = []
        self.state_shapes = []

class ActorReferenceDataUpdater:
    def __init__(self, actor_master_addr: str, update_group_port: int, is_actor: bool, timeout_minutes: int):
        self.logger = setup_logger("light_scale")
        self.actor_master_addr = actor_master_addr
        self.update_group_port = update_group_port
        self.is_actor = is_actor
        self.timeout_minutes = timeout_minutes
        self.update_group = self._init_update_group()

    def _init_update_group(self):
        self.logger.info("initializing actor/reference data update group")
        group = dist_utils.init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{self.actor_master_addr}:{self.update_group_port}",
            world_size=2,
            rank=0 if self.is_actor else 1,
            group_name="actor_reference_update_group",
            timeout=datetime.timedelta(minutes=self.timeout_minutes)
        )
        dist.barrier(group)
        return group
    
    def actor_send_reference_receive_2D_tensor(self, tensor: Tensor, dtype, shape_tensor: Optional[Tensor] = None):
        if self.is_actor:
            assert len(tensor.shape) == 2
        
        if shape_tensor is None:
            # first, we need to get the shape
            shape_tensor = torch.zeros((2,), dtype=torch.int32, device=dist_utils.get_device())

            if dist.get_rank() == 0 and self.is_actor:
                shape_tensor[0] = tensor.shape[0]
                shape_tensor[1] = tensor.shape[1]

            if self.is_actor:
                dist.send(shape_tensor, dst=1, group=self.update_group)
            else:
                dist.recv(shape_tensor, src=0, group=self.update_group)

        if tensor is None:
            tensor = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=dtype, device=dist_utils.get_device())
        
        if self.is_actor:
            dist.send(tensor, 1, self.update_group)
        else:
            dist.recv(tensor, 0, self.update_group)
        
        return tensor
    
    def actor_receive_reference_send_2D_tensor(self, tensor: Tensor, dtype):
        if not self.is_actor:
            assert len(tensor.shape) == 2
        
        # first, we need to get the shape
        shape_tensor = torch.zeros((2,), dtype=torch.int32, device=dist_utils.get_device())

        if dist.get_rank() == 0 and not self.is_actor:
            shape_tensor[0] = tensor.shape[0]
            shape_tensor[1] = tensor.shape[1]

        if not self.is_actor:
            self.update_group.send([shape_tensor], 0, 0).wait()
        else:
            dist.recv(shape_tensor, src=1, group=self.update_group)

        if tensor is None:
            tensor = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=dtype, device=dist_utils.get_device())
        
        if not self.is_actor:
            self.update_group.send([tensor], 0, 0).wait()
        else:
            dist.recv(tensor, 1, self.update_group)
        
        return tensor
    
    def actor_reference_exchange_meta(self, meta: dict):
        assert isinstance(meta, dict)
        
        meta_list = [None, None]

        dist.all_gather_object(meta_list, meta, group=self.update_group)

        return meta_list[0] if not self.is_actor else meta_list[1]