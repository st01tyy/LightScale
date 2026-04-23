import torch.distributed as dist
from megatron.core import mpu
import torch
from typing import Optional, List, Tuple
from torch.distributed import ReduceOp
import socket
import os

def _sync_2D_input_data(input_data: Optional[torch.Tensor], dtype: torch.dtype, shape_tensor: Optional[torch.Tensor] = None):
    # sync the 2-D input data, shape should be: batch_size, seq_length
    # only the first and last pp stage enter this function
    # TODO: what if only one pp stage?
    if dist.get_rank() == 0:
        assert len(input_data.shape) == 2
    
    # first, we need to get the shape
    if shape_tensor is None:
        shape_tensor = torch.zeros((2,), dtype=torch.int32, device=get_device())

        if dist.get_rank() == 0:
            shape_tensor[0] = input_data.shape[0]
            shape_tensor[1] = input_data.shape[1]
        
        # rank 0 sends shape to pp[-1]dp[0]cp[0]tp[0]
        if mpu.get_pipeline_model_parallel_world_size() > 1 and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
            if mpu.is_pipeline_first_stage():
                dist.send(shape_tensor, dst=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
            elif mpu.is_pipeline_last_stage():
                dist.recv(shape_tensor, src=mpu.get_pipeline_model_parallel_first_rank(), group=mpu.get_pipeline_model_parallel_group())
        wait_for_dp_and_cp_and_tp_neighbors()

        # broadcast shape in first and last pp stage
        if mpu.is_pipeline_first_stage():
            broadcast_in_pp_stage(shape_tensor)
        elif mpu.is_pipeline_last_stage():
            broadcast_in_pp_stage(shape_tensor)
        wait_for_dp_and_cp_and_tp_neighbors()

        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            # make sure everyone got the shape
            assert shape_tensor.sum().item() > 0
    
    if input_data is None:
        # every rank except for rank 0 should be none
        input_data = torch.zeros((shape_tensor[0], shape_tensor[1]), dtype=dtype, device=get_device())

    # rank 0 sends data to pp[-1]dp[0]cp[0]tp[0]
    if mpu.get_pipeline_model_parallel_world_size() > 1 and mpu.get_tensor_and_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
        if mpu.is_pipeline_first_stage():
            dist.send(input_data, dst=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        elif mpu.is_pipeline_last_stage():
            dist.recv(input_data, src=mpu.get_pipeline_model_parallel_first_rank(), group=mpu.get_pipeline_model_parallel_group())
    wait_for_dp_and_cp_and_tp_neighbors()

    # broadcast data in first and last pp stage
    if mpu.is_pipeline_first_stage():
        broadcast_in_pp_stage(input_data)
    elif mpu.is_pipeline_last_stage():
        broadcast_in_pp_stage(input_data)
    wait_for_dp_and_cp_and_tp_neighbors()

    return input_data

def slice_2D_tensor_for_data_parallelism(tensor: torch.Tensor, rank: int, parallel_size: int):
    # tensor: (batch, length)
    # slice along batch dim
    if parallel_size == 1:
        return tensor
    assert tensor.shape[0] % parallel_size == 0
    size_per_rank = tensor.shape[0] // parallel_size
    l = rank * size_per_rank
    r = l + size_per_rank
    sliced_tensor = tensor[l:r, :]
    return sliced_tensor

def slice_2D_tensor_for_context_parallelism(tensor: torch.Tensor, rank: int, parallel_size: int):
    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.

    if parallel_size == 1:
        return tensor
    assert tensor.shape[1] % (2 * parallel_size) == 0

    batch_size, seq_length = tensor.shape

    # Split the sequence into 2 * cp_size chunks
    chunks = 2 * parallel_size
    chunk_size = seq_length // chunks

    # Reshape the tensor to [batch_size, chunks, chunk_size]
    tensor = tensor.reshape(tensor.shape[0], chunks, chunk_size)

    # Indices for the current CP rank
    indices = torch.tensor([rank, 2 * parallel_size - rank - 1], device=tensor.device)

    # Index select the two relevant chunks
    sliced_tensor = tensor.index_select(1, indices)

    # Merge the two selected chunks back into a single sequence dimension
    sliced_tensor = sliced_tensor.reshape(batch_size, -1)

    return sliced_tensor

def get_device():
    return torch.device("cuda", torch.cuda.current_device())

def broadcast_in_pp_stage(data: torch.Tensor):
    # we assume data is in dp[0]cp[0]tp[0]

    # broadcast dp group
    dist.broadcast(data, src=mpu.get_data_parallel_src_rank(), group=mpu.get_data_parallel_group())

    # broadcast cp group
    if mpu.get_context_parallel_world_size() > 1:
        dist.broadcast(data, src=mpu.get_context_parallel_global_ranks()[0], group=mpu.get_context_parallel_group())

    # broadcast tp group
    dist.broadcast(data, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

def gather_and_merge_dp_sharded_tensor(tensor: torch.Tensor, merge_dim: int):
    if mpu.get_data_parallel_rank() == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(mpu.get_data_parallel_world_size())]
    else:
        gather_list = None
    dist.gather(tensor, gather_list, dst=mpu.get_data_parallel_src_rank(), group=mpu.get_data_parallel_group())
    merged_weight = None
    if mpu.get_data_parallel_rank() == 0:
        merged_weight = torch.cat(gather_list, dim=merge_dim)
    return merged_weight

def gather_and_merge_cp_sharded_tensor(tensor: torch.Tensor, merge_dim: int):
    sliced_seq_len = tensor.shape[1]
    chunk_size = sliced_seq_len // 2
    cp_size = mpu.get_context_parallel_world_size()

    if mpu.get_context_parallel_rank() == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(cp_size)]
    else:
        gather_list = None
    dist.gather(tensor, gather_list, dst=mpu.get_context_parallel_global_ranks()[0], group=mpu.get_context_parallel_group())

    merged_weight = None
    if mpu.get_context_parallel_rank() == 0:
        full_seq_len = sliced_seq_len * cp_size
        global_blocks = [None] * (2 * cp_size)
        for cp_rank in range(cp_size):
            data = gather_list[cp_rank]
            block_a = data[:, :chunk_size]  # First chunk
            block_b = data[:, chunk_size:]  # Second chunk

            # Determine the global positions of the two chunks
            global_idx_a = cp_rank
            global_idx_b = 2 * cp_size - cp_rank - 1

            global_blocks[global_idx_a] = block_a
            global_blocks[global_idx_b] = block_b
        merged_weight = torch.cat(global_blocks, dim=merge_dim).contiguous()

    return merged_weight

def gather_and_merge_tp_sharded_weight(weight: torch.Tensor, merge_dim: int):
    if mpu.get_tensor_model_parallel_rank() == 0:
        gather_list = [torch.empty_like(weight) for _ in range(mpu.get_tensor_model_parallel_world_size())]
    else:
        gather_list = None
    dist.gather(weight, gather_list, dst=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    merged_weight = None
    if mpu.get_tensor_model_parallel_rank() == 0:
        merged_weight = torch.cat(gather_list, dim=merge_dim)
    del gather_list
    return merged_weight

def stack_gather_and_merge_mtp_sharded_expert_weights(local_expert_weights: List[Tuple[int, torch.Tensor]], merge_dim: int, etp_src_rank: int):
    # ep[:]etp[:] should enter this function
    local_expert_ids = [ew[0] for ew in local_expert_weights]
    local_expert_ids = torch.tensor(local_expert_ids, dtype=torch.int32, device=get_device())

    local_expert_tensors = [ew[1] for ew in local_expert_weights]
    local_expert_tensors_stacked = torch.stack(local_expert_tensors, dim=0)

    if mpu.get_expert_tensor_parallel_rank() == 0:
        gather_list = [torch.empty_like(local_expert_tensors_stacked) for _ in range(mpu.get_expert_tensor_parallel_world_size())]
    else:
        gather_list = None
    dist.gather(local_expert_tensors_stacked, gather_list, dst=etp_src_rank, group=mpu.get_expert_tensor_parallel_group())
    merged_expert_tensors_stacked = None
    if mpu.get_expert_tensor_parallel_rank() == 0:
        merged_expert_tensors_stacked = torch.cat(gather_list, dim=merge_dim + 1)
    del gather_list
    return local_expert_ids, merged_expert_tensors_stacked

def gather_and_unbind_expert_weights(local_expert_ids: torch.Tensor, local_expert_weights_stacked: torch.Tensor, ep_src_rank: int) -> List[Tuple[int, torch.Tensor]]:
    # only ep[:]etp[0] should enter this function
    if mpu.get_expert_model_parallel_rank() == 0:
        expert_ids_gather_list = [torch.empty_like(local_expert_ids) for _ in range(mpu.get_expert_model_parallel_world_size())]
    else:
        expert_ids_gather_list = None
    dist.gather(local_expert_ids, expert_ids_gather_list, dst=ep_src_rank, group=mpu.get_expert_model_parallel_group())
    if mpu.get_expert_model_parallel_rank() == 0:
        global_expert_ids = torch.cat(expert_ids_gather_list, dim=0)
        global_expert_ids = global_expert_ids.cpu().tolist()
    del expert_ids_gather_list
    
    if mpu.get_expert_model_parallel_rank() == 0:
        expert_tensors_list = [torch.empty_like(local_expert_weights_stacked) for _ in range(mpu.get_expert_model_parallel_world_size())]
    else:
        expert_tensors_list = None
    dist.gather(local_expert_weights_stacked, expert_tensors_list, dst=ep_src_rank, group=mpu.get_expert_model_parallel_group())
    if mpu.get_expert_model_parallel_rank() == 0:
        global_expert_tensors_stacked = torch.cat(expert_tensors_list, dim=0)
        global_expert_tensors = list(torch.unbind(global_expert_tensors_stacked, dim=0))
        assert len(global_expert_tensors) == len(global_expert_ids)
    del expert_tensors_list

    global_expert_weights = None
    if mpu.get_expert_model_parallel_rank() == 0:
        global_expert_weights = [(expert_id, expert_tensor) for expert_id, expert_tensor in zip(global_expert_ids, global_expert_tensors)]
    return global_expert_weights

def get_ep_and_etp_src_rank():
    # ep[:]etp[:] should enter this function
    local_item = (dist.get_rank(), mpu.get_expert_model_parallel_rank(), mpu.get_expert_tensor_parallel_rank())
    gather_list = [None for _ in range(mpu.get_expert_tensor_and_model_parallel_world_size())]
    dist.all_gather_object(gather_list, local_item, group=mpu.get_expert_tensor_and_model_parallel_group())

    ep_src_rank = None
    etp_src_rank = None

    for item in gather_list:
        if item[2] == mpu.get_expert_tensor_parallel_rank() and item[1] == 0:
            ep_src_rank = item[0]
        if item[1] == mpu.get_expert_model_parallel_rank() and item[2] == 0:
            etp_src_rank = item[0]
    
    return ep_src_rank, etp_src_rank
                

def send_and_receive_local_layer_weights(num_layers: int,
                                         template: Optional[torch.Tensor] = None,
                                         local_layer_weights: Optional[List[torch.Tensor]] = None,
                                         sender_pp_rank: Optional[int] = None):
    if mpu.get_pipeline_model_parallel_rank() == 0:
        assert template is not None or local_layer_weights is not None
        assert sender_pp_rank is not None and sender_pp_rank > 0
        if local_layer_weights is None:
            received_layer_weights = [None] * num_layers
        else:
            received_layer_weights = local_layer_weights
        for i in range(num_layers):
            if received_layer_weights[i] is None:
                received_layer_weights[i] = torch.empty_like(template)
            sender_global_rank = dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())[sender_pp_rank]
            dist.recv(received_layer_weights[i], src=sender_global_rank, group=mpu.get_pipeline_model_parallel_group())
        return received_layer_weights
    else:
        assert local_layer_weights is not None
        assert num_layers == len(local_layer_weights), f"num_layers: {num_layers}, length of local_layer_weights: {len(local_layer_weights)}"
        for local_weight in local_layer_weights:
            receiver_global_rank = dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())[0]
            dist.send(local_weight, dst=receiver_global_rank, group=mpu.get_pipeline_model_parallel_group())
        return None

def wait_for_pp_and_tp_neighbors():
    dist.barrier(mpu.get_pipeline_model_parallel_group())
    dist.barrier(mpu.get_tensor_model_parallel_group())

def wait_for_tp_neighbors():
    dist.barrier(mpu.get_tensor_model_parallel_group())

def wait_for_ep_etp_neighbors():
    dist.barrier(mpu.get_expert_model_parallel_group())
    dist.barrier(mpu.get_expert_tensor_parallel_group())

def wait_for_pp_ep_and_etp_neighbors():
    dist.barrier(mpu.get_expert_model_parallel_group())
    dist.barrier(mpu.get_expert_tensor_parallel_group())
    dist.barrier(mpu.get_pipeline_model_parallel_group())

def wait_for_dp_and_cp_and_tp_neighbors():
    dist.barrier(mpu.get_data_parallel_group())
    dist.barrier(mpu.get_tensor_and_context_parallel_group())

# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg

def get_each_stage_tp0_available_memory():
    # only tp0 cp0 dp0 enter this function

    device = get_device()
    
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)

    free_memory = total_memory - reserved_memory

    free_memory_tensor = torch.tensor([free_memory], dtype=torch.int64, device=get_device())

    dist.all_reduce(free_memory_tensor, op=ReduceOp.MIN, group=mpu.get_pipeline_model_parallel_group())

    return free_memory_tensor.cpu().item()

def get_hostname():
    """
    获取当前主机的 hostname
    
    Returns:
        str: 主机名
    """
    hostname = os.environ.get("LIGHT_SCALE_HOSTNAME", os.environ.get("MRL_HOSTNAME", None))
    if hostname is None:
        return socket.gethostname()
    else:
        return hostname
    
def get_ip_address():
    # 先尝试解析hostname，若失败，尝试获取 LIGHT_SCALE_IP 环境变量，若仍失败，回退到旧的 MRL_IP。
    hostname = get_hostname()
    try:
        ip = socket.gethostbyname(hostname)
        return ip
    except Exception:
        pass

    ip_env = os.environ.get("LIGHT_SCALE_IP", os.environ.get("MRL_IP", None))
    if ip_env is not None:
        return ip_env

    raise RuntimeError("Unable to determine IP address")

def is_pp_src_rank():
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
        return True
    else:
        return False