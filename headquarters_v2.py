import argparse
import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
from commands_runner import run_commands_in_pool
import sys

import yaml

GPU_PER_NODE = 8


@dataclass
class ResourceInstancePlan:
    """描述一个需要在一个或多个节点启动的 resource 实例。"""
    # 服务且仅服务于resource实例的启动

    service_name: str # 所属 service 名称
    resource_type: str # resource 类型, 如 sglang
    instance_id: int # 同一service需要的resource实例序号
    base_port: int # 实例使用的基础端口号
    # gpu_ids: List[int] = field(default_factory=list) # 实例在当前节点上使用的 GPU 列表
    node_ranks: List[int] # 实例所占用的节点列表，按节点rank排序，第一个为主节点
    gpu_allocations: Optional[Dict[int, List[int]]] # 当前resource实例在各节点的GPU分配情况，key为node_rank，value为该节点分配的gpu_id列表；若为None表示CPU实例
    resource_params: Dict[str, Any] # 由用户定义的resource 启动参数

    # 以上字段应该在实例创建时完整提供，且每台节点的同一实例上述字段应完全一致

    # 以下两个字段在逐节点归纳ResourceInstancePlan实例时填写，
    # local_instance_offset: int # 在所属节点内的偏移量，此变量不需严格正确，只需要避免后续启动时端口冲突即可
    # _resolved_meta: Dict[str, Any] = field(default_factory=dict) # 内部使用的运行时元数据，不应写入用户提供的 resource_params


@dataclass
class HQAssignments:
    """总体角色分配结构，涵盖主训练任务与 async rollout 服务。"""
    # 定义了当前节点及全局资源分配计划，各节点应该构造出完全相同的 HQAssignments 实例，除前两个字段外。

    run_actor: bool = False # 当前节点是否运行 actor 角色
    run_reference: bool = False # 当前节点是否运行 reference 角色
    actor_master_rank: int = 0 # actor 主节点序号必须存在且通常为0
    reference_master_rank: Optional[int] = None # reference 主节点序号，仅当需要时存在
    actor_nodes: int = 0 # actor 角色节点数
    reference_nodes: int = 0 # reference 角色节点数
    reference_start_rank: Optional[int] = None # reference 起始节点序号，仅当需要时存在
    global_gpu_resources: List[ResourceInstancePlan] = field(default_factory=list) # 全局gpu资源实例规划
    global_cpu_resources: List[ResourceInstancePlan] = field(default_factory=list) # 全局cpu资源实例规划

    def to_metadata(self) -> Dict[str, Any]:
        """便于序列化的字典，用于写入 resolved 配置文件。"""

        return {
            "actor_master_rank": self.actor_master_rank,
            "reference_master_rank": self.reference_master_rank,
            "actor_nodes": self.actor_nodes,
            "reference_nodes": self.reference_nodes,
            "reference_start_rank": self.reference_start_rank,
            "async_gpu_services": [asdict(item) for item in self.global_gpu_resources],
            "async_cpu_services": [asdict(item) for item in self.global_cpu_resources],
        }


def parse_args() -> argparse.Namespace:
    """解析命令行参数，保持与原 HQ 一致的接口。"""

    parser = argparse.ArgumentParser(description="The HQ of Megatron-RL (Async V2)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--hostfile", type=str, default=None, help="Path to hostfile")
    parser.add_argument("--node-rank", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置，遇到空文件返回空字典。"""

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if config.get("main") == "main_actor_model":
        print(
            "WARNING: main_actor_model is deprecated and will be removed in a future release. "
            "Please migrate this config to main_async_actor.",
            flush=True,
        )
    return config


def get_node_list(args: argparse.Namespace) -> List[str]:
    """获取节点列表，可来自 hostfile 或 NODE_LIST 环境变量。"""

    if args.hostfile and os.path.exists(args.hostfile):
        print(f"using provided hostfile: {args.hostfile}")
        with open(args.hostfile, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    print("using env NODE_LIST")
    node_list_str = os.getenv("NODE_LIST")
    if node_list_str:
        return [node.strip() for node in node_list_str.split(",") if node.strip()]
    return []


def get_node_rank(args: argparse.Namespace, node_list: Sequence[str]) -> int:
    """判定当前节点序号，优先使用参数与环境变量。"""

    if args.node_rank is not None:
        print(f"using provided node rank: {args.node_rank}")
        return args.node_rank
    env_rank = os.getenv("NODE_RANK")
    if env_rank is not None:
        value = int(env_rank)
        print(f"using env NODE_RANK: {value}")
        return value
    print("get node rank from node list")
    from light_scale.launcher_utils import (  # 延迟导入避免循环依赖
        get_node_rank as get_node_rank_from_node_list,
    )

    return get_node_rank_from_node_list(node_list)


def should_enable_reference(config: Dict[str, Any]) -> bool:
    """复用旧逻辑决定是否启用 reference 角色。"""

    distillation_cfg = config.get("distillation")
    if isinstance(distillation_cfg, dict) and bool(distillation_cfg.get("enabled")):
        # 如果是GKD蒸馏训练，则一定启用 reference 角色
        return True
    try:
        # 否则根据 init_kl_coef 判断
        init_kl = float(config.get("algorithm", {}).get("init_kl_coef", 0.0))
    except (TypeError, ValueError):
        init_kl = 0.0
    return init_kl > 1e-8


def get_gpu_requirement(service_cfg: Dict[str, Any]) -> int:
    """根据 service 配置推导单实例 GPU 数量，可按需扩展。"""

    resource_cfg = service_cfg["resource_cfg"]
    resource_type = resource_cfg["type"]
    if resource_type == "sglang":
        params = resource_cfg["params"]
        # 当前版本sglang推理服务gpu数量完全由tp决定
        return int(params["tp"])
    else:
        raise NotImplementedError(f"暂不支持 resource type: {resource_type} 的 GPU 需求推导，请扩展 `get_gpu_requirement` 函数。")


def calculate_role_assignment(
    config: Dict[str, Any],
    node_rank: int,
    total_nodes: int,
    node_list: Sequence[str],
    offline_testing: bool = False,
) -> HQAssignments:
    """依据配置计算当前节点的角色，并准备 async rollout 服务规划。"""

    assignments = HQAssignments()
    if total_nodes == 0:
        raise RuntimeError("node list is empty")

    assert "actor" in config, "配置中缺少 actor 字段"
    actor_cfg = config["actor"]
    assert (actor_cfg["dp"] * actor_cfg["tp"] * actor_cfg["pp"] * actor_cfg["cp"]) % GPU_PER_NODE == 0, "actor 配置的并行度乘积必须是 GPU_PER_NODE 的整数倍"
    actor_nodes = (actor_cfg["dp"] * actor_cfg["tp"] * actor_cfg["pp"] * actor_cfg["cp"]) // GPU_PER_NODE # dp, tp等字段必须存在
    assert actor_nodes > 0, "actor 节点数必须大于 0"
    assignments.actor_nodes = actor_nodes
    assignments.actor_master_rank = 0
    if node_rank < actor_nodes:
        assignments.run_actor = True

    if offline_testing or config.get("main") == "main_sft":
        raise NotImplementedError("async rl入口暂不支持离线ckpt测试与SFT")
        # assignments.reference_nodes = 0
        # assignments.rollout_nodes = 0
        # return assignments

    if should_enable_reference(config):
        assert "reference" in config, "配置中缺少 reference 字段"
        reference_cfg = config["reference"]
        assert (reference_cfg["dp"] * reference_cfg["tp"] * reference_cfg["pp"] * reference_cfg["cp"]) % GPU_PER_NODE == 0, "reference 配置的并行度乘积必须是 GPU_PER_NODE 的整数倍"
        ref_nodes = (reference_cfg["dp"] * reference_cfg["tp"] * reference_cfg["pp"] * reference_cfg["cp"]) // GPU_PER_NODE
        assert ref_nodes > 0, "reference 节点数必须大于 0"
    else:
        ref_nodes = 0
    assignments.reference_nodes = ref_nodes
    ref_start_rank = actor_nodes
    assignments.reference_start_rank = ref_start_rank
    if ref_nodes > 0 and ref_start_rank <= node_rank < ref_start_rank + ref_nodes:
        assignments.run_reference = True

    # 旧版 rollout 流水线已被 async_rollout 各类服务取代，此处直接清零相关字段。
    # assignments.rollout_nodes = 0
    # assignments.rollout_instances = []
    # assignments.run_rollout = False

    gpu_slots, whole_node_pool = _build_initial_gpu_slots(total_nodes, actor_nodes, ref_nodes)
    _allocate_async_services(config, assignments, gpu_slots, whole_node_pool, total_nodes)
    _log_gpu_usage_summary(gpu_slots)
    return assignments


def _build_initial_gpu_slots(
    total_nodes: int,
    actor_nodes: int,
    ref_nodes: int,
) -> Tuple[List[List[int]], List[int]]:
    """初始化每个节点的可用 GPU 列表，并返回整机空闲节点池。"""

    gpu_slots = [[gpu for gpu in range(GPU_PER_NODE)] for _ in range(total_nodes)]
    for reserved_node_rank in range(actor_nodes + ref_nodes):
        if reserved_node_rank < total_nodes:
            gpu_slots[reserved_node_rank] = []
    # whole_node_pool 用于多机实例直接占整台节点，避免与单机多实例争抢。
    whole_node_pool = [idx for idx, slots in enumerate(gpu_slots) if len(slots) == GPU_PER_NODE]
    # gpu slots: [[0,1,2,3,4,5,6,7], [], [], [0,1,2,3,4,5,6,7], ...]
    # whole_node_pool: [0, 3, 4, ...]
    return gpu_slots, whole_node_pool

def _make_sure_service_names_unique_and_valid(services_cfg: List[Dict[str, Any]], actor_service_name: str) -> None:
    # 检查各service name没有重复，且actor_service_name存在
    service_names = set()
    for service_cfg in services_cfg:
        name = service_cfg.get("name", None)
        if not name:
            raise ValueError("每个 service 配置必须包含唯一的 name 字段")
        if name in service_names:
            raise ValueError(f"发现重复的 service name: {name}")
        service_names.add(name)
    if actor_service_name not in service_names:
        raise ValueError(f"actor_service_name '{actor_service_name}' 不在 services 配置中")
    
def _validate_existing_resource(resource: Dict[str, Any]) -> None:
    # 检查 resource 配置的合法性
    '''
    resource = Resource(
        type=resource_cfg["type"],
        name=resource_cfg["name"],
        base_url=resource_cfg["base_url"],
        # port=int(resource_cfg["port"]),
    )
    '''
    assert "type" in resource, f"{resource} 缺少 type 字段"
    assert "name" in resource, f"{resource} 缺少 name 字段"
    assert "base_url" in resource, f"{resource} 缺少 base_url 字段"
    # assert "port" in resource, f"{resource} 缺少 port 字段"

def _validate_resource_cfg(resource_cfg: Dict[str, Any]):
    # 检查 resource_cfg 的合法性
    '''
    type: sglang
    is_gpu_resource: True
    num_instances: 8 # 实例数量
    params:
        model_path: ''
        tp: 2 # 张量并行
        enable_ep: False # 启用专家并行
        
        port: 18580
    '''
    assert "type" in resource_cfg, f"{resource_cfg} 缺少 type 字段"
    assert "is_gpu_resource" in resource_cfg, f"{resource_cfg} 缺少 is_gpu_resource 字段"
    assert "params" in resource_cfg, f"{resource_cfg} 缺少 params 字段"
    assert "num_instances" in resource_cfg, f"{resource_cfg} 缺少 num_instances 字段"
    assert "base_port" in resource_cfg, f"{resource_cfg} 缺少 base_port 字段"
    assert type(resource_cfg["base_port"]) == int and resource_cfg["base_port"] > 0, f"{resource_cfg} 的 base_port 字段必须为正整数"

    if resource_cfg["type"] == "sglang":
        params = resource_cfg["params"]
        assert "tp" in params, f"{resource_cfg} 的 params 缺少 tp 字段"
        assert "model_path" in params, f"{resource_cfg} 的 params 缺少 model_path 字段"
        assert "dist_base_port" in params, f"{resource_cfg} 的 params 缺少 dist_base_port 字段"


def _allocate_async_services(
    config: Dict[str, Any],
    assignments: HQAssignments,
    gpu_slots: List[List[int]],
    whole_node_pool: List[int],
    total_nodes: int,
) -> None:
    """根据 async_rollout 配置分配服务实例，支持单机多实例与多机单实例。"""

    assert "async_rollout" in config, "配置中缺少 async_rollout 字段"
    async_cfg = config["async_rollout"]
    assert "services" in async_cfg, "async_rollout 配置中缺少 services 字段"
    services_cfg: List[Dict[str, Any]] = async_cfg["services"]
    assert type(services_cfg) == list, "services 字段必须是列表"
    assert len(services_cfg) > 0, "services 列表不能为空，至少有一个actor模型推理服务"
    assert "actor_service_name" in async_cfg, "async_rollout 配置中缺少 actor_service_name 字段"
    _make_sure_service_names_unique_and_valid(services_cfg, async_cfg["actor_service_name"])

    cpu_node_cursor = 0
    for service_cfg in services_cfg:
        # 必填字段直接取值，若不存在直接抛KeyError
        service_name = service_cfg["name"]
        resource_cfg: Optional[Dict[str, Any]] = service_cfg.get("resource_cfg", None)
        # params = resource_cfg["params"] # 若resource_cfg存在，则其中params必存在

        existing_resources = service_cfg.get("resources", [])
        if len(existing_resources) > 0:
            for resource in existing_resources:
                _validate_existing_resource(resource)
            # assignments.external_services[service_name] = existing_resources
        else:
            # 确保提供了resource_cfg
            assert resource_cfg is not None, f"{service_name} 未提供 resource_cfg 或 resources"
            _validate_resource_cfg(resource_cfg)

        num_instances = resource_cfg["num_instances"] if resource_cfg else len(existing_resources) # 如果同时提供了resources和resource_cfg，则实例数以resource_cfg为准
        pending_instances = num_instances - len(existing_resources)
        assert pending_instances >= 0, f"{service_name} 提供的 resources 数量超过 num_instances"

        if pending_instances == 0:
            continue

        is_gpu_service = resource_cfg["is_gpu_resource"]
        resource_type = resource_cfg["type"]
        base_port = resource_cfg["base_port"]
        gpu_requirement = None
        if is_gpu_service:
            # 所有 GPU 服务在进入实例循环前先统一做合法性校验，避免在内部多次重复判断。
            gpu_requirement = get_gpu_requirement(service_cfg)
            if gpu_requirement <= 0:
                raise ValueError(f"{service_name} 的 GPU 需求必须为正整数，当前值: {gpu_requirement}")
        
        for instance_id in range(len(existing_resources), num_instances):
            if is_gpu_service:
                if gpu_requirement <= GPU_PER_NODE:
                    # 单机多实例：同一节点可以分配多个 plan，每个 plan 只占用自身所需的显卡。
                    plan = _allocate_single_node_gpu_service(
                        service_name=service_name,
                        resource_type=resource_type,
                        gpu_slots=gpu_slots,
                        whole_node_pool=whole_node_pool,
                        gpus_required=gpu_requirement,
                        instance_id=instance_id,
                        base_port=base_port,
                        resource_params=resource_cfg.get("params", None),
                    )
                else:
                    if gpu_requirement % GPU_PER_NODE != 0:
                        raise ValueError(
                            f"{service_name} 单实例 GPU 数 {gpu_requirement} 不是 {GPU_PER_NODE} 的整数倍，"
                            "多机实例仅支持整节点倍数。"
                        )
                    # 多机单实例：一次性拿整机列表，保证每台节点全卡独占。
                    plan = _allocate_multi_node_gpu_service(
                        service_name=service_name,
                        resource_type=resource_type,
                        gpu_slots=gpu_slots,
                        whole_node_pool=whole_node_pool,
                        gpus_required=gpu_requirement,
                        instance_id=instance_id,
                        base_port=base_port,
                        resource_params=resource_cfg.get("params", None),
                    )
                assignments.global_gpu_resources.append(plan)
            else:
                plan = _allocate_single_cpu_service(
                    service_name=service_name,
                    resource_type=resource_type,
                    node_rank=cpu_node_cursor % total_nodes,
                    instance_id=instance_id,
                    base_port=base_port,
                    resource_params=resource_cfg.get("params", None),
                )
                assignments.global_cpu_resources.append(plan)
                cpu_node_cursor += 1


def _allocate_single_node_gpu_service(
    service_name: str,
    resource_type: str,
    gpu_slots: List[List[int]],
    whole_node_pool: List[int],
    gpus_required: int,
    instance_id: int,
    base_port: int,
    resource_params: Dict[str, Any],
) -> ResourceInstancePlan:
    """在单节点上分配一个 GPU 服务实例，支持同节点多实例。"""

    for node_rank, slots in enumerate(gpu_slots):
        if len(slots) >= gpus_required:
            gpu_ids = [slots.pop(0) for _ in range(gpus_required)] # 分配所需 GPU 后从槽位列表移除
            if node_rank in whole_node_pool and len(slots) < GPU_PER_NODE:
                # 该节点已不再是整机空闲，需从 whole_node_pool 移除
                whole_node_pool.remove(node_rank)
            return ResourceInstancePlan(
                service_name=service_name,
                resource_type=resource_type,
                instance_id=instance_id,
                # local_instance_offset=0, # for now we set it to 0
                # gpu_ids=gpu_ids,
                node_ranks=[node_rank],
                gpu_allocations={node_rank: gpu_ids},
                resource_params=resource_params,
                base_port=base_port
            )
    raise RuntimeError(
        f"无法在单节点上为 {service_name} 分配 {gpus_required} 张 GPU，请检查节点剩余资源。"
    )


def _allocate_multi_node_gpu_service(
    service_name: str,
    resource_type: str,
    gpu_slots: List[List[int]],
    whole_node_pool: List[int],
    gpus_required: int,
    instance_id: int,
    base_port: int,
    resource_params: Dict[str, Any],
) -> ResourceInstancePlan:
    """为需要整机倍数 GPU 的实例分配多个节点，每台节点全卡独占。"""

    nodes_needed = gpus_required // GPU_PER_NODE
    if len(whole_node_pool) < nodes_needed:
        raise RuntimeError(
            f"{service_name} 需要 {nodes_needed} 台整机，但可用整机仅 {len(whole_node_pool)} 台。"
        )
    allocated_nodes: List[int] = []
    
    for _ in range(nodes_needed):
        node_rank = whole_node_pool.pop(0)
        gpu_slots[node_rank] = []
        allocated_nodes.append(node_rank)
    # allocation_entries = [
    #     {"node_rank": node_rank, "gpu_ids": allocations[node_rank]}
    #     for node_rank in allocated_nodes
    # ]
    gpu_allocations = {node_rank: [gpu for gpu in range(GPU_PER_NODE)] for node_rank in allocated_nodes}
    # metadata = {
    #     "port": base_port + instance_id if base_port is not None else None,
    #     "extra_params": extra_metadata,
    #     "allocations": allocation_entries,
    # }
    # flattened_gpu_ids: List[int] = [gpu for gpu in allocations[allocated_nodes[0]]]
    return ResourceInstancePlan(
        service_name=service_name,
        resource_type=resource_type,
        instance_id=instance_id,
        # local_instance_offset=0, # for now we set it to 0
        # gpu_ids=[gpu for gpu in range(GPU_PER_NODE)],
        node_ranks=allocated_nodes,
        gpu_allocations=gpu_allocations,
        resource_params=resource_params,
        base_port=base_port
    )


def _allocate_single_cpu_service(
    service_name: str,
    resource_type: str,
    node_rank: int,
    instance_id: int,
    base_port: int,
    resource_params: Dict[str, Any],
) -> ResourceInstancePlan:
    """CPU 服务无需 GPU，占据指定节点即可。"""

    # metadata = {
    #     "port": base_port + instance_id if base_port is not None else None,
    #     "extra_params": extra_metadata,
    # }
    return ResourceInstancePlan(
        service_name=service_name,
        resource_type=resource_type,
        instance_id=instance_id,
        # local_instance_offset=0,
        node_ranks=[node_rank],
        resource_params=resource_params,
        base_port=base_port,
        gpu_allocations=None
    )


def _log_gpu_usage_summary(gpu_slots: List[List[int]]) -> None:
    """调试辅助：输出每个节点剩余的 GPU 槽位。"""

    summary = {idx: slots for idx, slots in enumerate(gpu_slots)}
    print("[async-rollout] GPU 槽位剩余情况:")
    for node, slots in summary.items():
        print(f"  node {node}: remaining GPUs {slots}")


# def build_node_service_map(assignments: HQAssignments, node_list: Sequence[str]) -> Dict[int, List[ResourceInstancePlan]]:
#     """
#     按节点聚合 resource 实例，写入本地 offset、端口、gpu_ids，并缓存到 assignments。
#     返回 node_rank -> [ResourceInstancePlan]。
#     """
#     node_to_services: Dict[int, List[ResourceInstancePlan]] = {}
#     node_instances: Dict[int, int] = {}  # 记录每个节点已分配的实例数量
#     node_port_cursor: Dict[int, Dict[str, int]] = {}  # node_rank -> service_name -> 当前端口游标

#     # 聚合所有 GPU/CPU 服务
#     all_plans = list(assignments.global_gpu_resources) + list(assignments.global_cpu_resources)
#     for plan in all_plans:
#         for node_rank in plan.node_ranks:
#             if node_rank not in node_to_services:
#                 node_to_services[node_rank] = []
#                 node_instances[node_rank] = 0
#                 node_port_cursor[node_rank] = {}
#             local_offset = node_instances[node_rank]
#             # 端口分配：每个节点每个 service_name 独立游标
#             service_name = plan.service_name
#             base_port = plan.base_port
#             if service_name not in node_port_cursor[node_rank]:
#                 node_port_cursor[node_rank][service_name] = base_port
#             port = node_port_cursor[node_rank][service_name]
#             node_port_cursor[node_rank][service_name] += 1

#             # 为每个参与节点创建独立的 per-node plan（避免就地修改原始 plan）
#             new_plan = copy.deepcopy(plan)
#             new_plan.node_ranks = [node_rank]
#             new_plan.local_instance_offset = local_offset

#             # 多节点实例需为每个节点写入本地 gpu_ids
#             # 如果为 GPU 多节点实例，`plan.gpu_allocations` 应为非空映射——直接索引以便在不满足不变量时抛出异常。
#             if plan.gpu_allocations:
#                 new_plan.gpu_ids = plan.gpu_allocations[node_rank]

#             # 不要把内部调度元数据写入 user-provided `resource_params`。
#             # 将本地解析后的运行元数据写到 per-node plan 的内部属性 `_resolved_meta`，以避免污染用户字段。
#             new_plan._resolved_meta = {
#                 'port': port,
#                 'local_instance_offset': local_offset,
#                 'node_rank': node_rank,
#                 'base_url': _build_base_url(node_list, node_rank, port),
#             }

#             node_to_services[node_rank].append(new_plan)
#             node_instances[node_rank] += 1

#     # 缓存到 assignments 便于后续复用
#     # assignments.node_service_map = node_to_services
#     return node_to_services

def build_per_node_resource_plans(assignments: HQAssignments):
    node_to_resource_plans: Dict[int, List[ResourceInstancePlan]] = dict()

    plans = assignments.global_gpu_resources + assignments.global_cpu_resources
    for plan in plans:
        for node_rank in plan.node_ranks:
            if node_rank not in node_to_resource_plans:
                node_to_resource_plans[node_rank] = []
            node_to_resource_plans[node_rank].append(plan)
    
    return node_to_resource_plans


def emit_resolved_async_config(
    config: Dict[str, Any],
    assignments: HQAssignments,
    node_list: Sequence[str],
    node_rank: int,
) -> Optional[str]:
    """
    仅 node_rank==0 写 resolved async_rollout 配置到 output_dir，返回路径。
    其余节点直接返回 None。
    """
    if node_rank != 0:
        return None

    # 获取输出目录
    output_dir = config['training']['output_dir']
    if not output_dir:
        raise RuntimeError('未指定 output_dir，无法写 resolved 配置')
    os.makedirs(output_dir, exist_ok=True)
    resolved_path = os.path.join(output_dir, 'resolved_async_rollout.yaml')

    # 构造 resolved 配置
    # node_to_services = getattr(assignments, 'node_service_map', None)
    # if node_to_services is None:
    #     node_to_services = build_node_service_map(assignments, node_list)

    # 合并 external_services 与 HQ 规划的实例
    async_cfg = config['async_rollout']
    async_cfg = copy.deepcopy(async_cfg)
    services_cfg: List[Dict[str, Any]] = async_cfg['services']
    # actor_service_name = async_cfg['actor_service_name']
    # resolved_services = []
    for service_cfg in services_cfg:
        name = service_cfg['name']
        # 先加 external_services（只保留严格字段）
        resources: List[Dict[str, Any]] = service_cfg.get('resources', [])
        # for r in assignments.external_services.get(name, []):
        #     resources.append({
        #         'type': r.get('type'),
        #         'name': r.get('name'),
        #         'base_url': r.get('base_url'),
        #         'port': r.get('port'),
        #     })

        if "resource_cfg" not in service_cfg:
            continue
        resource_cfg = service_cfg["resource_cfg"]

        # 再加 HQ 规划的实例 —— 为多机实例仅使用其首节点的 url
        all_plans = list(assignments.global_gpu_resources) + list(assignments.global_cpu_resources)
        for plan in all_plans:
            if plan.service_name != name:
                continue
            # primary node is the first node in the original plan's node_ranks
            # sanity check: allocator must provide at least one node_rank
            assert plan.node_ranks and len(plan.node_ranks) > 0, (
                f"ResourceInstancePlan {plan.service_name} instance {plan.instance_id} has empty node_ranks"
            )
            primary_node = plan.node_ranks[0]
            entry = {
                'type': resource_cfg['type'],
                'name': plan.service_name,
                'base_url': f"http://{node_list[primary_node]}:{plan.base_port + plan.instance_id}",
                'num_gpus': sum(len(gpus) for gpus in plan.gpu_allocations.values()) if plan.gpu_allocations else 0,
            }
            resources.append(entry)
        service_cfg['resources'] = resources

    # resolved_cfg = {
    #     'services': resolved_services,
    #     'actor_service_name': actor_service_name,
    # }

    with open(resolved_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(async_cfg, f, allow_unicode=True)
    print(f"[HQ] Resolved async_rollout config written to: {resolved_path}")
    return resolved_path


def _build_base_url(node_list: Sequence[str], node_rank: int, port: Optional[int]) -> str:
    """将节点主机名与端口拼装成 base_url。"""

    if node_rank >= len(node_list):
        raise RuntimeError("节点序号超出 node_list 长度")
    host = node_list[node_rank]
    if port is None:
        return f"http://{host}"
    return f"http://{host}:{port}"

def generate_dist_launcher_cmd(
    node_rank: int,
    master_addr: str,
    actor_nodes: int,
    master_port,
    log_dir
):
    cmd = [
        "./dist_launcher.sh",
        f"--MASTER_ADDR {master_addr}",
        f"--MASTER_PORT {master_port}",
        f"--WORLD_SIZE {GPU_PER_NODE * actor_nodes}",
        f"--NPROC_PER_NODE {GPU_PER_NODE}",
        f"--NODE_RANK {node_rank}",
        f"--LOG_DIR {log_dir}"
    ]
    
    return cmd

def generate_actor_cmd(
    config: dict,
    assignments: HQAssignments,
    node_list: list,
    node_rank: int
) -> str:
    """
    生成 actor 启动命令，参考旧 HQ 代码，适配新 assignments 结构。
    """
    # 异步RL训练不支持推1训N的场景
    assert config["training"]["rollout_batch_size"] * config['training']['n_samples'] == config["training"]["global_batch_size"], "rollout_batch_size * n_samples 必须等于 global_batch_size"
    output_dir = config['training']['output_dir']
    assert output_dir, "必须指定输出目录"
    assert config['async_rollout']['data'] and os.path.exists(config['async_rollout']['data']), "必须指定有效的数据路径"
    assert config['training']['resume_checkpoint'] or config['training']['from_pretrained'], "必须指定 resume_checkpoint 或 from_pretrained"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_dir = f"{output_dir}/tensorboard_log"
    os.makedirs(tensorboard_dir, exist_ok=True)

    if assignments.reference_nodes > 0:
        dp_size = math.lcm(config["actor"]["dp"], config["reference"]["dp"])
    else:
        dp_size = config["actor"]["dp"]
    assert config["training"]["global_batch_size"] % dp_size == 0

    # 这里只做最小实现，省略部分参数校验
    cmd = [
        f"python3 {config['main']}.py",
        "--use-checkpoint-args",
        "--bf16",
        "--use-distributed-optimizer",
        "--ckpt-format torch_dist",
        "--distributed-timeout-minutes 90",
        "--sequence-parallel",
        f"--save {os.path.abspath(checkpoint_dir)}",
        f"--tensorboard-dir {os.path.abspath(tensorboard_dir)}",
        f"--save-interval {config['training']['save_steps']}",
        f"--data-path {config['async_rollout']['data']}",
        f"--rollout_batch_size {config['training']['rollout_batch_size']}",
        f"--micro-batch-size {config['training']['micro_batch_size']}",
        f"--global-batch-size {config['training']['global_batch_size']}",
        f"--non-persistent-save-interval {config['training']['non_persistent_save_steps']}",
        f"--non-persistent-ckpt-type global",
        f"--save_test_step {config['training']['save_test_step']}",
        f"--train-iters {config['training']['max_steps']}",
        f"--lr {config['training']['optim']['lr']}",
        f"--lr-decay-style {config['training']['optim']['scheduler_type']}",
        f"--min-lr {config['training']['optim']['min_lr']}",
        f"--n_samples {config['training']['n_samples']}",
        # f"--clip_eps {config['algorithm']['clip_eps']}",
        f"--seq-length {config['training']['max_length']}",
        # f"--lr-warmup-fraction {config['training']['optim']['warmup_ratio']}",
        f"--seed {config['training']['seed']}",
        f"--tensor-model-parallel-size {config['actor']['tp']}",
        f"--pipeline-model-parallel-size {config['actor']['pp']}",
        f"--context-parallel-size {config['actor']['cp']}",
        # f"--max_tokens {config['training']['max_rollout_tokens']}",
        f"--kl_estimator {config['algorithm']['kl_estimator']}",
        f"--weight-decay {config['training']['optim']['weight_decay']}",
        f"--init_kl_coef {config['algorithm']['init_kl_coef']}",
        f"--micro_forward_batch_size {config['training']['micro_forward_batch_size']}",
        f"--light_scale_log_level {config['training']['log_level']}",
        f"--early_stop_steps {config['training']['early_stop_steps']}",
        f"--dist_lock_server_port {config['distributed_lock_server']['port']}",
        f"--async_rollout_cfg_path {os.path.join(output_dir, 'resolved_async_rollout.yaml')}",
        f"--weight_update_group_port {config['actor']['weight_update_group_port']}",
        f"--save_test_step {config['training']['save_test_step']}",
        f"--adam-beta1 {config['training']['optim']['adam_beta1']}",
        f"--adam-beta2 {config['training']['optim']['adam_beta2']}",
        f"--adam-eps {config['training']['optim']['adam_eps']}",
        f"--max_train_batch_size {config['async_rollout'].get('max_train_batch_size', -1)}",
    ]

    # Distillation and GKD related flags (optional; backward compatible)
    distillation_cfg = config.get('distillation')
    if isinstance(distillation_cfg, dict) and distillation_cfg.get('enabled', False):
        # Enable distillation training path
        cmd += ["--distillation_enabled"]

        # Logits express transport parameters
        le_cfg = distillation_cfg['logits_express']
        dtype_val = le_cfg['dtype']
        if dtype_val is not None:
            cmd += [f"--logits_dtype {dtype_val}"]
        bs_val = le_cfg['batch_size']
        if bs_val is not None:
            cmd += [f"--logits_transfer_batch_size {bs_val}"]
        base_port_val = le_cfg['base_port']
        if base_port_val is not None:
            cmd += [f"--logits_pg_base_port {base_port_val}"]

        # GKD loss parameters
        gkd_cfg = distillation_cfg['gkd']
        st = gkd_cfg['student_temperature']
        if st is not None:
            cmd += [f"--student_temperature {st}"]
        tt = gkd_cfg['teacher_temperature']
        if tt is not None:
            cmd += [f"--teacher_temperature {tt}"]
        beta = gkd_cfg['beta']
        if beta is not None:
            cmd += [f"--gkd_beta {beta}"]
        if gkd_cfg['seq_kd']:
            cmd += ["--seq_kd"]
        
        if gkd_cfg.get("gkd_sparse_topk_enabled", False):
            cmd += ["--gkd_sparse_topk_enabled"]
            cmd += [f"--gkd_topk {gkd_cfg['gkd_topk']}"]


    if config['training']['resume_checkpoint'] is None:
        cmd += [
            f"--load {os.path.abspath(config['training']['from_pretrained'])}",
            f"--finetune"
        ]
    else:
        cmd += [
            f"--load {os.path.abspath(config['training']['resume_checkpoint'])}",
        ]
    if config['training']['dump_experience']:
        dump_path = f"{output_dir}/experiences"
        os.makedirs(dump_path, exist_ok=True)
        cmd += [
            "--dump_experience",
            f"--dump_path {dump_path}"
        ]
        if config['training']['dump_tensors']:
            cmd += ["--dump_tensors"]
    if config['algorithm']['use_kl_loss']:
        cmd += ["--use_kl_loss"]
    if config['training']['pad_to_max_length']:
        cmd += ["--pad_to_max_length"]
    if config['training']['use_outcome_rewards_as_advantages']:
        cmd += ["--use_outcome_rewards_as_advantages"]
    if config['training']['skip_zero_reward_sample']:
        cmd += ["--skip_zero_reward_sample"]

    if assignments.reference_nodes > 0:
        cmd += [f"--reference_dp_size {config['reference']['dp'] * config['reference']['micro_batch_size']}"]

    if config['training'].get('mlp_weight_merging_batch_size', None) is not None:
        cmd += [f"--mlp_weight_merging_batch_size {config['training']['mlp_weight_merging_batch_size']}"]
    if config['training'].get('moe_weight_merging_layer_batch_size', None) is not None:
        cmd += [f"--moe_weight_merging_layer_batch_size {config['training']['moe_weight_merging_layer_batch_size']}"]
    if config['training'].get('moe_weight_merging_expert_batch_size', None) is not None:
        cmd += [f"--moe_weight_merging_expert_batch_size {config['training']['moe_weight_merging_expert_batch_size']}"]
    if config['training'].get('policy_loss_limit', None) is not None:
        cmd += [f"--policy_loss_limit '{config['training']['policy_loss_limit']}'"]
    if config['training'].get('kl_loss_limit', None) is not None:
        cmd += [f"--kl_loss_limit '{config['training']['kl_loss_limit']}'"]
    cmd += [f"--total_world_size {len(node_list) * GPU_PER_NODE}"]
    if config['algorithm'].get('entropy_loss_coef', None) is not None:
        cmd += [f"--entropy_loss_coef '{config['algorithm']['entropy_loss_coef']}'"]
    if config["training"].get("entropy_loss_threshold", None) is not None:
        cmd += [f"--entropy_loss_threshold {config['training']['entropy_loss_threshold']}"]

    # moe config
    if config['actor'].get('moe', None) is not None:
        cmd += [f"--moe-grouped-gemm"]
        ep = config['actor']['moe']['ep']
        etp = config['actor']['moe']['etp']
        cmd += [f"--expert-model-parallel-size {ep}"]
        if config['actor']['moe'].get('etp', None) is not None:
            etp = config['actor']['moe']['etp']
            cmd += [f"--expert-tensor-parallel-size {etp}"]

    # set warmup step or ratio
    if config['training']['optim'].get('warmup_steps', None) is not None:
        cmd += [f"--lr-warmup-iters {config['training']['optim']['warmup_steps']}"]
    else:
        cmd += [f"--lr-warmup-fraction {config['training']['optim']['warmup_ratio']}"]

    # Add extra arguments if any
    extra_args = config['actor'].get("extra_args", [])
    for arg in extra_args:
        if arg["key"] and arg["value"] is None:
            cmd.append(f"--{arg['key']}")
        elif arg["key"] and arg["value"]:
            cmd.append(f"--{arg['key']} {arg['value']}")
    cmd = generate_dist_launcher_cmd(
        node_rank, 
        node_list[assignments.actor_master_rank], 
        assignments.actor_nodes, 
        config["actor"]["weight_update_group_port"], 
        f"{config['training']['output_dir']}/actor_log"
    ) + cmd

    return " ".join(cmd)

def generate_ref_cmd(
    config: dict,
    assignments: HQAssignments,
    node_list: list,
    node_rank: int,
) -> str:
    cmd = [
        "python3 main_reference_model.py",
        f"--load {config['reference']['load_path']}",
        "--use-checkpoint-args",
        # "--use-mp-args-from-checkpoint-args",
        "--bf16",
        "--distributed-timeout-minutes 60",
        "--sequence-parallel",
        f"--actor_master_addr {node_list[assignments.actor_master_rank]}",
        f"--data_transfer_group_port {config['actor']['weight_update_group_port']}",
        f"--tensor-model-parallel-size {config['reference']['tp']}",
        f"--pipeline-model-parallel-size {config['reference']['pp']}",
        f"--context-parallel-size {config['reference']['cp']}",
        f"--micro-batch-size {config['reference']['micro_batch_size']}",
        f"--global-batch-size {config['training']['global_batch_size']}",
        f"--light_scale_log_level {config['training']['log_level']}",
    ]

    # Distillation and GKD related flags for reference (required when enabled)
    distillation_cfg = config.get('distillation')
    if isinstance(distillation_cfg, dict) and distillation_cfg.get('enabled', False):
        cmd += ["--distillation_enabled"]

        # Logits express transport parameters
        le_cfg = distillation_cfg['logits_express']
        gkd_cfg = distillation_cfg['gkd']
        dtype_val = le_cfg['dtype']
        if dtype_val is not None:
            cmd += [f"--logits_dtype {dtype_val}"]
        bs_val = le_cfg['batch_size']
        if bs_val is not None:
            cmd += [f"--logits_transfer_batch_size {bs_val}"]
        base_port_val = le_cfg['base_port']
        if base_port_val is not None:
            cmd += [f"--logits_pg_base_port {base_port_val}"]
        if gkd_cfg.get("gkd_sparse_topk_enabled", False):
            cmd += ["--gkd_sparse_topk_enabled"]
            cmd += [f"--gkd_topk {gkd_cfg['gkd_topk']}"]

    # moe config
    if config['reference'].get('moe', None) is not None:
        cmd += [f"--moe-grouped-gemm"]
        ep = config['reference']['moe']['ep']
        etp = config['reference']['moe']['etp']
        cmd += [f"--expert-model-parallel-size {ep}"]
        if config['reference']['moe'].get('etp', None) is not None:
            etp = config['reference']['moe']['etp']
            cmd += [f"--expert-tensor-parallel-size {etp}"]
    cmd = generate_dist_launcher_cmd(
        node_rank - assignments.reference_start_rank, 
        node_list[assignments.reference_start_rank], 
        assignments.reference_nodes, 
        14470, 
        f"{config['training']['output_dir']}/ref_log"
    ) + cmd
    return " ".join(cmd)

def generate_sglang_cmd(
    plan: ResourceInstancePlan,
    node_list: list,
    node_rank: int,
) -> str:
    """
    生成 sglang async 服务的启动命令，适配新 ResourceInstancePlan。
    """
    model_path = plan.resource_params['model_path']
    tp = plan.resource_params['tp']
    enable_ep = plan.resource_params.get('enable_ep', False)
    dist_base_port = plan.resource_params['dist_base_port']
    extra_args = plan.resource_params.get('extra_args', dict())
    instance_rank = None

    assert node_rank in plan.node_ranks, "本节点不在该 plan 的节点列表中"
    for i, r in enumerate(plan.node_ranks):
        if r == node_rank:
            instance_rank = i
            break

    # 使用本节点分配的 GPU 列表生成 CUDA_VISIBLE_DEVICES
    gpu_ids = plan.gpu_allocations[node_rank]
    env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in gpu_ids)} "
    cmd = [
        env_prefix,
        "python3 -m sglang.launch_server",
        f"--port {plan.base_port + plan.instance_id}",
        "--host 0.0.0.0",
        f"--model {model_path}",
        f"--served-model-name {plan.service_name}",
        f"--tp {tp}",
        "--trust-remote-code",
        f"--dist-init-addr {node_list[node_rank]}:{dist_base_port + plan.instance_id}",
        f"--nnodes {len(plan.node_ranks)}",
        f"--node-rank {instance_rank}"
    ]
    if enable_ep:
        cmd += [f"--ep-size {tp}"]
    # Add extra arguments if any
    for arg in extra_args:
        if arg["key"] and arg["value"] is None:
            cmd.append(f"--{arg['key']}")
        elif arg["key"] and arg["value"]:
            cmd.append(f"--{arg['key']} {arg['value']}")

    # 清理空段并返回单行命令
    cmd = " ".join([p for p in cmd if p])
    return cmd

def _validate_output_dir(config):
    # 确保用户在配置中提供了output_dir，并且路径存在或可创建
    training = config.get("training")
    if not training or "output_dir" not in training or not training["output_dir"]:
        raise RuntimeError("配置中必须包含 training.output_dir 且非空")
    output_dir = training["output_dir"]
    if not isinstance(output_dir, str):
        raise RuntimeError("training.output_dir 必须为字符串路径")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"无法创建 output_dir '{output_dir}': {e}")
    # 检查可写性
    test_path = os.path.join(output_dir, ".hq_write_test")
    try:
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception as e:
        raise RuntimeError(f"output_dir '{output_dir}' 不可写: {e}")

def main() -> None:
    """主流程：解析参数 -> 角色分配 -> 输出 resolved 配置。"""


    args = parse_args()
    config = load_config(args.config)
    node_list = get_node_list(args)
    if not node_list:
        print("未获取到节点列表，退出")
        return
    total_nodes = len(node_list)
    node_rank = get_node_rank(args, node_list)
    print(f"{total_nodes} nodes in total, this node rank: {node_rank}")

    _validate_output_dir(config)

    offline_testing = bool(
        config.get("weight_updater_test")
        and config["weight_updater_test"].get("online_test") is False
    )

    # 角色分配
    assignments: HQAssignments = calculate_role_assignment(
        config, node_rank, total_nodes, node_list, offline_testing=offline_testing
    )
    # 节点服务聚合
    # node_to_services = build_node_service_map(assignments, node_list)
    node_to_resource_plans = build_per_node_resource_plans(assignments)

    # 生成并打印本节点需要启动的命令
    cmds = []


    if assignments.run_actor:
        cmds.append(
            generate_actor_cmd(config, assignments, node_list, node_rank)
        )

    if assignments.run_reference:
        cmds.append(
            generate_ref_cmd(config, assignments, node_list, node_rank)
        )

    # async rollout sglang服务
    plans = node_to_resource_plans.get(node_rank, [])
    for plan in plans:
        if plan.resource_type == 'sglang':
            cmds.append(
                generate_sglang_cmd(plan, node_list, node_rank)
            )
        else:
            raise NotImplementedError(f"不支持的异步服务类型: {plan.resource_type}")

    print("\n[HQ] 本节点需启动如下命令：")
    for c in cmds:
        print(c)

    # 只在 node_rank==0 写 resolved 配置
    resolved_path = emit_resolved_async_config(
        config=config,
        assignments=assignments,
        node_list=node_list,
        node_rank=node_rank,
    )
    if resolved_path:
        print(f"[HQ] Resolved async_rollout config written to: {resolved_path}")

    if cmds:
        print("\n# Commands to run:")
        print("\n".join(cmds))
        run_commands(cmds)
    else:
        raise RuntimeError("# No roles assigned to this node")

def run_commands(cmd_list):
    exit_code = run_commands_in_pool(cmd_list)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
