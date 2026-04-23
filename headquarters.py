import os
import yaml
import argparse
from typing import Dict, List
import math
from commands_runner import run_commands
from light_scale.launcher_utils import get_node_rank as get_node_rank_from_node_list

def parse_args():
    parser = argparse.ArgumentParser(description="The HQ of Megatron-RL")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--hostfile", type=str, default=None, help="Path to hostfile")
    parser.add_argument("--node-rank", type=int, default=None)
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and config.get("main") == "main_actor_model":
        print(
            "WARNING: main_actor_model is deprecated and will be removed in a future release. "
            "Please migrate this config to main_async_actor.",
            flush=True,
        )
    return config

def get_node_list(args) -> List[str]:
    if args.hostfile is not None and os.path.exists(args.hostfile):
        print(f"using provided hostfile: {args.hostfile}")
        node_list = []
        with open(args.hostfile) as f:
            for raw_line in f:
                node_list.append(raw_line.strip())
        return node_list
    else:
        print("using env NODE_LIST")
        node_list = os.getenv("NODE_LIST", None)
        if node_list is not None:
            return [node.strip() for node in node_list.split(",") if node.strip()]
        else:
            return []

def get_node_rank(args, node_list) -> int:
    if args.node_rank is not None:
        print(f"using provided node rank: {args.node_rank}")
        return args.node_rank
    if os.getenv("NODE_RANK", None) is not None:
        node_rank = int(os.environ['NODE_RANK'])
        print(f"using env NODE_RANK: {node_rank}")
        return node_rank
    print("get node rank from node list")
    node_rank = get_node_rank_from_node_list(node_list)
    return node_rank

def should_enable_reference(config: Dict) -> bool:
    """Determine whether the reference role should be enabled.

    Rules:
    - If distillation.enabled is explicitly True, enable reference.
    - Otherwise, fall back to legacy behavior: enable if algorithm.init_kl_coef > 1e-8.
    - If the "distillation" field is missing, treat distillation as disabled (legacy path still applies).
    - If config fields are missing or malformed, default to False.
    """
    # New path: explicit distillation switch
    dist_cfg = config.get("distillation")
    if isinstance(dist_cfg, dict) and bool(dist_cfg.get("enabled", False)):
        return True

    # Legacy path: rely on init_kl_coef to decide
    try:
        init_kl = float(config.get("algorithm", {}).get("init_kl_coef", 0.0))
    except (TypeError, ValueError):
        init_kl = 0.0
    return init_kl > 1e-8

def calculate_role_assignment(
    config: Dict,
    node_rank: int,
    total_nodes: int,
    offline_testing: bool = False
) -> Dict[str, bool]:
    """Determine which roles should run on this node based on node rank"""
    assignments = {
        "run_actor": False,
        "run_reference": False,
        "run_rollout": False,
        "actor_master": None,
        "reference_master": None,
        "rollout_instances": [],
        "num_instances_per_node": None
    }
    
    # Calculate how many nodes each role needs
    actor_config = config["actor"]
    actor_nodes = (actor_config["dp"] * actor_config["tp"] * actor_config["pp"] * actor_config["cp"]) // 8

    assignments["actor_master"] = 0
    assignments["actor_nodes"] = actor_nodes
    # Assign actor role
    if node_rank < actor_nodes:
        assignments["run_actor"] = True

    if offline_testing or config['main'] == 'main_sft':
        assignments["reference_nodes"] = 0
        assignments["rollout_nodes"] = 0
        return assignments

    ref_config = config["reference"]
    if should_enable_reference(config):
        ref_nodes = (ref_config["dp"] * ref_config["tp"] * ref_config["pp"] * ref_config["cp"]) // 8
    else:
        ref_nodes = 0

    # Assign reference role (after actor nodes)
    ref_start = actor_nodes
    assignments["reference_start"] = ref_start
    assignments["reference_nodes"] = ref_nodes
    if ref_nodes > 0 and ref_start <= node_rank < ref_start + ref_nodes:
        assignments["run_reference"] = True
        
    # Assign rollout role (after actor and reference nodes)
    # 目前rollout支持单机多实例
    # TODO: 支持多机多实例
    rollout_config = config["rollout"]
    rollout_nodes = (rollout_config["tp"] * rollout_config["num_instances"]) // 8
    assignments['rollout_nodes'] = rollout_nodes
    assert 8 % rollout_config["tp"] == 0
    num_instances_per_node = 8 // rollout_config["tp"]
    assignments["num_instances_per_node"] = num_instances_per_node
    rollout_start = actor_nodes + ref_nodes
    rollout_instances = []
    for rollout_node_offset in range(rollout_nodes):
        rollout_node_rank = rollout_start + rollout_node_offset
        for i in range(num_instances_per_node):
            rollout_instances.append({
                "node_rank": rollout_node_rank,
                "offset":  i,
                "gpu_offset": i * rollout_config["tp"]
            })
    assignments["rollout_instances"] = rollout_instances

    if rollout_nodes > 0 and node_rank >= rollout_start:
        print(f"assigning rollout, rollout_nodes: {rollout_nodes}, node_rank: {node_rank}, rollout_start: {rollout_start}")
        assignments["run_rollout"] = True
    
    return assignments

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
        f"--WORLD_SIZE {8 * actor_nodes}",
        "--NPROC_PER_NODE 8",
        f"--NODE_RANK {node_rank}",
        f"--LOG_DIR {log_dir}"
    ]
    
    return cmd

def generate_actor_cmd(
    config, 
    actor_master, 
    node_list, 
    ref_start,
    node_rank: int,
    actor_nodes: int,
    assignments: dict
):
    assert config["training"]["rollout_batch_size"] * config['training']['n_samples'] >= config["training"]["global_batch_size"] and \
        (config["training"]["rollout_batch_size"] * config['training']['n_samples']) % config["training"]["global_batch_size"] == 0
    rollout_train_iter_coeff = (config["training"]["rollout_batch_size"] * config['training']['n_samples']) // config["training"]["global_batch_size"] * config["algorithm"]["num_repeat_times"]
    output_dir = config['training']['output_dir']
    assert output_dir, "you must specify output directory"
    assert config['training']['data'] and os.path.exists(config['training']['data']), "you must specify valid training data path"
    assert config['training']['resume_checkpoint'] or config['training']['from_pretrained'], "you must specify checkpoint loading path"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_dir = f"{output_dir}/tensorboard_log"
    os.makedirs(tensorboard_dir, exist_ok=True)
    rollout_base_url_list = [f"http://{node_list[instance['node_rank']]}:{config['rollout']['port'] + instance['offset']}" for instance in assignments["rollout_instances"]]

    if assignments["reference_nodes"] > 0:
        dp_size = math.lcm(config["actor"]["dp"], config["reference"]["dp"])
    else:
        dp_size = config["actor"]["dp"]
    assert config["training"]["global_batch_size"] % dp_size == 0
    cmd = [
        f"python3 {config['main']}.py",
        "--use-checkpoint-args",
        # "--use-mp-args-from-checkpoint-args", # should not use this after switching to torch_dist
        "--bf16",
        "--use-distributed-optimizer",
        "--ckpt-format torch_dist",
        "--distributed-timeout-minutes 90",
        "--sequence-parallel",
        f"--save {os.path.abspath(checkpoint_dir)}",
        f"--tensorboard-dir {os.path.abspath(tensorboard_dir)}",
        f"--save-interval {config['training']['save_steps'] * rollout_train_iter_coeff}",
        f"--data-path {config['training']['data']}",
        f"--rollout_batch_size {config['training']['rollout_batch_size']}",
        f"--rollout_base_url_list {' '.join(rollout_base_url_list)}",
        f"--rollout_model_name {config['training']['rollout_model_name']}",
        f"--server_world_size {config['rollout']['tp'] * config['rollout']['num_instances']}",
        f"--micro-batch-size {config['training']['micro_batch_size']}",
        f"--global-batch-size {config['training']['global_batch_size']}",
        f"--train-iters {config['training']['max_steps'] * rollout_train_iter_coeff}",
        f"--lr {config['training']['optim']['lr']}",
        f"--lr-decay-style {config['training']['optim']['scheduler_type']}",
        # f"--lr-warmup-iters {config['training']['optim']['warm_up'] * rollout_train_iter_coeff}",
        f"--min-lr {config['training']['optim']['min_lr']}",
        f"--n_samples {config['training']['n_samples']}",
        f"--num_repeat_times {config['algorithm']['num_repeat_times']}",
        f"--clip_eps {config['algorithm']['clip_eps']}",
        f"--seq-length {config['training']['max_length']}",
        # f"--lr-warmup-fraction {config['training']['optim']['warmup_ratio']}",
        f"--seed {config['training']['seed']}",
        f"--tensor-model-parallel-size {config['actor']['tp']}",
        f"--pipeline-model-parallel-size {config['actor']['pp']}",
        f"--context-parallel-size {config['actor']['cp']}",
        f"--max_tokens {config['training']['max_rollout_tokens']}",
        f"--stop '{config['rollout']['stop']}'",
        f"--kl_estimator {config['algorithm']['kl_estimator']}",
        f"--weight-decay {config['training']['optim']['weight_decay']}",
        f"--init_kl_coef {config['algorithm']['init_kl_coef']}",
        f"--sampling_pool_size {config['rollout']['sampling_pool_size']}",
        f"--micro_forward_batch_size {config['training']['micro_forward_batch_size']}",
        f"--light_scale_log_level {config['training']['log_level']}",
        f"--early_stop_steps {config['training']['early_stop_steps']}",
        f"--dist_lock_server_port {config['distributed_lock_server']['port']}",
        f"--sampling_temperature {config['rollout']['temperature']}",
        f"--sampling_top_k {config['rollout']['top_k']}",
        f"--sampling_top_p {config['rollout']['top_p']}",
        f"--sampling_presence_penalty {config['rollout']['presence_penalty']}",
    ]

    # Distillation and GKD related flags (optional; backward compatible)
    dist_cfg = config.get('distillation')
    if isinstance(dist_cfg, dict) and dist_cfg.get('enabled', False):
        # Enable distillation training path
        cmd += ["--distillation_enabled"]

        # Logits express transport parameters
        le_cfg = dist_cfg['logits_express']
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
        gkd_cfg = dist_cfg['gkd']
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

    if config['rollout']['add_stop']:
        cmd += ["--add_stop"]

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
    if config['reward']['use_cot_reward']:
        cmd += ["--use_cot_reward"]
    if config['reward'].get('begin_of_thinking', None) is not None:
        cmd += [f"--begin_of_thinking '{config['reward']['begin_of_thinking']}'"]
    if config['reward']['force_thinking']:
        cmd += ["--force_thinking"]
    if config['algorithm']['use_kl_loss']:
        cmd += ["--use_kl_loss"]
    if config['training']['pad_to_max_length']:
        cmd += ["--pad_to_max_length"]
    if config['training']['use_outcome_rewards_as_advantages']:
        cmd += ["--use_outcome_rewards_as_advantages"]
    if config['training']['skip_zero_reward_sample']:
        cmd += ["--skip_zero_reward_sample"]

    if assignments["reference_nodes"] > 0:
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
    cmd += [f"--total_world_size {len(node_list) * 8}"]
    if config['algorithm'].get('entropy_loss_coef', None) is not None:
        cmd += [f"--entropy_loss_coef '{config['algorithm']['entropy_loss_coef']}'"]
    
    if config.get("sandbox_fusion_hostfile", None) is not None:
        assert os.path.exists(config["sandbox_fusion_hostfile"])
        cmd += [f"--sandbox_fusion_hostfile '{config['sandbox_fusion_hostfile']}'"]
    
    if config["training"].get("entropy_loss_threshold", None) is not None:
        cmd += [f"--entropy_loss_threshold {config['training']['entropy_loss_threshold']}"]

    if config["algorithm"].get("policy_loss_coef", None) is not None:
        cmd += [f"--policy_loss_coef {config['algorithm']['policy_loss_coef']}"]

    if config["algorithm"].get("negate_kl_loss", False):
        cmd += [f"--negate_kl_loss"]

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
        node_list[actor_master], 
        actor_nodes, 
        config["actor"]["weight_update_group_port"], 
        f"{config['training']['output_dir']}/actor_log"
    ) + cmd

    return " ".join(cmd)

def generate_weight_updater_test_cmd(
    config, 
    actor_master, 
    node_list, 
    ref_start,
    node_rank: int,
    actor_nodes: int,
    assignments: dict
):
    '''
        --use-checkpoint-args \
        --use-mp-args-from-checkpoint-args \
        --no-load-optim \
        --bf16 \
    '''
    output_dir = config['training']['output_dir']
    assert output_dir, "you must specify output directory"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        f"python3 {config['main']}.py",
        "--use-checkpoint-args",
        # "--use-mp-args-from-checkpoint-args",
        "--bf16",
        "--no-load-optim",
        "--sequence-parallel",
        f"--load {os.path.abspath(config['training']['from_pretrained'])}",
        f"--tensor-model-parallel-size {config['actor']['tp']}",
        f"--pipeline-model-parallel-size {config['actor']['pp']}",
        f"--context-parallel-size {config['actor']['cp']}",
    ]

    # moe config
    if config['actor'].get('moe', None) is not None:
        cmd += [f"--moe-grouped-gemm"]
        ep = config['actor']['moe']['ep']
        etp = config['actor']['moe']['etp']
        cmd += [f"--expert-model-parallel-size {ep}"]
        if config['actor']['moe'].get('etp', None) is not None:
            etp = config['actor']['moe']['etp']
            cmd += [f"--expert-tensor-parallel-size {etp}"]

    if config['training'].get('mlp_weight_merging_batch_size', None) is not None:
        cmd += [f"--mlp_weight_merging_batch_size {config['training']['mlp_weight_merging_batch_size']}"]
    if config['training'].get('moe_weight_merging_layer_batch_size', None) is not None:
        cmd += [f"--moe_weight_merging_layer_batch_size {config['training']['moe_weight_merging_layer_batch_size']}"]
    if config['training'].get('moe_weight_merging_expert_batch_size', None) is not None:
        cmd += [f"--moe_weight_merging_expert_batch_size {config['training']['moe_weight_merging_expert_batch_size']}"]
    
    if config['weight_updater_test']['online_test']:
        rollout_base_url_list = [f"http://{node_list[instance['node_rank']]}:{config['rollout']['port'] + instance['offset']}" for instance in assignments["rollout_instances"]]
        cmd += [
            f"--online_test",
            f"--rollout_base_url_list {' '.join(rollout_base_url_list)}",
            f"--rollout_model_name {config['training']['rollout_model_name']}",
            f"--server_world_size {config['rollout']['tp'] * config['rollout']['num_instances']}",
        ]
    else:
        safetensors_save_path = os.path.join(output_dir, "dumped_safetensors")
        os.makedirs(safetensors_save_path, exist_ok=True)
        cmd += [
            f"--safetensors_save_path {safetensors_save_path}"
        ]

    cmd = generate_dist_launcher_cmd(
        node_rank, 
        node_list[actor_master], 
        actor_nodes, 
        config["actor"]["weight_update_group_port"], 
        f"{config['training']['output_dir']}/actor_log"
    ) + cmd

    return " ".join(cmd)


def generate_ref_cmd(
    config, 
    node_list, 
    ref_start,
    node_rank: int,
    ref_nodes: int,
    actor_master: int
):
    cmd = [
        "python3 main_reference_model.py",
        f"--load {config['reference']['load_path']}",
        "--use-checkpoint-args",
        # "--use-mp-args-from-checkpoint-args",
        "--bf16",
        "--distributed-timeout-minutes 60",
        "--sequence-parallel",
        f"--actor_master_addr {node_list[actor_master]}",
        f"--data_transfer_group_port {config['actor']['weight_update_group_port']}",
        f"--tensor-model-parallel-size {config['reference']['tp']}",
        f"--pipeline-model-parallel-size {config['reference']['pp']}",
        f"--context-parallel-size {config['reference']['cp']}",
        f"--micro-batch-size {config['reference']['micro_batch_size']}",
        f"--global-batch-size {config['training']['global_batch_size']}",
        f"--light_scale_log_level {config['training']['log_level']}",
    ]

    # Distillation and GKD related flags for reference (required when enabled)
    dist_cfg = config.get('distillation')
    if isinstance(dist_cfg, dict) and dist_cfg.get('enabled', False):
        cmd += ["--distillation_enabled"]

        # Logits express transport parameters
        le_cfg = dist_cfg['logits_express']
        dtype_val = le_cfg['dtype']
        if dtype_val is not None:
            cmd += [f"--logits_dtype {dtype_val}"]
        bs_val = le_cfg['batch_size']
        if bs_val is not None:
            cmd += [f"--logits_transfer_batch_size {bs_val}"]
        base_port_val = le_cfg['base_port']
        if base_port_val is not None:
            cmd += [f"--logits_pg_base_port {base_port_val}"]

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
        node_rank - ref_start, 
        node_list[ref_start], 
        ref_nodes, 
        14470, 
        f"{config['training']['output_dir']}/ref_log"
    ) + cmd
    return " ".join(cmd)

def get_cuda_visible_devices(gpu_offset: int, num_gpus: int):
    gpus = [str(i) for i in range(gpu_offset, gpu_offset + num_gpus)]
    return ",".join(gpus)

def generate_sglang_cmd(
    config: Dict,
    node_list: List[str],
    assignments: dict,
    node_rank: int,
) -> str:
    """Generate sglang command for rollout"""
    assert config['training']['rollout_model_name'] and os.path.exists(config['training']['rollout_model_name']), "you must specify loading path for inference service"
    cmd_list = []
    rollout_instances = assignments["rollout_instances"]
    for instance in rollout_instances:
        if instance["node_rank"] > node_rank:
            break
        if instance["node_rank"] < node_rank:
            continue
        cmd = [
            f"CUDA_VISIBLE_DEVICES={get_cuda_visible_devices(instance['gpu_offset'], config['rollout']['tp'])}",
            "python3 -m sglang.launch_server",
            f"--port {config['rollout']['port'] + instance['offset']}",
            "--host 0.0.0.0",
            f"--model {config['training']['rollout_model_name']}",
            f"--tp {config['rollout']['tp']}",
            "--trust-remote-code",
            f"--dist-init-addr {node_list[node_rank]}:{18670 + instance['offset']}",
            "--nnodes 1",
            "--node-rank 0"
        ]
        if config['rollout']['enable_ep']:
            cmd += [f"--ep-size {config['rollout']['tp']}"]
        if config['rollout'].get('cpu_offload_gb', None) is not None:
            cmd += [f"--cpu-offload-gb {config['rollout']['cpu_offload_gb']}"]
        # Add extra arguments if any
        extra_args = config['rollout'].get("extra_args", [])
        for arg in extra_args:
            if arg["key"] and arg["value"] is None:
                cmd.append(f"--{arg['key']}")
            elif arg["key"] and arg["value"]:
                cmd.append(f"--{arg['key']} {arg['value']}")
        cmd_list.append(" ".join(cmd))
    
    return cmd_list

def generate_sft_cmd(
    config, 
    actor_master, 
    node_list, 
    node_rank: int,
    actor_nodes: int,
):
    output_dir = config['training']['output_dir']
    assert output_dir, "you must specify output directory"
    assert config['training']['data'] and os.path.exists(config['training']['data']), "you must specify valid training data path"
    assert config['training']['resume_checkpoint'] or config['training']['from_pretrained'], "you must specify checkpoint loading path"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_dir = f"{output_dir}/tensorboard_log"
    os.makedirs(tensorboard_dir, exist_ok=True)

    dp_size = config["actor"]["dp"]
    assert config["training"]["global_batch_size"] % dp_size == 0

    cmd = [
        f"python3 {config['main']}.py",
        "--use-checkpoint-args",
        # "--use-mp-args-from-checkpoint-args", # should not use this after switching to torch_dist
        "--bf16",
        "--use-distributed-optimizer",
        "--ckpt-format torch_dist",
        "--distributed-timeout-minutes 90",
        "--sequence-parallel",
        # "--calculate-per-token-loss",
        f"--save {os.path.abspath(checkpoint_dir)}",
        f"--tensorboard-dir {os.path.abspath(tensorboard_dir)}",
        f"--save-interval {config['training']['save_steps']}",
        f"--non-persistent-save-interval {config['training']['non_persistant_save_steps']}",
        f"--non-persistent-ckpt-type global",
        f"--save_test_step {config['training']['save_test_step']}",
        f"--data-path {config['training']['data']}",
        f"--micro-batch-size {config['training']['micro_batch_size']}",
        f"--global-batch-size {config['training']['global_batch_size']}",
        f"--train-iters {config['training']['max_steps']}",
        f"--lr {config['training']['optim']['lr']}",
        f"--lr-decay-style {config['training']['optim']['scheduler_type']}",
        f"--min-lr {config['training']['optim']['min_lr']}",
        f"--seq-length {config['training']['max_length']}",
        f"--seed {config['training']['seed']}",
        f"--tensor-model-parallel-size {config['actor']['tp']}",
        f"--pipeline-model-parallel-size {config['actor']['pp']}",
        f"--context-parallel-size {config['actor']['cp']}",
        f"--weight-decay {config['training']['optim']['weight_decay']}",
        f"--light_scale_log_level {config['training']['log_level']}",
        f"--early_stop_steps {config['training']['early_stop_steps']}",
        f"--adam-beta1 {config['training']['optim']['adam_beta1']}",
        f"--adam-beta2 {config['training']['optim']['adam_beta2']}",
        f"--adam-eps {config['training']['optim']['adam_eps']}"
    ]

    if config['training']['shuffle']:
        cmd += [
            f"--sft_data_shuffle"
        ]

    if config['training']['resume_checkpoint'] is None:
        cmd += [
            f"--load {os.path.abspath(config['training']['from_pretrained'])}",
            f"--finetune"
        ]
    else:
        cmd += [
            f"--load {os.path.abspath(config['training']['resume_checkpoint'])}",
        ]

    # set warmup step or ratio
    if config['training']['optim'].get('warmup_steps', None) is not None:
        cmd += [f"--lr-warmup-iters {config['training']['optim']['warmup_steps']}"]
    else:
        cmd += [f"--lr-warmup-fraction {config['training']['optim']['warmup_ratio']}"]

    # packing
    if config['training']['sequence_packing']:
        cmd += ["--sequence_packing"]

    # moe config
    if config['actor'].get('moe', None) is not None:
        cmd += [f"--moe-grouped-gemm"]
        ep = config['actor']['moe']['ep']
        cmd += [f"--expert-model-parallel-size {ep}"]
        if config['actor']['moe'].get('etp', None) is not None:
            etp = config['actor']['moe']['etp']
            cmd += [f"--expert-tensor-parallel-size {etp}"]

    # Add extra arguments if any
    extra_args = config['actor'].get("extra_args", [])
    for arg in extra_args:
        if arg["key"] and arg["value"] is None:
            cmd.append(f"--{arg['key']}")
        elif arg["key"] and arg["value"]:
            cmd.append(f"--{arg['key']} {arg['value']}")

    cmd = generate_dist_launcher_cmd(
        node_rank, 
        node_list[actor_master], 
        actor_nodes, 
        config["actor"]["master_port"], 
        f"{config['training']['output_dir']}/train_log"
    ) + cmd

    return " ".join(cmd)

def main():
    args = parse_args()
    config = load_config(args.config)
    node_list = get_node_list(args)
    if node_list is None or len(node_list) == 0:
        raise RuntimeError("node list is empty")
    print(node_list)
    node_rank = get_node_rank(args, node_list)
    if node_rank < 0:
        raise RuntimeError("node rank is not provided")
    total_nodes = len(node_list)
    print(f"{total_nodes} nodes in total, this node rank: {node_rank}")

    offline_testing = False
    if config.get('weight_updater_test', None) is not None and config['weight_updater_test']['online_test'] is False:
        offline_testing = True
    
    # Determine which roles should run on this node
    assignments = calculate_role_assignment(config, node_rank, total_nodes, offline_testing=offline_testing)

    print(assignments)

    assert assignments["actor_nodes"] + assignments["reference_nodes"] + assignments["rollout_nodes"] == total_nodes, \
        f"Required {assignments['actor_nodes']} actor nodes, {assignments['reference_nodes']} reference nodes, {assignments['rollout_nodes']} rollout nodes, provided {total_nodes} nodes"
    
    # Generate commands for this node
    commands = []
    
    # Actor command
    if assignments["run_actor"]:
        if config['main'] == 'main_sft':
            actor_cmd = generate_sft_cmd(
                config=config,
                actor_master=assignments["actor_master"],
                node_rank=node_rank,
                node_list=node_list,
                actor_nodes=assignments["actor_nodes"]
            )
        elif config.get('weight_updater_test', None) is None:
            actor_cmd = generate_actor_cmd(
                config=config,
                actor_master=assignments["actor_master"],
                node_rank=node_rank,
                node_list=node_list,
                ref_start=assignments["reference_start"] if assignments["reference_nodes"] > 0 else -1,
                actor_nodes=assignments["actor_nodes"],
                assignments=assignments
            )
        else:
            actor_cmd = generate_weight_updater_test_cmd(
                config=config,
                actor_master=assignments["actor_master"],
                node_rank=node_rank,
                node_list=node_list,
                ref_start=assignments["reference_start"] if assignments["reference_nodes"] > 0 else -1,
                actor_nodes=assignments["actor_nodes"],
                assignments=assignments
            )
        commands.append(actor_cmd)
    
    # Reference command
    if assignments["run_reference"]:
        ref_cmd = generate_ref_cmd(
            config=config,
            node_rank=node_rank,
            node_list=node_list,
            ref_nodes=assignments["reference_nodes"],
            ref_start=assignments["reference_start"],
            actor_master=assignments["actor_master"]
        )
        commands.append(ref_cmd)
    
    # Rollout command
    if assignments["run_rollout"]:
        rollout_cmd_list = generate_sglang_cmd(
            config=config,
            node_list=node_list,
            assignments=assignments,
            node_rank=node_rank
        )
        commands = rollout_cmd_list
    
    print(commands)
    # Print commands that should be executed on this node
    if commands:
        print("\n# Commands to run:")
        print("\n".join(commands))
        run_commands(commands)
    else:
        raise RuntimeError("# No roles assigned to this node")

if __name__ == "__main__":
    main()