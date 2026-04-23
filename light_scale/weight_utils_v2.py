from megatron.training.initialize import initialize_megatron
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from megatron.core.enums import ModelType

from megatron.training.checkpointing import load_checkpoint
from pretrain_gpt import model_provider
import torch.distributed as dist
from megatron.core import mpu
import json
from tools.checkpoint.schema_core import get_model_schema
from tools.checkpoint.layer_action import LayerAction
from megatron.training.global_vars import get_args
import torch
from light_scale import dist_utils
from light_scale.param_mapping import get_dense_param_mapping, get_moe_param_mapping
from light_scale.sync_processor import SafetensorsSaver, SGLangSaver
import time
from light_scale.logger_utils import setup_logger

from typing import List, Optional, Tuple, Union

class DenseWeightUpdater:
    def __init__(self, model, sync_processor: Union[SafetensorsSaver|SGLangSaver]):
        self.model = model
        self.sync_processor = sync_processor
        self.logger = setup_logger("light_scale")
    
    @torch.no_grad()
    def __call__(self):
        # only pp[:]dp[0]cp[0]tp[:] should enter this func
        model = self.model
        logger = self.logger
        saver = self.sync_processor
        margs = get_args()
        schema = get_model_schema(
            'GPT',
            'transformer_engine',
            num_experts=margs.num_experts,
            expert_model_parallel_size=margs.expert_model_parallel_size
        )
        logger.debug("schema loaded")
        dist_utils.wait_for_pp_and_tp_neighbors()
        unwrapped_model = unwrap_model(model)[0]

        num_layers = schema.get_num_layers(unwrapped_model)
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # num_layers = len(model[0].decoder.layers)
        logger.debug(num_layers)
        local_num_layer = torch.tensor([num_layers], dtype=torch.int32, device=torch.device("cuda", torch.cuda.current_device()))
        global_num_layer_list = [torch.empty_like(local_num_layer) for _ in range(mpu.get_pipeline_model_parallel_world_size())]
        dist.all_gather(global_num_layer_list, local_num_layer, group=mpu.get_pipeline_model_parallel_group())
        global_num_layer = torch.cat(global_num_layer_list, dim=0)
        # assert torch.max(global_num_layer) == torch.min(global_num_layer), \
        #     "This function assumes that model layers are evenly distributed across pipeline parallelism stages"
        global_num_layer_prefix_sum = torch.cumsum(global_num_layer, dim=0)
        global_num_layer_prefix_sum = torch.cat((torch.zeros(size=(1,), dtype=global_num_layer_prefix_sum.dtype, device=global_num_layer_prefix_sum.device), 
                                                global_num_layer_prefix_sum), dim=0)
        if dist.get_rank() == 0:
            logger.debug(global_num_layer)
            logger.debug(global_num_layer_prefix_sum)

        global_num_layer = global_num_layer.detach().cpu().numpy()
        multi_latent_attention = getattr(margs, "multi_latent_attention", False)
        q_lora_rank = getattr(margs, "q_lora_rank", None)
        has_moe_layers = margs.num_experts is not None and margs.num_experts > 0
        shared_expert_hidden = getattr(margs, "moe_shared_expert_intermediate_size", None)
        has_shared_experts = has_moe_layers and shared_expert_hidden not in (None, 0)

        def get_weights_in_local_layers(weight_name: str):
            local_weights = []
            for layer_num in range(num_layers):
                layer = schema.get_layer(unwrapped_model, layer_num)
                local_weight = layer[weight_name]
                local_weights.append((layer_num, local_weight))
            return local_weights

        # Schema tensors are the source of truth; configs must guarantee required params exist.
        def require_layer_value(layer_params, key: str, layer_num: int):
            value = layer_params[key]
            if value is None:
                raise RuntimeError(f"Layer {layer_num} is missing required tensor '{key}'. Please ensure the schema and config are aligned.")
            return value

        def gather_and_merge_weights_in_local_layers(weight_name: str, merge_dim: int, num_layers_per_batch: int = num_layers, layer_offset: int = 0, layer_action=LayerAction.NoAction):
            merged_local_weights = []
            layer_num = layer_offset
            while layer_num < min(layer_offset + num_layers_per_batch, num_layers):
                layer = schema.get_layer(unwrapped_model, layer_num, layer_action=layer_action)
                if layer is None:
                    self.logger.debug(f"gather_and_merge_weights_in_local_layers, layer {layer_num} is None")
                    layer_num += 1
                    continue
                local_weight = layer[weight_name]
                merged_weight = dist_utils.gather_and_merge_tp_sharded_weight(local_weight, merge_dim=merge_dim)
                merged_local_weights.append((layer_num, merged_weight))
                layer_num += 1
            return merged_local_weights

        head_num = margs.num_attention_heads
        num_query_groups = margs.num_query_groups
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups
        head_size = margs.kv_channels
        hidden_size = margs.hidden_size
        ffn_hidden_size = margs.ffn_hidden_size

        def split_gather_and_merge_qkv_weights_in_local_layers(margs=margs):
            merged_q_weights = []
            merged_k_weights = []
            merged_v_weights = []

            q_slice = torch.cat(
                [
                    torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                    for i in range(num_query_groups)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
            v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
            for layer_num in range(num_layers):
                layer = schema.get_layer(unwrapped_model, layer_num)
                local_qkv_weight = layer["self_attn_qkv_weight"]
                merged_qkv_weight = dist_utils.gather_and_merge_tp_sharded_weight(local_qkv_weight, merge_dim=0)
                if mpu.get_tensor_model_parallel_rank() == 0:
                    merged_qkv_weight = merged_qkv_weight.reshape([qkv_total_dim, head_size, hidden_size])
                    merged_q_weight = merged_qkv_weight[q_slice].reshape(-1, hidden_size)
                    merged_k_weight = merged_qkv_weight[k_slice].reshape(-1, hidden_size)
                    merged_v_weight = merged_qkv_weight[v_slice].reshape(-1, hidden_size)
                    merged_q_weights.append((layer_num, merged_q_weight))
                    merged_k_weights.append((layer_num, merged_k_weight))
                    merged_v_weights.append((layer_num, merged_v_weight))

            return merged_q_weights, merged_k_weights, merged_v_weights

        def split_gather_and_merge_qkv_biases_in_local_layers(margs=margs):
            merged_q_weights = []
            merged_k_weights = []
            merged_v_weights = []

            q_slice = torch.cat(
                [
                    torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                    for i in range(num_query_groups)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
            v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
            for layer_num in range(num_layers):
                layer = schema.get_layer(unwrapped_model, layer_num)
                local_qkv_weight = layer["self_attn_qkv_bias"]
                merged_qkv_weight = dist_utils.gather_and_merge_tp_sharded_weight(local_qkv_weight, merge_dim=0)
                if mpu.get_tensor_model_parallel_rank() == 0:
                    merged_qkv_weight = merged_qkv_weight.reshape([qkv_total_dim, head_size])
                    merged_q_weight = merged_qkv_weight[q_slice].reshape(-1,)
                    merged_k_weight = merged_qkv_weight[k_slice].reshape(-1,)
                    merged_v_weight = merged_qkv_weight[v_slice].reshape(-1,)
                    merged_q_weights.append((layer_num, merged_q_weight))
                    merged_k_weights.append((layer_num, merged_k_weight))
                    merged_v_weights.append((layer_num, merged_v_weight))

            return merged_q_weights, merged_k_weights, merged_v_weights

        def split_gather_and_merge_gate_up_in_local_layers(num_layers_per_batch: int, layer_offset: int):
            self.logger.debug(f"split_gather_and_merge_gate_up_in_local_layers, layer_offset: {layer_offset}")
            merged_gate_weights = []
            merged_up_weights = []

            sharded_intermediate_size = margs.ffn_hidden_size // mpu.get_tensor_model_parallel_world_size()

            layer_num = layer_offset
            while layer_num < min(num_layers, num_layers_per_batch + layer_offset):
                self.logger.debug(f"split_gather_and_merge_gate_up_in_local_layers, itering layer_num: {layer_num}")
                layer = schema.get_layer(unwrapped_model, layer_num, layer_action=LayerAction.DenseOnly)
                if layer is None:
                    self.logger.debug(f"split_gather_and_merge_gate_up_in_local_layers, layer {layer_num} is None")
                    layer_num += 1
                    continue
                local_gate_up_weight = layer["mlp_fc1_weight"]
                assert local_gate_up_weight.shape == (2 * sharded_intermediate_size, margs.hidden_size)
                
                local_gate_weight = local_gate_up_weight[:sharded_intermediate_size, :]
                local_up_weight = local_gate_up_weight[sharded_intermediate_size:, :]

                merged_gate_weight = dist_utils.gather_and_merge_tp_sharded_weight(local_gate_weight, merge_dim=0)
                merged_up_weight = dist_utils.gather_and_merge_tp_sharded_weight(local_up_weight, merge_dim=0)

                if mpu.get_tensor_model_parallel_rank() == 0:
                    assert merged_gate_weight.shape == (margs.ffn_hidden_size, margs.hidden_size), f"{merged_gate_weight.shape}"
                    assert merged_up_weight.shape == (margs.ffn_hidden_size, margs.hidden_size), f"{merged_up_weight.shape}"

                merged_gate_weights.append((layer_num, merged_gate_weight))
                merged_up_weights.append((layer_num, merged_up_weight))

                layer_num += 1

            return merged_gate_weights, merged_up_weights

        def gather_mla_attention_weights():
            q_main_weights = []
            q_up_weights = []
            kv_down_weights = []
            kv_up_weights = []
            q_norm_weights = []
            kv_norm_weights = []

            for layer_num in range(num_layers):
                layer = schema.get_layer(unwrapped_model, layer_num)

                if q_lora_rank is not None:
                    local_q_down = require_layer_value(layer, "self_attn_mla_q_down_weight", layer_num)
                    merged_q_down = dist_utils.gather_and_merge_tp_sharded_weight(local_q_down, merge_dim=0)
                    q_main_weights.append((layer_num, merged_q_down))

                    local_q_up = require_layer_value(layer, "self_attn_mla_q_up_weight", layer_num)
                    merged_q_up = dist_utils.gather_and_merge_tp_sharded_weight(local_q_up, merge_dim=0)
                    q_up_weights.append((layer_num, merged_q_up))

                    layer_norm_weight = layer["self_attn_mla_q_norm_weight"]
                    if layer_norm_weight is not None:
                        q_norm_weights.append((layer_num, layer_norm_weight.clone()))
                else:
                    local_q_proj = require_layer_value(layer, "self_attn_mla_q_proj_weight", layer_num)
                    merged_q_proj = dist_utils.gather_and_merge_tp_sharded_weight(local_q_proj, merge_dim=0)
                    q_main_weights.append((layer_num, merged_q_proj))

                local_kv_down = require_layer_value(layer, "self_attn_mla_kv_down_weight", layer_num)
                merged_kv_down = dist_utils.gather_and_merge_tp_sharded_weight(local_kv_down, merge_dim=0)
                kv_down_weights.append((layer_num, merged_kv_down))

                local_kv_up = require_layer_value(layer, "self_attn_mla_kv_up_weight", layer_num)
                merged_kv_up = dist_utils.gather_and_merge_tp_sharded_weight(local_kv_up, merge_dim=0)
                kv_up_weights.append((layer_num, merged_kv_up))

                layer_norm_weight = layer["self_attn_mla_kv_norm_weight"]
                if layer_norm_weight is not None:
                    kv_norm_weights.append((layer_num, layer_norm_weight.clone()))

            return q_main_weights, q_up_weights, kv_down_weights, kv_up_weights, q_norm_weights, kv_norm_weights

        def split_gather_and_merge_shared_experts(num_layers_per_batch: int, layer_offset: int):
            merged_shared_gate = []
            merged_shared_up = []
            merged_shared_down = []
            merged_shared_router = []

            layer_num = layer_offset
            while layer_num < min(num_layers, num_layers_per_batch + layer_offset):
                layer = schema.get_layer(unwrapped_model, layer_num, layer_action=LayerAction.MoeOnly)
                if layer is None:
                    layer_num += 1
                    continue

                local_shared_fc1 = require_layer_value(layer, "shared_experts_fc1_weight", layer_num)
                sharded_intermediate_size = local_shared_fc1.shape[0] // 2
                local_shared_gate = local_shared_fc1[:sharded_intermediate_size, :]
                local_shared_up = local_shared_fc1[sharded_intermediate_size:, :]

                merged_shared_gate.append((layer_num, dist_utils.gather_and_merge_tp_sharded_weight(local_shared_gate, merge_dim=0)))
                merged_shared_up.append((layer_num, dist_utils.gather_and_merge_tp_sharded_weight(local_shared_up, merge_dim=0)))

                local_shared_down = require_layer_value(layer, "shared_experts_fc2_weight", layer_num)
                merged_shared_down.append((layer_num, dist_utils.gather_and_merge_tp_sharded_weight(local_shared_down, merge_dim=1)))

                shared_expert_gate_weight = layer["shared_experts_gate_weight"]
                if shared_expert_gate_weight is not None:
                    merged_shared_router.append((layer_num, shared_expert_gate_weight.clone()))

                layer_num += 1

            return merged_shared_gate, merged_shared_up, merged_shared_down, merged_shared_router

        def get_global_layer_number(pp_rank: int, local_layer_num: int):
            global_layer_num = global_num_layer_prefix_sum[pp_rank] + local_layer_num
            return global_layer_num

        def tp0_sync_layer_weights(merged_local_layer_weights: List[Tuple[int, torch.Tensor]],
                                   layer_weight_name: str,
                                   post_fix: Optional[str] = None,
                                   layer_offset: int = 0,
                                   num_layers_per_batch: int = None):
            # merged_local_layer_weights: List[Tuple[local_layer_num, param Tensor]]
            self.logger.debug(f"tp0_sync_layer_weights layer offset: {layer_offset}")
            assert "{layer_num}" in layer_weight_name
            for local_layer_num, local_weight in merged_local_layer_weights:
                global_layer_number = get_global_layer_number(pp_rank, local_layer_num)
                weight_name = layer_weight_name.format(layer_num=global_layer_number)
                if post_fix is not None:
                    weight_name += f".{post_fix}"
                saver.add(
                    weight_name=weight_name,
                    weight=local_weight
                )
            saver.commit()
            del merged_local_layer_weights
            torch.cuda.empty_cache()

        param_mapping = get_dense_param_mapping()

        torch.cuda.empty_cache()
        start_time = time.time()
        dtype = None

        # word embeddings
        if mpu.is_pipeline_first_stage():
            local_word_embedding = schema.get("embeddings", unwrapped_model)["word"]
            dtype = local_word_embedding.dtype
            merged_word_embeddings = dist_utils.gather_and_merge_tp_sharded_weight(local_word_embedding, merge_dim=0)
            if mpu.get_tensor_model_parallel_rank() == 0:
                # sync with vllm
                saver.add(weight_name=param_mapping["embeddings"], weight=merged_word_embeddings)
                saver.commit()
                del merged_word_embeddings
                torch.cuda.empty_cache()
        dist_utils.wait_for_tp_neighbors()

        mla_q_norm_weights = []
        mla_kv_norm_weights = []
        if multi_latent_attention:
            (merged_mla_q_weights,
             merged_mla_q_up_weights,
             merged_mla_kv_down_weights,
             merged_mla_kv_up_weights,
             mla_q_norm_weights,
             mla_kv_norm_weights) = gather_mla_attention_weights()

            if mpu.get_tensor_model_parallel_rank() == 0:
                if len(merged_mla_q_weights) > 0:
                    q_mapping_key = "mla_q_down" if q_lora_rank is not None else "query"
                    tp0_sync_layer_weights(merged_mla_q_weights, param_mapping[q_mapping_key], post_fix="weight")
                if q_lora_rank is not None and len(merged_mla_q_up_weights) > 0:
                    tp0_sync_layer_weights(merged_mla_q_up_weights, param_mapping["mla_q_up"], post_fix="weight")
                if len(merged_mla_kv_down_weights) > 0:
                    tp0_sync_layer_weights(merged_mla_kv_down_weights, param_mapping["mla_kv_down"], post_fix="weight")
                if len(merged_mla_kv_up_weights) > 0:
                    tp0_sync_layer_weights(merged_mla_kv_up_weights, param_mapping["mla_kv_up"], post_fix="weight")
                del merged_mla_q_weights
                del merged_mla_q_up_weights
                del merged_mla_kv_down_weights
                del merged_mla_kv_up_weights
                torch.cuda.empty_cache()
            dist_utils.wait_for_tp_neighbors()
        else:
            # attention qkv weight
            merged_local_q_weights, merged_local_k_weights, merged_local_v_weights = split_gather_and_merge_qkv_weights_in_local_layers()
            if mpu.get_tensor_model_parallel_rank() == 0:
                tp0_sync_layer_weights(merged_local_q_weights, param_mapping["query"], post_fix="weight")
                del merged_local_q_weights
                tp0_sync_layer_weights(merged_local_k_weights, param_mapping["key"], post_fix="weight")
                del merged_local_k_weights
                tp0_sync_layer_weights(merged_local_v_weights, param_mapping["value"], post_fix="weight")
                del merged_local_v_weights
                torch.cuda.empty_cache()
            dist_utils.wait_for_tp_neighbors()

            if margs.add_qkv_bias:
                # attention qkv bias
                merged_local_q_biases, merged_local_k_biases, merged_local_v_biases = split_gather_and_merge_qkv_biases_in_local_layers()
                if mpu.get_tensor_model_parallel_rank() == 0:
                    tp0_sync_layer_weights(merged_local_q_biases, param_mapping["query"], post_fix="bias")
                    del merged_local_q_biases
                    tp0_sync_layer_weights(merged_local_k_biases, param_mapping["key"], post_fix="bias")
                    del merged_local_k_biases
                    tp0_sync_layer_weights(merged_local_v_biases, param_mapping["value"], post_fix="bias")
                    del merged_local_v_biases
                    torch.cuda.empty_cache()
                dist_utils.wait_for_tp_neighbors()

        # attention output, self_attn_proj_weight
        merged_local_output_weights = gather_and_merge_weights_in_local_layers("self_attn_proj_weight", merge_dim=1)
        if mpu.get_tensor_model_parallel_rank() == 0:
            tp0_sync_layer_weights(merged_local_output_weights, param_mapping["output"])
            del merged_local_output_weights
            torch.cuda.empty_cache()
        dist_utils.wait_for_tp_neighbors()

        # attention layernorm
        local_attention_norm_weights = get_weights_in_local_layers("self_attn_norm_weight")
        if mpu.get_tensor_model_parallel_rank() == 0:
            tp0_sync_layer_weights(local_attention_norm_weights, param_mapping["attention_norm"])
            del local_attention_norm_weights
            torch.cuda.empty_cache()
        dist_utils.wait_for_tp_neighbors()

        # qk norm
        self.logger.debug(f"qk norm: {margs.qk_layernorm}")
        if multi_latent_attention:
            if margs.qk_layernorm and mpu.get_tensor_model_parallel_rank() == 0:
                if q_lora_rank is not None and len(mla_q_norm_weights) > 0:
                    tp0_sync_layer_weights(mla_q_norm_weights, param_mapping["mla_q_norm"])
                if len(mla_kv_norm_weights) > 0:
                    tp0_sync_layer_weights(mla_kv_norm_weights, param_mapping["mla_kv_norm"])
                del mla_q_norm_weights
                del mla_kv_norm_weights
                torch.cuda.empty_cache()
            dist_utils.wait_for_tp_neighbors()
        else:
            if margs.qk_layernorm:
                local_q_norm_weights = get_weights_in_local_layers("self_attn_q_norm")
                local_k_norm_weights = get_weights_in_local_layers("self_attn_k_norm")
                if mpu.get_tensor_model_parallel_rank() == 0:
                    tp0_sync_layer_weights(local_q_norm_weights, param_mapping["q_norm"])
                    tp0_sync_layer_weights(local_k_norm_weights, param_mapping["k_norm"])
                    torch.cuda.empty_cache()
            dist_utils.wait_for_tp_neighbors()

        # mlp gate_up weight
        num_layers_per_batch = margs.mlp_weight_merging_batch_size if hasattr(margs, "mlp_weight_merging_batch_size") and margs.mlp_weight_merging_batch_size > 0 else num_layers
        layer_offset = 0
        while layer_offset < num_layers:
            merged_local_gate_weights, merged_local_up_weights = split_gather_and_merge_gate_up_in_local_layers(num_layers_per_batch, layer_offset)
            assert len(merged_local_gate_weights) <= num_layers_per_batch
            if mpu.get_tensor_model_parallel_rank() == 0 and len(merged_local_gate_weights) > 0:
                tp0_sync_layer_weights(merged_local_gate_weights, param_mapping["gate"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch)
                del merged_local_gate_weights
                torch.cuda.empty_cache()
                tp0_sync_layer_weights(merged_local_up_weights, param_mapping["up"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch)
                del merged_local_up_weights
                torch.cuda.empty_cache()
            dist_utils.wait_for_tp_neighbors()
            layer_offset += num_layers_per_batch

        # mlp down weight
        layer_offset = 0
        while layer_offset < num_layers:
            merged_local_down_weights = gather_and_merge_weights_in_local_layers("mlp_fc2_weight", merge_dim=1, 
                                                                                 num_layers_per_batch=num_layers_per_batch, layer_offset=layer_offset, layer_action=LayerAction.DenseOnly)
            if mpu.get_tensor_model_parallel_rank() == 0 and len(merged_local_down_weights) > 0:
                tp0_sync_layer_weights(merged_local_down_weights, param_mapping["down"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch)
                del merged_local_down_weights
                torch.cuda.empty_cache()
            dist_utils.wait_for_tp_neighbors()
            layer_offset += num_layers_per_batch

        if has_shared_experts:
            layer_offset = 0
            while layer_offset < num_layers:
                (merged_shared_gate_weights,
                 merged_shared_up_weights,
                 merged_shared_down_weights,
                 merged_shared_router_weights) = split_gather_and_merge_shared_experts(num_layers_per_batch, layer_offset)

                if mpu.get_tensor_model_parallel_rank() == 0:
                    if len(merged_shared_gate_weights) > 0:
                        tp0_sync_layer_weights(merged_shared_gate_weights, param_mapping["shared_gate"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch)
                        del merged_shared_gate_weights
                        torch.cuda.empty_cache()
                    if len(merged_shared_up_weights) > 0:
                        tp0_sync_layer_weights(merged_shared_up_weights, param_mapping["shared_up"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch)
                        del merged_shared_up_weights
                        torch.cuda.empty_cache()
                    if len(merged_shared_down_weights) > 0:
                        tp0_sync_layer_weights(merged_shared_down_weights, param_mapping["shared_down"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch)
                        del merged_shared_down_weights
                        torch.cuda.empty_cache()
                    if len(merged_shared_router_weights) > 0:
                        tp0_sync_layer_weights(merged_shared_router_weights, param_mapping["shared_router"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch)
                        del merged_shared_router_weights
                        torch.cuda.empty_cache()
                dist_utils.wait_for_tp_neighbors()
                layer_offset += num_layers_per_batch

        # mlp layernorm
        if margs.num_experts is not None and margs.num_experts > 0:
            local_mlp_norm_weights = get_weights_in_local_layers("moe_mlp_norm_weight")
        else:
            local_mlp_norm_weights = get_weights_in_local_layers("mlp_norm_weight")
        if mpu.get_tensor_model_parallel_rank() == 0:
            tp0_sync_layer_weights(local_mlp_norm_weights, param_mapping["mlp_norm"])
            del local_mlp_norm_weights
            torch.cuda.empty_cache()
        dist_utils.wait_for_tp_neighbors()

        # mlp router
        if margs.num_experts is not None and margs.num_experts > 0:
            local_mlp_router_weights = get_weights_in_local_layers("router_weight")
            if mpu.get_tensor_model_parallel_rank() == 0:
                tp0_sync_layer_weights(local_mlp_router_weights, param_mapping["router"])
                del local_mlp_router_weights
                torch.cuda.empty_cache()
            dist_utils.wait_for_tp_neighbors()

        # final layernorm and lm_head
        merged_lm_head = None
        final_norm_weight = None
        if mpu.is_pipeline_last_stage():
            local_lm_head = schema.get("output_layer", unwrapped_model)["weight"]
            merged_lm_head = dist_utils.gather_and_merge_tp_sharded_weight(local_lm_head, merge_dim=0)
            if mpu.get_tensor_model_parallel_rank() == 0:
                final_norm_weight = schema.get("final_norm", unwrapped_model)["weight"]
                saver.add(weight_name=param_mapping["lm_head"], weight=merged_lm_head)
                saver.add(weight_name=param_mapping["final_norm"], weight=final_norm_weight)
                saver.commit()
                del merged_lm_head
                del final_norm_weight
                torch.cuda.empty_cache()
        dist_utils.wait_for_tp_neighbors()

        dist_utils.wait_for_pp_and_tp_neighbors()
        end_time = time.time()
        if dist.get_rank() == 0:
            logger.info(f"total consumed time: {end_time - start_time} seconds")

class MoeWeightUpdater:
    def __init__(self, model, sync_processor: Union[SafetensorsSaver|SGLangSaver]):
        self.model = model
        self.sync_processor = sync_processor
        self.logger = setup_logger("light_scale")
        ep_src_rank, etp_src_rank = dist_utils.get_ep_and_etp_src_rank()
        self.logger.debug(f"ep_src_rank: {ep_src_rank}, etp_src_rank: {etp_src_rank}")
        if mpu.get_expert_model_parallel_rank() == 0:
            assert dist.get_rank() == ep_src_rank
        if mpu.get_expert_tensor_parallel_rank() == 0:
            assert dist.get_rank() == etp_src_rank
        self.ep_src_rank = ep_src_rank
        self.etp_src_rank = etp_src_rank
    
    @torch.no_grad()
    def __call__(self):
        # only pp[:]mdp[0]ep[:]mtp[:] should enter this func
        model = self.model
        logger = self.logger
        saver = self.sync_processor
        margs = get_args()
        schema = get_model_schema(
            'GPT',
            'transformer_engine',
            num_experts=margs.num_experts,
            expert_model_parallel_size=margs.expert_model_parallel_size
        )
        logger.debug("schema loaded")
        dist_utils.wait_for_pp_ep_and_etp_neighbors()

        unwrapped_model = unwrap_model(model)[0]
        num_layers = schema.get_num_layers(unwrapped_model)
        num_experts = margs.num_experts // margs.expert_model_parallel_size
        ep_experts_offset = mpu.get_expert_model_parallel_rank() * num_experts

        pp_rank = mpu.get_pipeline_model_parallel_rank()

        local_num_layer = torch.tensor([num_layers], dtype=torch.int32, device=torch.device("cuda", torch.cuda.current_device()))
        global_num_layer_list = [torch.empty_like(local_num_layer) for _ in range(mpu.get_pipeline_model_parallel_world_size())]
        dist.all_gather(global_num_layer_list, local_num_layer, group=mpu.get_pipeline_model_parallel_group())
        global_num_layer = torch.cat(global_num_layer_list, dim=0)
        # assert torch.max(global_num_layer) == torch.min(global_num_layer), \
        #     "This function assumes that model layers are evenly distributed across pipeline parallelism stages"
        global_num_layer_prefix_sum = torch.cumsum(global_num_layer, dim=0)
        global_num_layer_prefix_sum = torch.cat((torch.zeros(size=(1,), dtype=global_num_layer_prefix_sum.dtype, device=global_num_layer_prefix_sum.device), 
                                                global_num_layer_prefix_sum), dim=0)

        def split_gather_and_merge_gate_up_in_local_layers(num_layers_per_batch: int, layer_offset: int, num_experts_per_batch: int, expert_offset: int):
            self.logger.debug(f"split_gather_and_merge_gate_up_in_local_layers, layer_offset: {layer_offset}, expert_offset: {expert_offset}")
            merged_layer_gate_weights = [] # List[Tuple[int, List[Tuple[int, Tensor]]]], [(layer_id, [(expert_id, weight)])]
            merged_layer_up_weights = []

            ffn_hidden_size = margs.moe_ffn_hidden_size if margs.moe_ffn_hidden_size is not None else margs.ffn_hidden_size
            sharded_intermediate_size = ffn_hidden_size // mpu.get_expert_tensor_parallel_world_size()

            layer_num = layer_offset
            while layer_num < min(num_layers, num_layers_per_batch + layer_offset):
                self.logger.debug(f"split_gather_and_merge_gate_up_in_local_layers, itering layer_num: {layer_num}")
                layer = schema.get_layer(unwrapped_model, layer_num, layer_action=LayerAction.MoeOnly)
                if layer is None:
                    layer_num += 1
                    continue
                local_expert_gate_weights = []
                local_expert_up_weights = []
                expert_num = expert_offset
                while expert_num < min(num_experts, num_experts_per_batch + expert_offset):
                    self.logger.debug(f"split_gather_and_merge_gate_up_in_local_layers, itering expert id: {expert_num}")
                    local_gate_up_weight = layer[f"mlp_fc1_weight.{expert_num}"]
                    assert local_gate_up_weight.shape == (2 * sharded_intermediate_size, margs.hidden_size)
                    
                    local_gate_weight = local_gate_up_weight[:sharded_intermediate_size, :]
                    local_up_weight = local_gate_up_weight[sharded_intermediate_size:, :]

                    local_expert_gate_weights.append((expert_num + ep_experts_offset, local_gate_weight))
                    local_expert_up_weights.append((expert_num + ep_experts_offset, local_up_weight))

                    expert_num += 1

                expert_ids, merged_expert_gate_weights_stacked = dist_utils.stack_gather_and_merge_mtp_sharded_expert_weights(
                    local_expert_gate_weights, merge_dim=0, etp_src_rank=self.etp_src_rank
                )
                _, merged_expert_up_weights_stacked = dist_utils.stack_gather_and_merge_mtp_sharded_expert_weights(
                    local_expert_up_weights, merge_dim=0, etp_src_rank=self.etp_src_rank
                )

                merged_expert_gate_weights = None
                merged_expert_up_weights = None
                if mpu.get_expert_tensor_parallel_rank() == 0:
                    assert merged_expert_gate_weights_stacked.shape == (len(expert_ids), ffn_hidden_size, margs.hidden_size), f"{merged_expert_gate_weights_stacked.shape}"
                    assert merged_expert_up_weights_stacked.shape == (len(expert_ids), ffn_hidden_size, margs.hidden_size), f"{merged_expert_up_weights_stacked.shape}"
                    merged_expert_gate_weights = dist_utils.gather_and_unbind_expert_weights(expert_ids, merged_expert_gate_weights_stacked, ep_src_rank=self.ep_src_rank)
                    merged_expert_up_weights = dist_utils.gather_and_unbind_expert_weights(expert_ids, merged_expert_up_weights_stacked, ep_src_rank=self.ep_src_rank)


                self.logger.debug(f"merged_layer_gate_weights.append: {layer_num}")
                merged_layer_gate_weights.append((layer_num, merged_expert_gate_weights))
                merged_layer_up_weights.append((layer_num, merged_expert_up_weights))

                layer_num += 1

            return merged_layer_gate_weights, merged_layer_up_weights
        
        def gather_and_merge_down_in_local_layers(num_layers_per_batch: int, layer_offset: int, num_experts_per_batch: int, expert_offset: int):
            self.logger.debug(f"gather_and_merge_down_in_local_layers, layer_offset: {layer_offset}, expert_offset: {expert_offset}")
            merged_layer_down_weights = []
            layer_num = layer_offset
            ffn_hidden_size = margs.moe_ffn_hidden_size if margs.moe_ffn_hidden_size is not None else margs.ffn_hidden_size
            sharded_intermediate_size = ffn_hidden_size // mpu.get_expert_tensor_parallel_world_size()
            while layer_num < min(num_layers, num_layers_per_batch + layer_offset):
                self.logger.debug(f"gather_and_merge_down_in_local_layers, itering layer_num: {layer_num}")
                layer = schema.get_layer(unwrapped_model, layer_num, layer_action=LayerAction.MoeOnly)
                if layer is None:
                    layer_num += 1
                    continue
                local_expert_down_weights = []
                expert_num = expert_offset
                while expert_num < min(num_experts, num_experts_per_batch + expert_offset):
                    local_down_weight = layer[f"mlp_fc2_weight.{expert_num}"]
                    assert local_down_weight.shape == (margs.hidden_size, sharded_intermediate_size)

                    local_expert_down_weights.append((expert_num + ep_experts_offset, local_down_weight))

                    expert_num += 1
                expert_ids, merged_expert_down_weights_stacked = dist_utils.stack_gather_and_merge_mtp_sharded_expert_weights(
                    local_expert_down_weights, merge_dim=1, etp_src_rank=self.etp_src_rank
                )
                merged_expert_down_weights = None
                if mpu.get_expert_tensor_parallel_rank() == 0:
                    assert merged_expert_down_weights_stacked.shape == (len(expert_ids), margs.hidden_size, ffn_hidden_size), f"{merged_expert_down_weights_stacked.shape}"
                    merged_expert_down_weights = dist_utils.gather_and_unbind_expert_weights(expert_ids, merged_expert_down_weights_stacked, ep_src_rank=self.ep_src_rank)
                merged_layer_down_weights.append((layer_num, merged_expert_down_weights))
                layer_num += 1

            return merged_layer_down_weights
        
        def get_global_layer_number(pp_rank: int, local_layer_num: int):
            self.logger.debug(f"get_global_layer_number: pp_rank: {pp_rank}, local_layer_num: {local_layer_num}")
            global_layer_num = global_num_layer_prefix_sum[pp_rank] + local_layer_num
            self.logger.debug(f"global_layer_num = {global_num_layer_prefix_sum[pp_rank]} + {local_layer_num} = {global_layer_num}")
            return global_layer_num
        
        def ep0_tp0_sync_layer_weights(merged_local_layer_weights: List[Tuple[int, List[Tuple[int, torch.Tensor]]]],
                                       layer_weight_name: str,
                                       post_fix: Optional[str] = None,
                                       layer_offset: int = 0,
                                       num_layers_per_batch: int = None):
            # merged_local_layer_weights: List[Tuple[local_layer_num, param Tensor]]
            self.logger.debug(f"ep0_tp0_sync_layer_weights layer offset: {layer_offset}")
            assert "{layer_num}" in layer_weight_name
            assert "{expert_id}" in layer_weight_name
            for local_layer_num, local_expert_weights in merged_local_layer_weights:
                # self.logger.debug(f"local_layer_num + layer_offset = {local_layer_num} + {layer_offset} = {local_layer_num + layer_offset}")
                for local_expert_id, local_weight in local_expert_weights:
                    global_layer_number = get_global_layer_number(pp_rank, local_layer_num)
                    weight_name = layer_weight_name.format(layer_num=global_layer_number, expert_id=local_expert_id)
                    if post_fix is not None:
                        weight_name += f".{post_fix}"
                    saver.add(
                        weight_name=weight_name,
                        weight=local_weight
                    )
            saver.commit()
            del merged_local_layer_weights
            torch.cuda.empty_cache()

        num_layers_per_batch = margs.moe_weight_merging_layer_batch_size \
            if hasattr(margs, "moe_weight_merging_layer_batch_size") and margs.moe_weight_merging_layer_batch_size > 0 else num_layers
        num_experts_per_batch = margs.moe_weight_merging_expert_batch_size \
            if hasattr(margs, "moe_weight_merging_expert_batch_size") and margs.moe_weight_merging_expert_batch_size > 0 else num_experts
        
        param_mapping = get_moe_param_mapping()
        torch.cuda.empty_cache()
        start_time = time.time()

        layer_offset = 0
        while layer_offset < num_layers:
            expert_offset = 0
            while expert_offset < num_experts:
                merged_layer_gate_weights, merged_layer_up_weights = split_gather_and_merge_gate_up_in_local_layers(
                    num_layers_per_batch, layer_offset, num_experts_per_batch, expert_offset
                )
                assert len(merged_layer_gate_weights) <= num_layers_per_batch
                if mpu.get_expert_model_parallel_rank() == 0 and mpu.get_expert_tensor_parallel_rank() == 0 and len(merged_layer_gate_weights) > 0:
                    ep0_tp0_sync_layer_weights(
                        merged_layer_gate_weights, param_mapping["gate"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch
                    )
                    del merged_layer_gate_weights
                    torch.cuda.empty_cache()
                    ep0_tp0_sync_layer_weights(
                        merged_layer_up_weights, param_mapping["up"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch
                    )
                    del merged_layer_up_weights
                    torch.cuda.empty_cache()
                dist_utils.wait_for_ep_etp_neighbors()
                expert_offset += num_experts_per_batch
            layer_offset += num_layers_per_batch

        layer_offset = 0
        while layer_offset < num_layers:
            expert_offset = 0
            while expert_offset < num_experts:
                merged_layer_down_weights = gather_and_merge_down_in_local_layers(
                    num_layers_per_batch, layer_offset, num_experts_per_batch, expert_offset
                )
                assert len(merged_layer_down_weights) <= num_layers_per_batch
                if mpu.get_expert_model_parallel_rank() == 0 and mpu.get_expert_tensor_parallel_rank() == 0 and len(merged_layer_down_weights) > 0:
                    ep0_tp0_sync_layer_weights(
                        merged_layer_down_weights, param_mapping["down"], layer_offset=layer_offset, num_layers_per_batch=num_layers_per_batch
                    )
                    del merged_layer_down_weights
                    torch.cuda.empty_cache()
                dist_utils.wait_for_ep_etp_neighbors()
                expert_offset += num_experts_per_batch
            layer_offset += num_layers_per_batch

        dist_utils.wait_for_pp_ep_and_etp_neighbors()
        end_time = time.time()
        if dist.get_rank() == 0:
            logger.info(f"total consumed time: {end_time - start_time} seconds")
