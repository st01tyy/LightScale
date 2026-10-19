def get_jiutian_or_llama_param_mapping():
    mapping = {
        "embeddings": "model.embed_tokens.weight",
        "query": "model.layers.{layer_num}.self_attn.q_proj",
        "key": "model.layers.{layer_num}.self_attn.k_proj",
        "value": "model.layers.{layer_num}.self_attn.v_proj",
        "output": "model.layers.{layer_num}.self_attn.o_proj.weight",
        "gate": "model.layers.{layer_num}.mlp.gate_proj.weight",
        "up": "model.layers.{layer_num}.mlp.up_proj.weight",
        "down": "model.layers.{layer_num}.mlp.down_proj.weight",
        "attention_norm": "model.layers.{layer_num}.input_layernorm.weight",
        "mlp_norm": "model.layers.{layer_num}.post_attention_layernorm.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight"
    }
    return mapping

def get_dense_param_mapping():
    mapping = {
        "embeddings": "model.embed_tokens.weight",
        "query": "model.layers.{layer_num}.self_attn.q_proj",
        "key": "model.layers.{layer_num}.self_attn.k_proj",
        "value": "model.layers.{layer_num}.self_attn.v_proj",
        "output": "model.layers.{layer_num}.self_attn.o_proj.weight",
        "mla_q_down": "model.layers.{layer_num}.self_attn.q_a_proj",
        "mla_q_up": "model.layers.{layer_num}.self_attn.q_b_proj",
        "mla_kv_down": "model.layers.{layer_num}.self_attn.kv_a_proj_with_mqa",
        "mla_kv_up": "model.layers.{layer_num}.self_attn.kv_b_proj",
        "mla_q_norm": "model.layers.{layer_num}.self_attn.q_a_layernorm.weight",
        "mla_kv_norm": "model.layers.{layer_num}.self_attn.kv_a_layernorm.weight",
        "mla_output": "model.layers.{layer_num}.self_attn.o_proj.weight",
        "gate": "model.layers.{layer_num}.mlp.gate_proj.weight",
        "up": "model.layers.{layer_num}.mlp.up_proj.weight",
        "down": "model.layers.{layer_num}.mlp.down_proj.weight",
        "shared_gate": "model.layers.{layer_num}.mlp.shared_experts.gate_proj.weight",
        "shared_up": "model.layers.{layer_num}.mlp.shared_experts.up_proj.weight",
        "shared_down": "model.layers.{layer_num}.mlp.shared_experts.down_proj.weight",
        "shared_router": "model.layers.{layer_num}.mlp.shared_experts.gate.weight",
        "attention_norm": "model.layers.{layer_num}.input_layernorm.weight",
        "mlp_norm": "model.layers.{layer_num}.post_attention_layernorm.weight",
        "q_norm": "model.layers.{layer_num}.self_attn.q_norm.weight",
        "k_norm": "model.layers.{layer_num}.self_attn.k_norm.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "router": "model.layers.{layer_num}.mlp.gate.weight"
    }
    return mapping

def get_moe_param_mapping():
    mapping = {
        "up": "model.layers.{layer_num}.mlp.experts.{expert_id}.up_proj.weight",
        "down": "model.layers.{layer_num}.mlp.experts.{expert_id}.down_proj.weight",
        "gate": "model.layers.{layer_num}.mlp.experts.{expert_id}.gate_proj.weight",
    }
    return mapping