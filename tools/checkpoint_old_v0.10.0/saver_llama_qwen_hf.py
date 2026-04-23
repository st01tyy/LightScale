import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer

def add_arguments(parser):
    group = parser.add_argument_group(title='Jiutian HuggingFace saver')

    group.add_argument('--base-model-path', type=str, required=True,
                       help='Base hf model for SFT or RL')
    
def save_checkpoint(queue, args):
    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)
    
    md = queue_get()
    print(f"iter: {md.iteration}")

    # Embeddings
    #-----------
    embeddings_msg = queue_get("embeddings")

    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    hf_state_dict = dict()

    # Deal with padding
    def pad_weight(orig_word_embed, true_vocab_size):
        if true_vocab_size is not None:
            # figure out what our padded vocab size is
            orig_vocab_size = orig_word_embed.shape[0]

            # Cut out extra padding we don't need
            if orig_vocab_size > true_vocab_size:
                print(f"Cut out extra padding we don't need, true_vocab_size: {true_vocab_size}")
                full_word_embed = orig_word_embed[0:true_vocab_size,:]

            # Same size!
            else:
                print("Same size!")
                full_word_embed = orig_word_embed
        else:
            print("Original vocab size not specified, leaving embedding table as-is. "
                "If you've changed the tensor parallel size this could cause problems.")
            full_word_embed = orig_word_embed
        return full_word_embed
    
    full_word_embed = pad_weight(orig_word_embed, md.true_vocab_size)
    hf_state_dict["model.embed_tokens.weight"] = full_word_embed

    print(f"num layers: {md.num_layers}")
    head_num = md.checkpoint_args.num_attention_heads
    num_query_groups = md.checkpoint_args.num_query_groups
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups
    head_size = md.checkpoint_args.kv_channels
    hidden_size = md.checkpoint_args.hidden_size

    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_bias_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_bias_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_bias_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    print(q_bias_slice)
    print(k_bias_slice)
    print(v_bias_slice)

    def split_qkv_weight(merged_qkv_weight):    
        merged_qkv_weight = merged_qkv_weight.reshape([qkv_total_dim, head_size, hidden_size])
        merged_q_weight = merged_qkv_weight[q_slice].reshape(-1, hidden_size)
        merged_k_weight = merged_qkv_weight[k_slice].reshape(-1, hidden_size)
        merged_v_weight = merged_qkv_weight[v_slice].reshape(-1, hidden_size)
        return merged_q_weight, merged_k_weight, merged_v_weight
    
    sharded_intermediate_size = md.checkpoint_args.ffn_hidden_size
    
    def split_gate_up_weight(local_gate_up_weight):
        assert local_gate_up_weight.shape == (2 * sharded_intermediate_size, hidden_size)
        
        local_gate_weight = local_gate_up_weight[:sharded_intermediate_size, :]
        local_up_weight = local_gate_up_weight[sharded_intermediate_size:, :]

        return local_gate_weight, local_up_weight
    
    def split_qkv_bias(merged_qkv_weight):
        merged_qkv_weight = merged_qkv_weight.reshape([qkv_total_dim, head_size])
        merged_q_weight = merged_qkv_weight[q_bias_slice].reshape(-1,)
        merged_k_weight = merged_qkv_weight[k_bias_slice].reshape(-1,)
        merged_v_weight = merged_qkv_weight[v_bias_slice].reshape(-1,)
        return merged_q_weight, merged_k_weight, merged_v_weight

    for layer_id in range(md.num_layers):
        msg = queue_get(f"transformer layer {layer_id}")
        # duplicated tensors
        input_norm_weight = msg.pop("input norm weight")
        post_norm_weight = msg.pop("post norm weight")
        if md.norm_has_bias:
            raise RuntimeError("this should not happend")
        
        hf_state_dict[f"model.layers.{layer_id}.input_layernorm.weight"] = input_norm_weight
        hf_state_dict[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = post_norm_weight

        qkv_weight = msg.pop("qkv weight")
        dense_weight = msg.pop("dense weight")
        mlp_l1_weight = msg.pop("mlp l1 weight")

        q_weight, k_weight, v_weight = split_qkv_weight(qkv_weight)
        hf_state_dict[f"model.layers.{layer_id}.self_attn.q_proj.weight"] = q_weight
        hf_state_dict[f"model.layers.{layer_id}.self_attn.k_proj.weight"] = k_weight
        hf_state_dict[f"model.layers.{layer_id}.self_attn.v_proj.weight"] = v_weight

        hf_state_dict[f"model.layers.{layer_id}.self_attn.o_proj.weight"] = dense_weight

        hf_state_dict[f"model.layers.{layer_id}.mlp.down_proj.weight"] = mlp_l1_weight

        if hasattr(md, "num_experts") and md.num_experts:
            raise RuntimeError("this should not happend")

        # Special handling for swiglu
        if md.swiglu:
            mlp_l0_weight_W = msg.pop("mlp l0 weight W")
            mlp_l0_weight_V = msg.pop("mlp l0 weight V")
            mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
            gate_weight, up_weight = split_gate_up_weight(mlp_l0_weight)
            hf_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"] = gate_weight
            hf_state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"] = up_weight
        else:
            raise RuntimeError("this should not happend")

        if md.qkv_bias:
            qkv_bias = msg.pop("qkv bias")
            print(f"qkv bias shape: {qkv_bias.shape}")
            q_bias, k_bias, v_bias = split_qkv_bias(qkv_bias)
            hf_state_dict[f"model.layers.{layer_id}.self_attn.q_proj.bias"] = q_bias
            hf_state_dict[f"model.layers.{layer_id}.self_attn.k_proj.bias"] = k_bias
            hf_state_dict[f"model.layers.{layer_id}.self_attn.v_proj.bias"] = v_bias
        if hasattr(md, "linear_bias") and md.linear_bias:
            raise RuntimeError("this should not happend")
        
        check_message(msg)

    msg = queue_get("final norm")
    final_norm_weight = msg.pop("weight")
    hf_state_dict["model.norm.weight"] = final_norm_weight
    if hasattr(md, "norm_has_bias") and md.norm_has_bias:
        raise RuntimeError("This should NOT happend")
    
    msg = queue_get("output layer")
    output_layer_weight = pad_weight(msg.pop("weight"), md.true_vocab_size)
    hf_state_dict["lm_head.weight"] = output_layer_weight
    print("hf state dict is ready")

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    print("base model loaded")

    print("making sure state_dict is bf16")
    for key, value in hf_state_dict.items():
        if value.dtype != torch.bfloat16:
            hf_state_dict[key] = value.to(dtype=torch.bfloat16)

    model.load_state_dict(hf_state_dict)
    print("model loaded checkpoint")

    save_dir = f"{args.save_dir}/{md.iteration}"
    print(f"saving to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("done")