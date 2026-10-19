# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Base model schema."""

import torch
from enum import Enum
from tools.checkpoint.layer_action import LayerAction


class ModelSchema:

    def __init__(self, mapping):
        self._mapping = dict(mapping)

        for key in (
            "embeddings",
            "layer_prefix",
            "layer",
            "final_norm",
            "output_layer",
            "pooler",
            "lm_head",
            "binary_head",
        ):
            assert key in mapping

    def __getitem__(self, key):
        return self._mapping[key]

    # Utilities.
    @classmethod
    def _get_deep_attr(cls, obj, path):
        assert isinstance(path, str)
        path = path.split(".")
        for key in path:
            try:
                obj = getattr(obj, key)
            except AttributeError:
                return None
        if isinstance(obj, torch.Tensor):
            obj = obj.data
        return obj

    @classmethod
    def _set_deep_tensor(cls, obj, path, src):
        if src is None:
            return
        dst = cls._get_deep_attr(obj, path)
        assert isinstance(src, torch.Tensor), "src is <%s>." % type(src).__name__
        assert isinstance(dst, torch.Tensor), "dst is <%s>." % type(dst).__name__
        assert not dst.requires_grad, "should be using '.data', from getter above."
        dst.copy_(src)

    def _get_layers(self, model):
        layers = self._get_deep_attr(model, self["layer_prefix"])
        assert layers is not None, "'layers' attribute not found."
        return layers

    def get_num_layers(self, model):
        return len(self._get_layers(model))

    # Getters.
    @classmethod
    def _get(cls, schema, model):
        return { k: cls._get_deep_attr(model, m) for k, m in schema.items() }

    def get(self, key, model):
        return self._get(self[key], model)
    
    def _is_moe_layer(self, layer):
        """
        Check if a layer is a MoE layer by inspecting layer attributes.
        
        Args:
            layer: Layer instance
            
        Returns:
            bool: True if layer is MoE, False otherwise
        """
        try:
            # Check if layer has MLP component
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                
                # Check for MoE attributes
                if hasattr(mlp, 'router'):
                    # print("mlp has router", flush=True)
                    # print(mlp.router, flush=True)
                    return True
                # if hasattr(mlp, 'experts'):
                #     return True
                # if hasattr(mlp, 'local_experts'):
                #     return True
                    
        except Exception:
            pass
            
        return False

    def get_layer(self, model, layer_idx, layer_action=LayerAction.NoAction):
        # print(f"layer action is {layer_action}", flush=True)
        schema = self["layer"]
        layer = self._get_layers(model)[layer_idx]
        params = self._get(schema, layer)
        
        # Apply layer action filtering
        if layer_action is LayerAction.NoAction:
            return params
        elif layer_action is LayerAction.DenseOnly:
            # Check if this is a dense layer (not MoE)
            is_moe = self._is_moe_layer(layer)
            return params if not is_moe else None
        elif layer_action is LayerAction.MoeOnly:
            # Check if this is a MoE layer
            is_moe = self._is_moe_layer(layer)
            return params if is_moe else None
        else:
            print(f"invalid layer action is {layer_action}, type is {type(layer_action)}, type compare is {type(layer_action) == type(LayerAction.NoAction)}, compare is {layer_action == LayerAction.NoAction}", flush=True)
            raise NotImplementedError

    # Setters.
    @classmethod
    def _set(cls, schema, model, params):
        for k, m in schema.items():
            if k in params:
                cls._set_deep_tensor(model, m, params[k])

    def set(self, key, model, params):
        self._set(self[key], model, params)

    def set_layer(self, model, layer_idx, params):
        schema = self["layer"]
        layer = self._get_layers(model)[layer_idx]
        self._set(schema, layer, params)

    # Other.
    def has_position_embeddings(self, model):
        pos_path = self["embeddings"]["pos"]
        pos = self._get_deep_attr(model, pos_path)
        return pos is not None
