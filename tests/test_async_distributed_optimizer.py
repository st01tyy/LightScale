#!/usr/bin/env python3
"""Focused tests for asynchronous DistributedOptimizer gradient accumulation."""

from types import SimpleNamespace
import unittest

import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer


class TestAsyncDistributedOptimizer(unittest.TestCase):
    def _make_optimizer(self):
        bf16_param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        fp32_param = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
        bf16_main = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        fp32_main = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        ranges = {
            bf16_param: SimpleNamespace(start=1, end=3),
            fp32_param: SimpleNamespace(start=0, end=2),
        }
        optimizer = SimpleNamespace(
            is_stub_optimizer=False,
            config=SimpleNamespace(
                optimizer='adam',
                bf16=True,
                use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
            ),
            grad_scaler=None,
            ddp_config=SimpleNamespace(use_megatron_fsdp=False),
            model_float16_groups=[[bf16_param]],
            shard_fp32_from_float16_groups=[[bf16_main]],
            model_fp32_groups=[[fp32_param]],
            shard_fp32_groups=[[fp32_main]],
        )
        optimizer._get_model_param_range_map = lambda param: {'param': ranges[param]}
        return optimizer, bf16_param, fp32_param, bf16_main, fp32_main

    def test_accumulate_grad_step_adds_local_bf16_and_fp32_shards(self):
        optimizer, bf16_param, fp32_param, bf16_main, fp32_main = self._make_optimizer()
        bf16_param.main_grad = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
        fp32_param.main_grad = torch.tensor([5, 6, 7], dtype=torch.float32)

        DistributedOptimizer.accumulate_grad_step(optimizer)
        torch.testing.assert_close(bf16_main.grad, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(fp32_main.grad, torch.tensor([5.0, 6.0]))

        bf16_param.main_grad = torch.tensor([4, 3, 2, 1], dtype=torch.bfloat16)
        fp32_param.main_grad = torch.tensor([1, 2, 3], dtype=torch.float32)
        DistributedOptimizer.accumulate_grad_step(optimizer)

        torch.testing.assert_close(bf16_main.grad, torch.tensor([5.0, 5.0]))
        torch.testing.assert_close(fp32_main.grad, torch.tensor([6.0, 8.0]))

    def test_accumulate_grad_step_rejects_unsupported_optimizer(self):
        optimizer, *_ = self._make_optimizer()
        optimizer.config.optimizer = 'sgd'

        with self.assertRaises(NotImplementedError):
            DistributedOptimizer.accumulate_grad_step(optimizer)

    def test_final_step_uses_ready_grads_without_prepare_grads(self):
        calls = []
        chain = SimpleNamespace(
            chained_optimizers=[],
            config=SimpleNamespace(log_num_zeros_in_grad=False),
        )
        chain.get_grad_norm = lambda: 2.0
        chain.step_with_ready_grads = lambda: calls.append('step') or True

        update_successful, grad_norm, num_zeros = ChainedOptimizer.step_with_accumulated_grads(
            chain
        )

        self.assertTrue(update_successful)
        self.assertEqual(grad_norm, 2.0)
        self.assertIsNone(num_zeros)
        self.assertEqual(calls, ['step'])


if __name__ == '__main__':
    unittest.main()