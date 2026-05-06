# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test MLP module correctness against a reference implementation.

Uses get_model() to load the full model and extract model.layers.0.mlp so
that the test exercises the same code path as real inference.  Individual
layer OOT-registration is verified on the same extracted module.
"""

import pytest
import torch
import torch.nn.functional as F

_MODEL_NAME = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"


def reference_mlp_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Golden reference: standard MLP forward (gate_up_proj → SiluAndMul → down_proj) in PyTorch."""
    gate = F.linear(x, gate_weight, gate_bias)
    up = F.linear(x, up_weight, up_bias)
    gate_up = torch.cat([gate, up], dim=-1)
    d = gate_up.shape[-1] // 2
    activated = F.silu(gate_up[..., :d]) * gate_up[..., d:]
    return F.linear(activated, down_weight, down_bias)


@pytest.fixture(scope="module")
def loaded_mlp(_distributed_init):
    """Load model.layers.0.mlp from _MODEL_NAME with dummy weights.

    Sets up vLLM world group and model-parallel state (TP=1), activates the
    OOT platform, and calls get_model() so all OOT layer registrations are
    applied exactly as they are in production.  The extracted LlamaMLP
    submodule is yielded; model-parallel state is torn down on exit.

    Depends on _distributed_init so torch.distributed is already initialized
    when init_distributed_environment runs.
    """
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationConfig
    from vllm.config.load import LoadConfig
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model
    from vllm.platforms import PlatformEnum, current_platform

    from spyre_inference.custom_ops import register_all

    platform_cls = type(current_platform)
    original_enum = platform_cls._enum
    platform_cls._enum = PlatformEnum.OOT
    register_all()

    vllm_config = VllmConfig(
        model_config=ModelConfig(model=_MODEL_NAME, dtype=torch.float16),
        cache_config=CacheConfig(block_size=16),
        parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1)
        model = get_model(vllm_config=vllm_config)

    modules = dict(model.named_modules())
    yield modules["model.layers.0.mlp"]

    destroy_model_parallel()
    platform_cls._enum = original_enum


@pytest.mark.spyre
@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
def test_mlp_forward_matches_reference(loaded_mlp, num_tokens: int):
    """SpyreMergedColumnParallelLinear + SpyreSiluAndMul + SpyreRowParallelLinear output matches golden reference."""
    mlp = loaded_mlp
    gate_up_weight = mlp.gate_up_proj.weight  # [2*intermediate_size, hidden_size]
    down_weight = mlp.down_proj.weight
    intermediate_size = gate_up_weight.shape[0] // 2
    hidden_size = gate_up_weight.shape[1]
    dtype = gate_up_weight.dtype

    gate_weight = gate_up_weight[:intermediate_size, :]
    up_weight = gate_up_weight[intermediate_size:, :]

    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    expected = reference_mlp_forward(x, gate_weight, up_weight, down_weight)
    actual = mlp(x)

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"MLP output mismatch for num_tokens={num_tokens}",
    )


@pytest.mark.spyre
@pytest.mark.mlp
def test_mlp_layer_oot_registration(loaded_mlp):
    """Verify OOT registration in the loaded model: class swaps and forward_oot routing."""
    from spyre_inference.custom_ops.linear import (
        SpyreMergedColumnParallelLinear,
        SpyreRowParallelLinear,
    )
    from spyre_inference.custom_ops.silu_and_mul import SpyreSiluAndMul

    mlp = loaded_mlp
    assert isinstance(mlp.gate_up_proj, SpyreMergedColumnParallelLinear)
    assert isinstance(mlp.act_fn, SpyreSiluAndMul)
    assert isinstance(mlp.down_proj, SpyreRowParallelLinear)
    assert mlp.act_fn._forward_method == mlp.act_fn.forward_oot

