# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

from functools import lru_cache
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Shared registry: layer_name -> layer instance (for custom op lookup)
_LAYER_REGISTRY: dict[str, Any] = {}
_INSTANCE_COUNTERS: dict[str, int] = {}


def register_layer(instance: Any, prefix: str) -> str:
    """Register a layer instance and return its unique name.

    Used by custom ops that need to look up `self` from a standalone
    function (the custom op runs outside torch.compile and receives
    only a string key).

    Args:
        instance: The layer instance to register.
        prefix: Base name, e.g. "spyre_rmsnorm".

    Returns:
        Unique layer name, e.g. "spyre_rmsnorm_0".
    """
    count = _INSTANCE_COUNTERS.get(prefix, 0)
    name = f"{prefix}_{count}"
    _INSTANCE_COUNTERS[prefix] = count + 1
    _LAYER_REGISTRY[name] = instance
    return name


def get_layer(name: str) -> Any:
    """Look up a registered layer by name."""
    return _LAYER_REGISTRY[name]


def _convert_op_func(
    tensor: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Opaque-op body: device/dtype conversion with the Spyre dtype detour.

    Hidden behind `torch.ops.vllm.spyre_convert` so the device transfers and
    the spyre `torch.Tensor.to` monkey-patch are not traced into outer
    torch.compile graphs (no DeviceCopy nodes leak into the Inductor IR).
    """
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype

    if tensor.device.type == target_device.type and tensor.dtype == target_dtype:
        return tensor

    # Spyre requires CPU for dtype changes
    if tensor.device.type == "spyre" and tensor.dtype != target_dtype:
        tensor = tensor.to(device="cpu")

    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)

    if tensor.device.type != target_device.type:
        tensor = tensor.to(device=target_device)

    return tensor


def _convert_op_fake(
    tensor: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype
    return torch.empty(tensor.shape, dtype=target_dtype, device=target_device)


def convert(tensor, device=None, dtype=None):
    """Convert tensor device and/or dtype. No-op when both are None.

    Routes through the opaque custom op `torch.ops.vllm.spyre_convert` so the
    transfer is invisible to torch.compile / Dynamo. None tensors are
    short-circuited at the Python boundary because `infer_schema` does not
    accept Optional[Tensor] returns.

    Args:
        tensor: Input tensor, or None (passed through as None).
        device: Target device as `str` or `torch.device` (None = keep current).
        dtype: Target dtype (None = keep current).

    Returns:
        Converted tensor, or None if input is None.
    """
    if tensor is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    return torch.ops.vllm.spyre_convert(
        tensor,
        device,  # ty: ignore[invalid-argument-type]
        dtype,  # ty: ignore[invalid-argument-type]
    )


@lru_cache(maxsize=1)
def register():
    """Register the spyre_convert custom op with vLLM."""
    # CompositeExplicitAutograd so the op dispatches regardless of input device
    # (convert is called with both CPU and Spyre input tensors).
    direct_register_custom_op(
        op_name="spyre_convert",
        op_func=_convert_op_func,
        fake_impl=_convert_op_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_convert")
