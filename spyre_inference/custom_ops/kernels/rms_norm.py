"""Spyre IR provider for rms_norm.

Self-contained provider that handles:
- Device transfer: CPU → Spyre → compute → CPU
- No dtype promotion (torch-spyre limitation, stays in input dtype)
- Epsilon as tensor (scalar broadcast limited on Spyre)
"""

import torch

from vllm import ir


def _supports_spyre(x, weight, epsilon, variance_size=None):
    """Accept tensors when variance_size is not used.
    Falls back to native provider when
    variance_size is set (not yet supported on Spyre).
    """
    return variance_size is None and all(t.dtype == torch.float16 for t in [x, weight])


@ir.ops.rms_norm.register_impl("spyre", supports_args=_supports_spyre, supported=True)
def spyre_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,  # noqa: ARG001 — required by IR schema
) -> torch.Tensor:
    """Spyre IR provider for rms_norm.

    Spyre-specific implementation details:
    - No dtype promotion: torch-spyre limitation, stays in input dtype.
    - variance_size: not supported; _supports_spyre rejects it so dispatch
      falls back to native.
    """
    # Implementation adopted from vllm/ir/ops/layernorm.py
    orig_dtype = x.dtype

    # Additional components in upstream vLLM
    # x = x.to(torch.float32)
    # x_var = x if variance_size is None else x[..., :variance_size]

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x
