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

"""MRV2: the single kernel seam for MRV2's Triton kernels.

bind() replaces every Triton kernel MRV2 launches with a host op from csrc/
(``torch.ops._spyre_C.mrv2_*``), so the V2 runner needs no Triton on any
architecture. Each launch site in upstream is ``_foo_kernel[grid](...)`` on a
module global, so assigning a HostKernel over that global reaches every caller.
The op takes the grid and then the kernel's own arguments by name, so a renamed
or reordered argument fails the coverage test rather than binding the wrong argument.

Kernels behind features the Spyre platform rejects (mamba align, batch-sharded
sampling, block/fly verification, adaptive verification) get an UnsupportedKernel
that names the feature if it is ever reached. Of gpu/spec_decode/, only what XPress
reaches is bound; the other speculators are refused at config time.

SPYRE_MRV2_KERNELS=triton keeps upstream's Triton kernels on x86 with
triton-cpu (the parity tests' reference); bind() then binds nothing.
"""

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_only, apply_top_k_top_p_pytorch

from spyre_inference.v1.sample.topk_topp_sampler import apply_top_k_top_p_sort_free

logger = init_logger(__name__)

_PACKAGE = "vllm.v1.worker.gpu"

# Launch options a Triton launcher accepts beside the kernel's own arguments.
_LAUNCH_OPTIONS = frozenset({"num_warps", "num_stages", "num_ctas", "maxnreg"})

# "<module>.<kernel>" (relative to vllm/v1/worker, so gpu/... is MRV2's package)
# -> the csrc op that replaces it. The op's arguments after the grid are the
# kernel's, in order.
HOST_KERNELS: dict[str, str] = {
    "gpu.input_batch._prepare_prefill_inputs_kernel": "mrv2_prepare_prefill_inputs",
    "gpu.input_batch._prepare_pos_seq_lens_kernel": "mrv2_prepare_pos_seq_lens",
    "gpu.input_batch._combine_sampled_and_draft_tokens_kernel": (
        "mrv2_combine_sampled_and_draft_tokens"
    ),
    "gpu.input_batch._get_num_sampled_and_rejected_kernel": "mrv2_get_num_sampled_and_rejected",
    "gpu.input_batch._post_update_kernel": "mrv2_post_update",
    "gpu.input_batch._post_update_num_computed_tokens_kernel": (
        "mrv2_post_update_num_computed_tokens"
    ),
    "gpu.input_batch._expand_idx_mapping_kernel": "mrv2_expand_idx_mapping",
    "gpu.block_table._gather_block_tables_kernel": "mrv2_gather_block_tables",
    "gpu.block_table._compute_slot_mappings_kernel": "mrv2_compute_slot_mappings",
    "gpu.buffer_utils._apply_write_kernel": "mrv2_apply_write",
    "gpu.sample.gumbel._temperature_kernel": "mrv2_temperature",
    "gpu.sample.gumbel._gumbel_sample_kernel": "mrv2_gumbel_sample",
    "gpu.sample.min_p._min_p_kernel": "mrv2_min_p",
    "gpu.sample.penalties._penalties_kernel": "mrv2_penalties",
    "gpu.sample.penalties._bincount_kernel": "mrv2_bincount",
    "gpu.sample.logit_bias._bias_kernel": "mrv2_logit_bias",
    "gpu.sample.bad_words._bad_words_kernel": "mrv2_bad_words",
    "gpu.sample.logprob._topk_log_softmax_kernel": "mrv2_topk_log_softmax",
    "gpu.sample.logprob._ranks_kernel": "mrv2_ranks",
    "gpu.sample.logprob._fill_logprob_token_ids_kernel": "mrv2_fill_logprob_token_ids",
    "gpu.sample.prompt_logprob._prompt_logprobs_token_ids_kernel": (
        "mrv2_prompt_logprobs_token_ids"
    ),
    "gpu.sample.output._compact_sampling_mask_kernel": "mrv2_compact_sampling_mask",
    "gpu.sample.trace_replay._trace_replay_kernel": "mrv2_trace_replay",
    "gpu.sample.thinking_budget._update_committed_marker_cache_kernel": (
        "mrv2_update_committed_marker_cache"
    ),
    "gpu.sample.thinking_budget._thinking_budget_kernel": "mrv2_thinking_budget",
    "gpu.sample.watermark._philox_gumbel_kernel": "mrv2_philox_gumbel",
    "gpu.sample.watermark._draft_step_kernel": "mrv2_draft_step",
    "gpu.structured_outputs._apply_grammar_bitmask_kernel": "mrv2_apply_grammar_bitmask",
    "gpu.metrics.logits._num_nans_kernel": "mrv2_num_nans",
    "gpu.mm.rope._prepare_rope_positions_kernel": "mrv2_prepare_rope_positions",
    "gpu.model_states.prompt_embeds._apply_prompt_embeds_kernel": "mrv2_apply_prompt_embeds",
    "gpu.model_states.mamba_hybrid._scatter_num_accepted_kernel": "mrv2_scatter_num_accepted",
    "gpu.model_states.mamba_hybrid._fill_num_accepted_kernel": "mrv2_fill_num_accepted",
    "gpu.cp_utils._dcp_local_seq_lens_kernel": "mrv2_dcp_local_seq_lens",
    "utils._zero_kv_blocks_kernel": "mrv2_zero_kv_blocks",
    # Speculative decoding: the target's rejection sampler and DFlash/XPress input prep.
    "gpu.spec_decode.rejection_sampler._flatten_sampled_kernel": "mrv2_flatten_sampled",
    "gpu.spec_decode.rejection_sampler._gather_draft_sampled_kernel": ("mrv2_gather_draft_sampled"),
    "gpu.spec_decode.rejection_sampler_utils._compute_local_logits_stats_kernel": (
        "mrv2_compute_local_logits_stats"
    ),
    "gpu.spec_decode.rejection_sampler_utils._rejection_kernel": "mrv2_rejection",
    "gpu.spec_decode.rejection_sampler_utils._resample_kernel": "mrv2_resample",
    "gpu.spec_decode.rejection_sampler_utils._insert_resampled_kernel": "mrv2_insert_resampled",
    "gpu.spec_decode.dflash.speculator._prepare_dflash_inputs_kernel": (
        "mrv2_prepare_dflash_inputs"
    ),
}

# Launch sites behind a feature the platform rejects -> that feature.
UNSUPPORTED_KERNELS: dict[str, str] = {
    "gpu.sample.batch_shard._build_shard_plan_kernel": "batch-sharded sampling",
    "gpu.sample.batch_shard._pack_logprobs_kernel": "batch-sharded sampling",
    "gpu.sample.batch_shard._unpack_logprobs_kernel": "batch-sharded sampling",
    "gpu.sample.batch_shard._pack_sampler_output_kernel": "batch-sharded sampling",
    "gpu.sample.batch_shard._unpack_gathered_output_kernel": "batch-sharded sampling",
    "gpu.model_states.mamba_hybrid.preprocess_mamba_align_fused_kernel": (
        "mamba prefix caching (mamba_cache_mode='align')"
    ),
    "gpu.model_states.recoverssm._postprocess_recoverssm_align_kernel": "RecoverSSM",
    "mamba_utils.get_aligned_state_indices_multi_group_kernel": (
        "mamba prefix caching (mamba_cache_mode='align')"
    ),
    "mamba_utils.precopy_mamba_align_fused_kernel": (
        "mamba prefix caching (mamba_cache_mode='align')"
    ),
    "mamba_utils.postprocess_mamba_fused_kernel": (
        "mamba prefix caching (mamba_cache_mode='align')"
    ),
    "mamba_utils.batch_memcpy_kernel": "mamba prefix caching (mamba_cache_mode='align')",
    "gpu.sample.watermark._repeated_context_mask_kernel": (
        "watermarking on a device tensor (upstream runs host tensors in torch)"
    ),
    "gpu.spec_decode.rejection_sampler_utils._compute_cumulative_log_p_kernel": (
        "rejection_sample_method='block'"
    ),
    "gpu.spec_decode.rejection_sampler_utils._compute_local_residual_mass_kernel": (
        "rejection_sample_method='block'"
    ),
    "gpu.spec_decode.acceptance_estimator._accumulate_kernel": "adaptive draft verification",
    "gpu.spec_decode.acceptance_estimator._refit_kernel": "adaptive draft verification",
    "gpu.spec_decode.acceptance_estimator._local_max_sumexp_kernel": (
        "adaptive draft verification"
    ),
    "gpu.spec_decode.acceptance_estimator._predict_kernel": "adaptive draft verification",
    # XPress's Triton Jacobi loop: the Spyre XPress model refines with its own head.
    "gpu.spec_decode.xpress.kernels._xpress_latent_kernel": "upstream's XPress Jacobi loop",
    "gpu.spec_decode.xpress.kernels._xpress_add_argmax_partial_kernel": (
        "upstream's XPress Jacobi loop"
    ),
    "gpu.spec_decode.xpress.kernels._xpress_add_argmax_reduce_to_blk_kernel": (
        "upstream's XPress Jacobi loop"
    ),
    "gpu.spec_decode.xpress.kernels._xpress_topc_argmax_to_blk_kernel": (
        "upstream's XPress Jacobi loop"
    ),
}


class HostKernel:
    """Stands in for a ``@triton.jit`` kernel: ``kernel[grid](*args)`` runs a host op."""

    def __init__(self, name: str, op: Callable[..., Any], params: list[str]) -> None:
        self.name = name
        self._op = op
        self._params = params
        self._adapt = _ARG_ADAPTERS.get(name)

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        def launch(*args: Any, **kwargs: Any) -> None:
            for option in _LAUNCH_OPTIONS:
                kwargs.pop(option, None)
            values = dict(zip(self._params, args))
            values.update(kwargs)
            if self._adapt is not None:
                self._adapt(values)
            self._op(list(grid), *(values[p] for p in self._params))

        return launch


def _apply_write_stride(values: dict[str, Any]) -> None:
    # MULTI_GROUP passes a per-group stride table, the single-group launch an int.
    if isinstance(values["output_stride"], int):
        values["output_stride"] = torch.tensor(values["output_stride"])


_ARG_ADAPTERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "gpu.buffer_utils._apply_write_kernel": _apply_write_stride,
}


class UnsupportedKernel:
    """Stands in for a kernel behind a feature Spyre rejects at config time."""

    def __init__(self, name: str, feature: str) -> None:
        self.name = name
        self.feature = feature

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        def launch(*args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                f"{self.name} has no Spyre host implementation: "
                f"Spyre's V2 model runner does not support {self.feature}."
            )

        return launch


# Host-side replacements bound over upstream's, at the modules that import them by name.
# The logits are already on the host, so V1's sort-free top-k/top-p replaces MRV2's
# full-vocab sort (or its Triton version), measured on x86 bring-up.
_TOP_K_TOP_P_SITES = ("gpu.sample.sampler", "gpu.sample.states")


def spyre_apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    """apply_top_k_top_p without the full-vocab sort whenever top-k bounds it."""
    if k is None:
        return logits if p is None else apply_top_k_top_p_pytorch(logits, k, p)
    if p is None:
        return apply_top_k_only(logits, k)
    return apply_top_k_top_p_sort_free(logits, k, p)


def _module(name: str) -> Any:
    return importlib.import_module(f"vllm.v1.worker.{name.rpartition('.')[0]}")


def kernel_params(kernel: object) -> list[str]:
    """The kernel's argument names, as the Triton launcher binds them."""
    fn = getattr(kernel, "fn", kernel)  # a JITFunction wraps the Python function
    fn = getattr(fn, "fn", fn)  # through @triton.heuristics/@triton.autotune
    return list(inspect.signature(fn).parameters)  # ty: ignore[invalid-argument-type]


def op_params(op_name: str) -> list[str]:
    """The csrc op's argument names after the grid (its schema keeps C++'s)."""
    schema = getattr(torch.ops._spyre_C, op_name).default._schema
    return [a.name for a in schema.arguments[1:]]


def _host_op(op_name: str) -> Callable[..., Any]:
    from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels

    if not has_sampling_kernels():
        raise RuntimeError(
            "Spyre's V2 model runner needs the csrc/ host kernels (spyre_inference._C), "
            "which failed to build or import; see the install log."
        )
    return getattr(torch.ops._spyre_C, op_name)


def use_triton() -> bool:
    from spyre_inference import envs

    return envs.SPYRE_MRV2_KERNELS == "triton"


_originals: dict[str, object] = {}


def bind(force_host: bool = False) -> set[str]:
    """Bind every HOST_KERNELS / UNSUPPORTED_KERNELS entry and the top-k/top-p override.

    Returns the names bound, so bind_or_raise() and the coverage test can diff
    them against the full lists. ``force_host`` binds the host ops even under
    SPYRE_MRV2_KERNELS=triton, for the parity tests.
    """
    bound = set()
    for site in _TOP_K_TOP_P_SITES:
        _module(f"{site}.x").apply_top_k_top_p = spyre_apply_top_k_top_p
        bound.add(f"{site}.apply_top_k_top_p")
    if use_triton() and not force_host:
        return bound

    for name, op_name in HOST_KERNELS.items():
        module, attr = _module(name), name.rpartition(".")[2]
        _originals.setdefault(name, getattr(module, attr))
        setattr(module, attr, HostKernel(name, _host_op(op_name), op_params(op_name)))
        bound.add(name)
    for name, feature in UNSUPPORTED_KERNELS.items():
        module, attr = _module(name), name.rpartition(".")[2]
        _originals.setdefault(name, getattr(module, attr))
        setattr(module, attr, UnsupportedKernel(name, feature))
        bound.add(name)
    # vllm.triton_utils' placeholder lacks next_power_of_2, which several MRV2
    # wrappers call to size a block argument before the launch.
    from vllm.triton_utils import triton

    if not hasattr(triton, "next_power_of_2"):
        triton.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (int(n) - 1).bit_length()
    return bound


def unbind() -> None:
    """Restore upstream's kernels (tests only)."""
    for name, original in _originals.items():
        setattr(_module(name), name.rpartition(".")[2], original)
    _originals.clear()


def bind_or_raise() -> None:
    """Call bind() and raise if any kernel is still unbound, rather than letting
    it fail at its first launch mid-serving."""
    expected = {f"{site}.apply_top_k_top_p" for site in _TOP_K_TOP_P_SITES}
    if not use_triton():
        expected |= set(HOST_KERNELS) | set(UNSUPPORTED_KERNELS)
    bound = bind()
    missing = expected - bound
    if missing:
        raise RuntimeError(f"kernels.bind() left {sorted(missing)} unbound.")
    logger.info(
        "MRV2 kernels: %s",
        "Triton (triton-cpu)"
        if use_triton()
        else f"{len(HOST_KERNELS)} bound to host ops, {len(UNSUPPORTED_KERNELS)} unsupported",
    )
