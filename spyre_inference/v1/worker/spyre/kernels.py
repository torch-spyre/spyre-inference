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

"""MRV2: bind() puts a csrc/ host op (``torch.ops._spyre_C.mrv2_*``) in place of every
``@triton.jit`` kernel MRV2 launches, so V2 needs no Triton on any architecture."""

import ast
import importlib
import importlib.util
import inspect
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_only, apply_top_k_top_p_pytorch

logger = init_logger(__name__)

_PACKAGE = "vllm.v1.worker.gpu"

# Launch options a Triton launcher accepts beside the kernel's own arguments.
_LAUNCH_OPTIONS = frozenset({"num_warps", "num_stages", "num_ctas", "maxnreg"})

# "<module>.<kernel>" under vllm/v1/worker -> its csrc op, which takes the grid and then
# the kernel's own arguments.
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
    "gpu.sample.gumbel._gumbel_combine_kernel": "mrv2_gumbel_combine",
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
    "gpu.sample.trace_replay._trace_replay_verified_kernel": "mrv2_trace_replay_verified",
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
    "replayssm_utils._reset_new_replayssm_slots_kernel": "ReplaySSM",
    "replayssm_utils._postprocess_replayssm_kernel": "ReplaySSM",
    "replayssm_utils._compact_replayssm_requests_kernel": "ReplaySSM",
    # greedy_argmax launches these on CUDA logits only; host logits take torch.argmax.
    "gpu.sample.greedy_argmax._argmax_partial": "greedy_argmax on CUDA logits",
    "gpu.sample.greedy_argmax._argmax_finish": "greedy_argmax on CUDA logits",
    # A sharded draft logits cache (row_mapping set) only exists with batch sharding.
    "gpu.spec_decode.draft_logits._assign_rows_kernel": "batch-sharded sampling",
    "gpu.spec_decode.draft_logits._stale_drafts_kernel": "batch-sharded sampling",
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
    "gpu.spec_decode.adaptive_verification._assign_draft_token_budget_kernel": (
        "adaptive draft verification"
    ),
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
    """Stands in for a ``@triton.jit`` kernel: ``kernel[grid](*args)`` runs a host op.

    Launch arguments bind to the upstream kernel's own parameter names, as Triton binds
    them, so an upstream signature change can't shift an argument into the wrong slot."""

    def __init__(
        self, name: str, op: Callable[..., Any], kernel_params: list[str], op_params: list[str]
    ) -> None:
        if sorted(kernel_params) != sorted(op_params):
            raise RuntimeError(
                f"{name} takes {kernel_params} upstream, but its host op takes {op_params}: "
                "port the signature change into csrc/."
            )
        self.name = name
        self._op = op
        self._kernel_params = kernel_params
        self._op_params = op_params
        self._adapt = _ARG_ADAPTERS.get(name)

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        def launch(*args: Any, **kwargs: Any) -> None:
            for option in _LAUNCH_OPTIONS:
                kwargs.pop(option, None)
            if len(args) > len(self._kernel_params):
                raise TypeError(f"{self.name} got {len(args)} positional arguments.")
            values = dict(zip(self._kernel_params, args))
            unknown = set(kwargs) - set(self._kernel_params)
            if unknown:
                raise TypeError(f"{self.name} got unknown arguments {sorted(unknown)}.")
            values.update(kwargs)
            if self._adapt is not None:
                self._adapt(values)
            self._op(list(grid), *(values[p] for p in self._op_params))

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


def apply_top_k_top_p_sort_free(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """``apply_top_k_top_p_pytorch`` without the full-vocab sort.

    Top-p only ever keeps top-k survivors, so it runs on a ``topk`` window.
    vllm keeps every token tied with a row's k-th value, and fp16 logits tie
    there often, so the window carries slack and widens to the exact tie count
    when ties run past it.

    The logits tensor is updated in-place."""
    max_k = int(k.max())
    vocab = logits.shape[1]
    if max_k >= vocab:
        return apply_top_k_top_p_pytorch(logits, k, p)
    vals, idx = logits.topk(min(2 * max_k, vocab), dim=-1)
    kth = vals.gather(1, k.long().unsqueeze(1) - 1)
    if (vals[:, -1:] == kth).any():
        vals, idx = logits.topk(int((logits >= kth).sum(dim=-1).max()), dim=-1)
    vals.masked_fill_(vals < kth, -float("inf"))
    vals, idx = vals.flip(-1), idx.flip(-1)
    probs_sum = vals.softmax(dim=-1).cumsum_(dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    vals.masked_fill_(top_p_mask, -float("inf"))
    return logits.fill_(-float("inf")).scatter_(1, idx, vals)


def apply_top_k_top_p_kernels(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """``apply_top_k_top_p_pytorch`` on the csrc/ kernels.

    Top-k alone matches the PyTorch filter with no sort, ties with the k-th
    value included. Top-p alone is vLLM's CPU kernel, whose cut can differ from a
    full sort by a few boundary tokens. Top-k together with top-p sorts the k
    survivors and cuts those exactly, so a large k is close to a full-vocab sort.

    The op masks logits in place, so a tensor that is not float32 and contiguous
    is copied here. A copy made inside the op would not be the caller's tensor.
    Serving already passes float32 logits, and then this cast is a no-op. vLLM's
    ``InputBatch.top_k`` is int32; int64 is accepted too. Only a non-contiguous
    k is copied."""
    if k is None and p is None:
        return logits
    logits = logits.to(torch.float32).contiguous()
    if k is not None:
        k = k.contiguous()
    if p is not None:
        p = p.to(torch.float32).contiguous()
    if p is None:
        torch.ops._spyre_C.cpu_topk_sampling(logits, k)  # ty: ignore[invalid-argument-type]
    elif k is None:
        torch.ops._spyre_C.cpu_topp_sampling(logits, p)  # ty: ignore[invalid-argument-type]
    else:
        torch.ops._spyre_C.cpu_topk_topp_sampling(logits, k, p)  # ty: ignore[invalid-argument-type]
    return logits


# Logits are already on the host, so the csrc/ filters (or, on an extension that predates
# them, the sort-free top-k/top-p) replace MRV2's full sort.
_TOP_K_TOP_P_SITES = ("gpu.sample.sampler", "gpu.sample.states")


def spyre_apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    """apply_top_k_top_p on the csrc/ filters, else without the full-vocab sort whenever
    top-k bounds it."""
    from spyre_inference.v1.sample.sampling_kernels import has_topk_topp_kernels

    if has_topk_topp_kernels():
        return apply_top_k_top_p_kernels(logits, k, p)
    if k is None:
        return logits if p is None else apply_top_k_top_p_pytorch(logits, k, p)
    if p is None:
        return apply_top_k_only(logits, k)
    return apply_top_k_top_p_sort_free(logits, k, p)


_WORKER = "vllm.v1.worker"


def _module(name: str) -> Any:
    return importlib.import_module(f"{_WORKER}.{name.rpartition('.')[0]}")


def kernel_params(kernel: object) -> list[str]:
    """The kernel's argument names, as the Triton launcher binds them."""
    fn = getattr(kernel, "fn", kernel)  # a JITFunction wraps the Python function
    fn = getattr(fn, "fn", fn)  # through @triton.heuristics/@triton.autotune
    return list(inspect.signature(fn).parameters)  # ty: ignore[invalid-argument-type]


def op_params(op_name: str) -> list[str]:
    """The csrc op's argument names after the grid (its schema keeps C++'s)."""
    schema = getattr(_host_namespace(), op_name).default._schema
    return [a.name for a in schema.arguments[1:]]


def _host_namespace() -> Any:
    from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels

    if not has_sampling_kernels():
        raise RuntimeError(
            "Spyre's V2 model runner needs the csrc/ host kernels (spyre_inference._C), "
            "which failed to build or import; see the install log."
        )
    return torch.ops._spyre_C


def use_triton() -> bool:
    from spyre_inference import envs

    return envs.SPYRE_MRV2_KERNELS == "triton"


# Speculators the platform refuses (only DFlash/XPress run on Spyre): their kernels are
# never reached, so the sweep leaves them out. The verifier's modules stay in.
_REFUSED_SPECULATORS = re.compile(
    r"gpu\.spec_decode\.(?!(rejection_sampler|rejection_sampler_utils|acceptance_estimator"
    r"|adaptive_verification|draft_logits|dflash\.speculator|xpress\.kernels)$)"
)


def _is_jit(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return isinstance(target, ast.Attribute) and target.attr == "jit"


def launch_sites() -> set[str]:
    """Every ``<module>.<kernel>`` launched with ``kernel[grid]`` under vllm/v1/worker."""
    root = Path(importlib.util.find_spec(_WORKER).origin).parent  # ty: ignore
    sources = {
        ".".join(p.relative_to(root).with_suffix("").parts): p.read_text()
        for p in root.rglob("*.py")
    }
    sources = {
        m: src for m, src in sources.items() if "jit" in src and not _REFUSED_SPECULATORS.match(m)
    }
    trees = {module: ast.parse(source) for module, source in sources.items()}
    jitted = {
        node.name
        for tree in trees.values()
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and any(_is_jit(d) for d in node.decorator_list)
    }
    # Kernels can be launched from a module that imports them by name.
    launch = re.compile(rf"\b(?:{'|'.join(map(re.escape, jitted))})\[")
    return {
        f"{module}.{node.func.value.id}"
        for module, tree in trees.items()
        if launch.search(sources[module])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in jitted
    }


# name -> (the module holding it, the upstream kernel, the stand-in registered over it).
_bound: dict[str, tuple[Any, Any, Any]] = {}
_top_k_top_p_originals: dict[str, Callable[..., Any]] = {}


def original(name: str) -> Any:
    if name in _bound:
        return _bound[name][1]
    return getattr(_module(name), name.rpartition(".")[2])


def bind(force_host: bool = False) -> set[str]:
    """Returns the names bound; ``force_host`` binds host ops under =triton too (parity)."""
    bound = set()
    for site in _TOP_K_TOP_P_SITES:
        module = _module(f"{site}.x")
        _top_k_top_p_originals.setdefault(site, module.apply_top_k_top_p)
        module.apply_top_k_top_p = spyre_apply_top_k_top_p
        bound.add(f"{site}.apply_top_k_top_p")
    if use_triton() and not force_host:
        return bound

    stand_ins: dict[str, Any] = {}
    for name, op_name in HOST_KERNELS.items():
        module, attr = _module(name), name.rpartition(".")[2]
        if name in _bound or not hasattr(module, attr):  # gone upstream: bind_or_raise names it
            continue
        stand_ins[name] = HostKernel(
            name,
            getattr(_host_namespace(), op_name),
            kernel_params(getattr(module, attr)),
            op_params(op_name),
        )
    for name, feature in UNSUPPORTED_KERNELS.items():
        module, attr = _module(name), name.rpartition(".")[2]
        if name in _bound or not hasattr(module, attr):
            continue
        stand_ins[name] = UnsupportedKernel(name, feature)
    if stand_ins:
        originals = {name: getattr(_module(name), name.rpartition(".")[2]) for name in stand_ins}
        _rebind({id(originals[name]): (originals[name], impl) for name, impl in stand_ins.items()})
        for name, kernel in originals.items():
            _bound[name] = (_module(name), kernel, stand_ins[name])
    bound |= set(_bound)
    # vllm.triton_utils' placeholder lacks next_power_of_2, which several MRV2
    # wrappers call to size a block argument before the launch.
    from vllm.triton_utils import triton

    if not hasattr(triton, "next_power_of_2"):
        triton.next_power_of_2 = _next_power_of_2
    return bound


def _rebind(replacements: dict[int, tuple[Any, Any]]) -> None:
    """Swap each kernel for its stand-in wherever a vLLM module holds it, ``from module
    import kernel`` copies included. register_kernels does this too, but compares every
    attribute of every loaded module against every kernel, which takes seconds."""
    for name, module in list(sys.modules.items()):
        if not name.startswith("vllm.") or module is None:
            continue
        namespace = vars(module)
        for attr, value in list(namespace.items()):
            hit = replacements.get(id(value))
            if hit is not None and value is hit[0]:
                namespace[attr] = hit[1]


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def unbind() -> None:
    """Restore upstream's kernels (tests only)."""
    for site, kernel in _top_k_top_p_originals.items():
        _module(f"{site}.x").apply_top_k_top_p = kernel
    _top_k_top_p_originals.clear()
    _rebind({id(stand_in): (stand_in, kernel) for _, kernel, stand_in in _bound.values()})
    _bound.clear()
    from vllm.triton_utils import triton

    if getattr(triton, "next_power_of_2", None) is _next_power_of_2:
        del triton.next_power_of_2


def bind_or_raise() -> None:
    """Raise at startup if bind() left a kernel unbound, not at its first launch: one
    listed here but gone upstream, or a launch site under vllm/v1/worker listed nowhere."""
    expected = {f"{site}.apply_top_k_top_p" for site in _TOP_K_TOP_P_SITES}
    if not use_triton():
        listed = set(HOST_KERNELS) | set(UNSUPPORTED_KERNELS)
        unlisted = launch_sites() - listed
        if unlisted:
            raise RuntimeError(
                f"MRV2 launches {sorted(unlisted)}, which kernels.py neither binds to a "
                "host op nor lists as unsupported."
            )
        expected |= listed
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
