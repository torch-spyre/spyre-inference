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

import functools
import importlib.metadata
import math
import multiprocessing
import os
import sys
from string import Template
from typing import TYPE_CHECKING

import torch

# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get("triton"):
        del sys.modules["triton"]

from vllm.logger import init_logger
from vllm.platforms import PlatformEnum
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

if TYPE_CHECKING:
    # NB: We can't eagerly import many things from vllm since vllm.config
    # will import this file. These would lead to circular imports
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def _disable_torch_accelerator() -> None:
    # Spyre has no torch.accelerator device, so empty_cache()/synchronize()/
    # empty_host_cache() raise "Cannot access accelerator device when none is
    # available." Our OOT platform (not CPU) makes vLLM's
    # cleanup_dist_env_and_memory() skip its is_cpu() guard and call these at
    # EngineCore shutdown. Patch at import to cover every process; matches
    # vLLM's CPU worker (issue #327).
    def _noop(*args, **kwargs) -> None:
        return None

    torch.accelerator.empty_cache = _noop  # ty: ignore[invalid-assignment]
    torch.accelerator.synchronize = _noop  # ty: ignore[invalid-assignment]
    if hasattr(torch.accelerator, "empty_host_cache"):
        torch.accelerator.empty_host_cache = _noop  # ty: ignore[invalid-assignment]


_disable_torch_accelerator()


def _raise_dynamo_recompile_limits() -> None:
    # torch-spyre runs every aten op on the spyre device as its own
    # torch.compile(op, dynamic=False), and all of them funnel through a single
    # shared dynamo frame. dynamo specializes per input signature, so the
    # accumulated recompile counter on that one frame climbs with every distinct
    # batch shape (the prefill token dimension is not bucketed) across every
    # op in the forward. A realistic serve workload overruns dynamo's default
    # accumulated_recompile_limit (256), and the limit handler then re-enters the
    # compile path recursively -> RecursionError, killing the engine.
    #
    # The (op × shape) set is finite and every recompile is correct, so raise
    # both limits far out of reach. Set at import to cover every process (engine
    # + TP workers); torch._dynamo.config is process-local (torch-spyre #444).
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 100000
    torch._dynamo.config.accumulated_recompile_limit = 100000  # ty: ignore[invalid-assignment]


_raise_dynamo_recompile_limits()


class TorchSpyrePlatform(CpuPlatform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"

    dispatch_key: str = "PrivateUse1"

    # Multi-backend init string consumed by both vllm's
    # `init_distributed_environment` and `torch.distributed.new_group`.
    # `gloo` handles CPU tensors (used by vllm's parallel-state cpu_group
    # and any host-side coordination); `spyreccl` handles Spyre tensors
    # for the device_group. See `torch_spyre._autoload` (registers
    # DISTRIBUTED_BACKEND_NAME via `dist.Backend.register_backend`).
    dist_backend: str = "cpu:gloo,spyre:spyreccl"

    # Cap applied to `max_model_len` only when the user didn't pass one —
    # `check_max_model_len` runs only in vLLM's model-derived branch.
    _DEFAULT_DERIVED_MAX_MODEL_LEN = 2048

    # Applied only when the user didn't pass `--max-num-seqs`; vLLM's own
    # LLM_CLASS default is 256, which is heavy for CI/fixtures. Enforced by
    # `pre_register_and_update`.
    _DEFAULT_MAX_NUM_SEQS = 4

    # Register the PyTorch Native Attention implementation as the CUSTOM backend.
    _backend_path = "spyre_inference.v1.attention.backends.spyre_attn.SpyreAttentionBackend"
    register_backend(AttentionBackendEnum.CUSTOM, _backend_path)

    @classmethod
    def check_max_model_len(cls, max_model_len: int) -> int:
        # vLLM only calls this on the user-didn't-specify branch of
        # `_get_and_verify_max_len`, so user-supplied values are untouched.
        return min(max_model_len, cls._DEFAULT_DERIVED_MAX_MODEL_LEN)

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        # Runs at the top of `EngineArgs.create_engine_config`, before
        # `_set_default_max_num_seqs_and_batched_tokens_args`. This is the
        # earliest safe seam to monkey-patch `EngineArgs`: doing it from
        # `register()` cyclically re-imports arg_utils during platform
        # discovery, and the swallowed ImportError silently downgrades us
        # to CpuPlatform.
        from vllm.engine.arg_utils import EngineArgs

        original = EngineArgs._set_default_max_num_seqs_and_batched_tokens_args
        if getattr(original, "_spyre_patched", False):
            return

        @functools.wraps(original)
        def _spyre_patched(self, usage_context, model_config, parallel_config):
            user_supplied = self.max_num_seqs is not None
            original(self, usage_context, model_config, parallel_config)
            if not user_supplied and self.max_num_seqs is not None:
                self.max_num_seqs = min(self.max_num_seqs, cls._DEFAULT_MAX_NUM_SEQS)

        _spyre_patched._spyre_patched = True
        EngineArgs._set_default_max_num_seqs_and_batched_tokens_args = _spyre_patched  # ty: ignore[invalid-assignment]

        # Delegate per-model EngineArgs overrides (e.g. text-only backbone
        # selection) to spyre_inference.models before ModelConfig is built.
        create_model_config = EngineArgs.create_model_config

        @functools.wraps(create_model_config)
        def _spyre_create_model_config(self):
            from spyre_inference.models import apply_prelaunch_overrides

            apply_prelaunch_overrides(self)
            return create_model_config(self)

        EngineArgs.create_model_config = _spyre_create_model_config  # ty: ignore[invalid-assignment]

    @classmethod
    def import_kernels(cls) -> None:
        # CpuPlatform.import_kernels() attempts to load vllm._C / _C_AVX*
        # which don't exist with VLLM_TARGET_DEVICE=empty. Override to no-op.
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "torch-spyre"

    @classmethod
    def device_count(cls) -> int:
        # CpuPlatform returns 1 (CPU = single device); for TP>1 we need the
        # actual Spyre card count so upstream gates like
        # `@multi_gpu_test(num_gpus=2)` don't skip on multi-card hosts.
        return torch.spyre.device_count()

    @classmethod
    def log_server_boot(cls, vllm_config: VllmConfig) -> None:
        # Only log in main process (not in TP workers)
        if multiprocessing.current_process().name != "MainProcess":
            return

        # yapf: disable
        logo_template = Template(
            template="\n    ${red}▄█▀▀█▄${r}  ${orange}█▀▀▀█▄${r}  ${yellow}█   █${r}  ${green}█▀▀▀█▄${r}  ${blue}█▀▀▀▀${r}    ${w}█  █▄   █  █▀▀▀▀ █▀▀▀▀  █▀▀▀█▄ █▀▀▀▀  █▄   █  ▄█▀▀█▄ █▀▀▀▀${r}\n" # noqa: E501
            "    ${red}▀▀▄▄▄${r}   ${orange}█▄▄▄█▀${r}  ${yellow}▀▄ ▄▀${r}  ${green}█▄▄▄█▀${r}  ${blue}█▄▄▄${r}     ${w}█  █ █  █  █▄▄▄  █▄▄▄   █▄▄▄█▀ █▄▄▄   █ █  █  █      █▄▄▄${r}\n" # noqa: E501
            "         ${red}█${r}  ${orange}█${r}        ${yellow}▀█▀${r}   ${green}█ ▀█▄${r}   ${blue}█${r}        ${w}█  █  █ █  █     █      █ ▀█▄  █      █  █ █  █      █${r}\n" # noqa: E501
            "    ${red}▀▄▄▄█▀${r}  ${orange}█${r}         ${yellow}█${r}    ${green}█   ▀█${r}  ${blue}█▄▄▄▄${r}    ${w}█  █   ▀█  █     █▄▄▄▄  █   ▀█ █▄▄▄▄  █   ▀█  ▀█▄▄█▀ █▄▄▄▄${r}\n" # noqa: E501
            "\n    version ${w}%s${r}    model ${w}%s${r}\n"
        )
        # yapf: enable
        colors = {
            "w": "\033[97;1m",  # white
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
            "red": "\033[91m",  # red (rainbow start)
            "orange": "\033[38;5;208m",  # orange
            "yellow": "\033[93m",  # yellow
            "green": "\033[92m",  # green
            "blue": "\033[94m",  # blue (rainbow end)
        }

        message = logo_template.substitute(colors)

        version = importlib.metadata.version("spyre_inference")

        model_name = vllm_config.model_config.model if vllm_config.model_config else "N/A"

        print(message % (version, model_name), flush=True)

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        """Set Spyre-specific config defaults before vLLM's defaulting logic."""
        from vllm.config import CompilationMode

        # A bare VllmConfig() (no model) reaches this hook too; every default below
        # is model-specific.
        if vllm_config.model_config is None:
            return

        # Key off enforce_eager, not compilation_config.mode: vLLM rewrites the
        # mode between repeated invocations of this hook (e.g. in the EngineCore
        # subprocess), while enforce_eager persists, so it's the only stable signal.
        if vllm_config.model_config.enforce_eager:
            vllm_config.compilation_config.mode = CompilationMode.NONE
        else:
            if vllm_config.compilation_config.mode in (
                CompilationMode.DYNAMO_TRACE_ONCE,
                CompilationMode.VLLM_COMPILE,
            ):
                logger.warning_once(
                    "Spyre-inference currently only supports ``STOCK_TORCH_COMPILE``"
                    + f", but {vllm_config.compilation_config.mode} selected!"
                )

            vllm_config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE

            # Keep vLLM's CustomOp dispatch for the OOT path.
            # vLLM defaults custom_ops to "none" whenever backend=="inductor" and
            # mode!=NONE.
            if all(s not in vllm_config.compilation_config.custom_ops for s in ("all", "none")):
                vllm_config.compilation_config.custom_ops.append("all")

            # Body: 1D compile_sizes (packed token counts). Attention (B, L)
            # is independent — see SpyreEncoderAttentionImpl gather-pack.
            # Honor a user-set list (#638); otherwise generate defaults.
            if vllm_config.compilation_config.compile_sizes:
                compile_sizes = vllm_config.compilation_config.compile_sizes
            else:
                # Largest default bucket: scheduler limit and 512 (Spyre max).
                max_capture_size = min(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    512,
                )
                if vllm_config.model_config.runner_type != "pooling":
                    # Decode packs one token per running sequence; prefill lands on
                    # the single largest bucket. Denser sizes only cost warmup time.
                    num_seqs = min(vllm_config.scheduler_config.max_num_seqs, max_capture_size)
                    sizes = {max_capture_size, num_seqs}
                    size = 1
                    while size < num_seqs:
                        sizes.add(size)
                        size *= 2
                    compile_sizes = sorted(sizes)
                else:
                    from spyre_inference.v1.worker.spyre_shape_bucketer import (
                        default_encoder_len_buckets,
                    )

                    compile_sizes = [*default_encoder_len_buckets(max_capture_size)]
                    logger.info(
                        "Pooling body token buckets (1D compile_sizes): %s",
                        compile_sizes,
                    )
                vllm_config.compilation_config.compile_sizes = compile_sizes

            max_capture_size = max(int(s) for s in compile_sizes)
            # Scheduler must not send more tokens than the largest body bucket.
            vllm_config.scheduler_config.max_num_batched_tokens = max_capture_size
            logger.warning(
                "Capping max_num_batched_tokens to %d ",
                max_capture_size,
            )

        # In check_and_update_config we assert this must be float16 for spyre.
        # This must be set here as the default, otherwise all usage (including test fixtures) would
        # require setting the dtype.
        vllm_config.model_config.dtype = torch.float16

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # The base `CpuPlatform` returns `CpuCommunicator`, which delegates
        # to gloo collectives. With `dist_backend = "cpu:gloo,spyre:spyreccl"`
        # the device_group is bound to spyreccl, so we need a Spyre-aware
        # communicator that knows which collectives the comms library
        # actually implements (and falls back manually for the rest).
        # See `spyre_inference/distributed/spyre_communicator.py`.
        return "spyre_inference.distributed.spyre_communicator.SpyreCommunicator"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, *args, **kwargs) -> str:
        # Encoder (pooling) layers have no KV cache and run bidirectional SDPA;
        # decoders use the paged backend. vLLM passes attn_type via the selector
        # config, so the choice lives here rather than as a branch in the impl.
        from vllm.v1.attention.backend import AttentionType

        attn_selector_config = kwargs.get("attn_selector_config") or (args[0] if args else None)
        attn_type = getattr(attn_selector_config, "attn_type", None)
        if attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            # Specific Spyre attention for encoder models.
            backend_path = (
                "spyre_inference.v1.attention.backends.spyre_encoder_attn."
                "SpyreEncoderAttentionBackend"
            )
        else:
            # Standard Spyre attention.
            backend_path = cls._backend_path

        # Register the selected Spyre attention implementation as CUSTOM.
        register_backend(AttentionBackendEnum.CUSTOM, backend_path)
        return AttentionBackendEnum.CUSTOM.get_path()

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        # `False` reaches `device_communicator.<op>` directly, which dynamo inlines
        # so the reduction compiles into the graph. The `torch.ops.vllm.*` wrappers
        # are opaque to inductor, and their no-mutation declaration is wrong for the
        # in-place `dist.all_reduce` they wrap, which corrupted compiled TP output.
        return False

    @classmethod
    def supports_fp8(cls) -> bool:
        # Linear layers use SpyreFp8LinearKernel (aten._scaled_mm).
        return True

    @classmethod
    def _maybe_pad_head_dim(cls, vllm_config: VllmConfig) -> None:
        """Override hf_config.head_dim to a 128-multiple when the native head_dim
        is not stick-aligned, stashing the original as ``_spyre_orig_head_dim``.

        Applies to the Transformers backend too: padding only the RoPE rotation leaves
        the KV cache allocated at the native ``get_head_size()``, which the device copy
        requires to be stick-aligned.

        No-op for models whose head_dim is already a multiple of 128 (e.g.
        head_size=128 Granite) and for models without RoPE. The restickify failure
        this works around is RoPE-induced, so non-RoPE models (OPT, GPT-2,
        GPT-BigCode) lower fine at head=64; padding them is both unnecessary and
        unsupported by the port, which assumes a RoPE model that sizes attention from
        ``config.head_dim`` and names its output projection ``o_proj`` (OPT ignores
        ``config.head_dim`` and uses ``out_proj``).
        """
        from spyre_inference.custom_ops.head_pad import reduced_rotary_dim_reason

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        num_heads = getattr(hf_config, "num_attention_heads", None)
        hidden_size = getattr(hf_config, "hidden_size", None)
        if num_heads is None or hidden_size is None:
            return

        # transformers 5.x unifies all RoPE config under `rope_parameters`
        cfgs = (hf_config, model_config.hf_text_config)
        if not any(getattr(c, "rope_parameters", None) for c in cfgs):
            return

        orig = getattr(hf_config, "head_dim", None) or hidden_size // num_heads
        if orig % 128 == 0:
            return

        padded = ((orig + 127) // 128) * 128
        for cfg in (hf_config, model_config.hf_text_config):
            reason = reduced_rotary_dim_reason(cfg)
            if reason is not None:
                raise NotImplementedError(
                    f"Spyre must pad attention head_dim {orig} -> {padded} for stick "
                    f"alignment, but this model reduces the rotary dimension below "
                    f"head_dim ({reason})."
                )
        for cfg in {id(c): c for c in (hf_config, model_config.hf_text_config)}.values():
            cfg._spyre_orig_head_dim = orig
            cfg.head_dim = padded
        # ModelConfig snapshots head_size into model_arch_config in __post_init__,
        # before this hook runs; keep it in sync or get_head_size() (and the KV
        # page-size accounting built on it) reports the pre-pad width.
        model_config.model_arch_config.head_size = padded
        logger.info(
            "Padding attention head_dim %d -> %d for Spyre stick alignment "
            "(original preserved as _spyre_orig_head_dim).",
            orig,
            padded,
        )

    @classmethod
    def _maybe_pad_intermediate_size(cls, vllm_config: VllmConfig) -> None:
        """Round hf_config.intermediate_size up to a 64-multiple when it is not
        stick-aligned, stashing the original as ``_spyre_orig_intermediate_size``.

        A SwiGLU MLP whose ``intermediate_size`` is not a multiple of the fp16
        stick fuses gate+up and slices the up half at an unaligned offset, which
        Spyre inductor cannot lower. Unlike head_dim, ``Qwen2MLP``/``Qwen3`` read
        ``config.intermediate_size`` directly, so overriding the config value
        before the model is built widens the modules with no per-class shim.

        Dense SwiGLU only (SiLU/swish is unique to it); MoE and other MLPs are
        left unpadded, as the loader can't reach their projections. Zero-padding
        is inert for SwiGLU (see ``custom_ops.mlp_pad``).
        """
        from spyre_inference.custom_ops.mlp_pad import BLOCK_SIZE

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        orig = getattr(hf_config, "intermediate_size", None)
        if not orig or orig % BLOCK_SIZE == 0:
            return
        moe_attrs = ("num_experts", "num_local_experts", "n_routed_experts")
        is_moe = any(getattr(hf_config, a, None) for a in moe_attrs)
        act = getattr(hf_config, "hidden_act", None) or getattr(
            hf_config, "hidden_activation", None
        )
        if is_moe or act not in ("silu", "swish"):
            return

        padded = ((orig + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        for cfg in {id(c): c for c in (hf_config, model_config.hf_text_config)}.values():
            cfg._spyre_orig_intermediate_size = orig
            cfg.intermediate_size = padded
        logger.info(
            "Padding MLP intermediate_size %d -> %d for Spyre stick alignment "
            "(original preserved as _spyre_orig_intermediate_size).",
            orig,
            padded,
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cls.log_server_boot(vllm_config)

        # A bare VllmConfig() (no model) reaches this hook too; guard each
        # model_config access like upstream CpuPlatform.
        if vllm_config.model_config is not None:
            # Check if the model dtype is different from float16,
            # which is only currently supported in torch-spyre
            if vllm_config.model_config.dtype != torch.float16:
                raise ValueError(
                    f"The model dtype needs to be torch.float16 for spyre, "
                    f"but was specified to be {vllm_config.model_config.dtype}"
                )

            # Pad attention head_dim up to a stick-aligned size on the native path.
            cls._maybe_pad_head_dim(vllm_config)

        # Pad SwiGLU MLP intermediate_size up to a stick-aligned size on the native path.
        cls._maybe_pad_intermediate_size(vllm_config)

        # Override block_size to a multiple of 64 if the user didn't explicitly set it.
        # The Spyre paged attention backend requires 64-element stick alignment for
        # torch.compile.
        cache_config = vllm_config.cache_config
        original_block_size = cache_config.block_size
        if original_block_size % 64 != 0:
            new_block_size = ((original_block_size + 63) // 64) * 64
            logger.warning(
                "Block size must be a multiple of 64 for the Spyre paged attention "
                "backend. Overriding block_size from %d to %d.",
                original_block_size,
                new_block_size,
            )
            cache_config.block_size = new_block_size

        parallel_config = vllm_config.parallel_config

        # Spyre does not currently support data parallelism. The worker's
        # WORLD_SIZE / RANK derivation in spyre_worker.init_device assumes a
        # single DP replica, and the spyre-comms global rank space has not
        # been validated for DP×TP configurations.
        if parallel_config.data_parallel_size > 1:
            raise ValueError(
                f"Spyre does not support data_parallel_size > 1 "
                f"(got {parallel_config.data_parallel_size})."
            )

        # The collectives torch-spyre lowers to reduce over the whole comms world
        # and ignore the group name they are handed, so the TP group must *be* the
        # world: with DP already rejected, pipeline parallelism has to go too.
        if parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                f"Spyre does not support pipeline_parallel_size > 1 "
                f"(got {parallel_config.pipeline_parallel_size})."
            )

        # Clamp CPU threading env vars before workers fork so they inherit the
        # corrected values. DP is rejected above, so world_size is the worker count.
        from spyre_inference.threading_config import configure_threading

        configure_threading(parallel_config.world_size)

        # ---- worker ----
        if parallel_config.worker_cls == "auto":
            worker_class = "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker"
            logger.info("Loading worker from: %s", worker_class)
            parallel_config.worker_cls = worker_class

        # ---- scheduler ----
        scheduler_config = vllm_config.scheduler_config
        # default scheduler
        scheduler_class = "vllm.v1.core.sched.scheduler.Scheduler"
        # if a torch spyre specific scheduler class is needed it can be loaded with
        # scheduler_class = "spyre_inference.v1.core.scheduler.TorchSpyreScheduler"
        logger.info("Loading scheduler from: %s", scheduler_class)
        scheduler_config.scheduler_cls = scheduler_class

        # Spyre can't offset- or shape-re-view one on-device KV buffer per layer
        # (torch-spyre#3770, "Unexpected stick expression"). Disabling the hybrid
        # KV-cache manager gives every layer its own buffer; SWA is still computed
        # in the model runner. No-op for non-hybrid models.
        scheduler_config.disable_hybrid_kv_cache_manager = True

        # Spyre's KV cache lives on-device with a fixed budget — the host-RAM
        # math in CpuPlatform.check_and_update_config is meaningless for us.
        # Setting VLLM_CPU_KVCACHE_SPACE makes CpuPlatform.check_and_update_config
        # populate `cache_config.kv_cache_memory_bytes`, which
        # TorchSpyreWorker.determine_available_memory returns directly.
        # Skip when the user has explicitly supplied --kv-cache-memory-bytes.
        if vllm_config.cache_config.kv_cache_memory_bytes is None:
            os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "4")

        # call CpuPlatform.check_and_update_config()
        super().check_and_update_config(vllm_config)

        # Pin the on-device KV cache to what's needed to fill the batch area:
        # max_num_seqs × ceil(max_model_len / block_size) blocks. Holds for hybrid
        # decoders too: `disable_hybrid_kv_cache_manager` above collapses every layer into
        # one `UniformTypeKVCacheSpecs` group drawing from the single global BlockPool.
        # Pooling / encoder-only models have no KV cache — do not size one.
        cache_config = vllm_config.cache_config
        if vllm_config.model_config is not None and cache_config.num_gpu_blocks_override is None:
            if cls._is_pooling_model(vllm_config):
                logger.info(
                    "Pooling/encoder model has no KV cache; leaving num_gpu_blocks_override unset."
                )
            else:
                max_num_seqs = vllm_config.scheduler_config.max_num_seqs
                max_model_len = vllm_config.model_config.max_model_len
                blocks_per_seq = math.ceil(max_model_len / cache_config.block_size)
                # +1 for BlockPool's reserved null block, which is never allocatable.
                cache_config.num_gpu_blocks_override = max_num_seqs * blocks_per_seq + 1
                logger.info(
                    "Setting num_gpu_blocks_override=%d (%d seqs × %d blocks/seq + 1 null block)",
                    cache_config.num_gpu_blocks_override,
                    max_num_seqs,
                    blocks_per_seq,
                )

    @staticmethod
    def _is_pooling_model(vllm_config: VllmConfig) -> bool:
        """Encoder / embedding / scoring models (no paged KV cache)."""
        model_config = vllm_config.model_config
        return getattr(model_config, "runner_type", None) == "pooling"
