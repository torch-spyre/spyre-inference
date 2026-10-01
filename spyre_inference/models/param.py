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

"""BharatGen Param: register the architecture, and translate the config fields.

Param *is* Llama. The checkpoint is a Llama state dict tensor for tensor, and the
remote code is Llama's arithmetic under renamed classes: RMSNorm (fp32 upcast,
``rsqrt``), rope (``1/base**(arange(0,d,2)/d)``, ``cat((freqs,freqs))``,
``rotate_half``), SiLU ``gate*up`` -> ``down``, ``1/sqrt(head_dim)`` GQA. Where the
two differ, the remote code is the weaker one: it hardcodes ``bias=False`` while
declaring ``attention_bias``/``mlp_bias``, never applies the ``attention_dropout``
it stores, cannot express a non-default ``head_dim``, keeps ``pretraining_tp``
slicing that upstream Llama has dropped, and reads ``rope_scaling["type"]`` while
defaulting that dict to the ``{"rope_type": ...}`` spelling, which is a latent
``KeyError``. vLLM's ``LlamaForCausalLM`` is the more faithful executor of these
weights, not a substitute for a Param implementation.

So ``ParamBharatGenForCausalLM`` is registered straight to it (``_ALIASED_ARCHS`` in
``models/__init__.py``), which is how vLLM itself serves InternLM3, TeleChat3 and
IQuestCoder: mapped to Llama, remote-code config, no model code of their own.
Registering the architecture is also what gets Param past
``_supports_attention_backend``, which is ``False`` and honestly so: the module
never consults ``config._attn_implementation``, keeps a legacy tuple KV cache and
returns a 3-tuple. That flag gates only ``model_impl="transformers"``, because
``is_backend_compatible()`` is reached from ``inspect_model_cls`` solely when no
architecture is registered.

What is left here is the config, and it is short: ``custom_mlp_ratio`` silently
changes the MLP width and has to be folded into ``intermediate_size``,
``pretraining_tp > 1`` has no vLLM equivalent and is refused, and ``head_dim`` is
pinned. Rope is deliberately untouched, because vLLM normalizes it after this runs.
The architecture and ``model_type`` are left alone too, which is what keeps Param
visible in the logs and in ``/v1/models``.

Verified on Spyre (fp16) to produce token-for-token the same greedy output as the
config-rewrite mechanism this replaces, which is the equivalence that matters: the
executing class is the same either way. Both diverge from an fp32 CPU reference
after 21 tokens, at a step where the top-2 logit gap is 0.011 -- fp16 resolving a
near-tie, not a dropped field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# Param config model_types this mapping applies to.
_PARAM_MODEL_TYPES = {"parambharatgen"}


def _param_config_override(config: Any) -> Any:
    """Translate the fields Param spells differently from ``LlamaConfig``, in place.

    Module-level (not a closure) so it survives the pickle to EngineCore.

    ``architectures`` and ``model_type`` are deliberately *not* touched: the
    architecture is registered, so Param is served under its own name. Only the
    attributes Param spells differently are translated; every field Llama reads that
    Param already provides under the same name (hidden_size, num_key_value_heads,
    rms_norm_eps, ...) is left exactly as loaded, so this cannot drift from the
    checkpoint.

    Every field is read defensively, because vLLM calls a callable ``hf_overrides``
    twice: once on a bare ``PretrainedConfig(architectures=[""], model_type=...)``
    just to read the ``model_type`` back off it
    (``vllm/transformers_utils/config.py``), and once on the real config.
    """
    # Belt and braces: vLLM derives head_size as hidden_size // num_attention_heads
    # when head_dim is absent, in both ModelConfig and Llama's own __init__, and that
    # is what Param's remote code computes too. So this is not what makes the mapping
    # work; it pins the value against a future change to that default. It has to
    # happen here rather than later: ModelConfig freezes head_size into a
    # ModelArchitectureConfig snapshot while building, so a write after that is unseen.
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if getattr(config, "head_dim", None) is None and hidden_size and num_heads:
        config.head_dim = hidden_size // num_heads

    # ParamBharatGenMLP prefers custom_mlp_ratio over intermediate_size:
    #     if hasattr(config, "custom_mlp_ratio"):
    #         self.intermediate_size = int(config.hidden_size * config.custom_mlp_ratio)
    # so the ratio, not intermediate_size, is what the weights were trained at
    # whenever the two disagree, and ParamBharatGenConfig defaults it to 3.5 even when
    # config.json omits it, so the branch is always taken. Llama has no such attribute
    # and reads only intermediate_size, so fold the ratio in. They agree on Param-1-5B
    # (4096 * 3.5 == 14336); a variant where they do not would otherwise load
    # silently at the wrong width and fail on a shape mismatch, or worse.
    ratio = getattr(config, "custom_mlp_ratio", None)
    if ratio is not None and hidden_size:
        derived = int(hidden_size * ratio)
        if derived != getattr(config, "intermediate_size", derived):
            logger.warning(
                "Param: custom_mlp_ratio %s implies intermediate_size %d but the "
                "config declares %d; using %d, which is what the MLP is built at.",
                ratio,
                derived,
                config.intermediate_size,
                derived,
            )
        config.intermediate_size = derived

    # Param inherits Llama's pretraining_tp weight-slicing branches. vLLM implements
    # its own tensor parallelism and has no equivalent, so a checkpoint that really
    # needs the slicing cannot be served this way, so refuse rather than load it wrong.
    tp = getattr(config, "pretraining_tp", 1)
    if tp not in (1, None):
        raise ValueError(
            f"Param checkpoints with pretraining_tp={tp} are not supported on Spyre: "
            "vLLM's Llama implementation has no equivalent of the remote code's "
            "weight-slicing path. Re-export the checkpoint with pretraining_tp=1."
        )

    # Rope needs nothing here, deliberately. vLLM calls patch_rope_parameters on the
    # config this returns (transformers_utils/config.py, after hf_overrides_fn), and
    # patch_legacy_rope_type inside it already promotes the legacy rope_scaling["type"]
    # spelling to rope_type, while standardize_rope_params fills in a missing one. The
    # unguarded rp["rope_type"] in _get_and_verify_max_len is safe for that reason.
    # Writing rope_type ourselves would be worse than redundant: with both keys set,
    # patch_legacy_rope_type raises ValueError on any mismatch between them.
    #
    # The KeyError: 'type' that Param hits under transformers 5 is its own remote code
    # reading the legacy spelling off a dict transformers now writes in the modern one.
    # vLLM's Llama never reads that key, so serving Param as Llama is what fixes it.

    return config


def normalize_param_config(engine_args: EngineArgs) -> None:
    """Translate a BharatGen Param config into the fields vLLM's Llama reads.

    Runs before ``create_model_config`` builds the ``ModelConfig``. Not because
    ``ModelConfig`` reads these fields (``intermediate_size`` and ``pretraining_tp``
    are read nowhere in ``vllm/config``, the first reader of the former being
    ``LlamaMLP``), but because ``hf_overrides`` is the last point that runs in the
    engine process, once, before ``ModelConfig`` freezes ``head_size``,
    ``hidden_size`` and the max-len derivation into a ``ModelArchitectureConfig``
    snapshot. A model ``__init__`` would be too late for that snapshot, runs per
    worker rank on an already-serialised config, and would mean subclassing
    ``LlamaForCausalLM`` rather than pointing the registry straight at it. It also
    turns a ``custom_mlp_ratio`` disagreement into a launch-time warning instead of a
    weight-shape mismatch mid-load.

    Skipped when the user set ``hf_overrides`` of their own.
    """
    if engine_args.hf_overrides:
        return
    from vllm.transformers_utils.config import get_config

    # Detect by config model_type rather than the checkpoint name, so Param
    # derivatives and local copies under unrelated names are still recognised. On
    # any load failure defer to ModelConfig, which loads the same config and
    # raises the real error.
    try:
        hf_config = get_config(
            engine_args.hf_config_path or engine_args.model,
            engine_args.trust_remote_code,
            engine_args.revision,
            engine_args.code_revision,
            engine_args.config_format,
            token=engine_args.hf_token,
        )
    except Exception:
        return
    if getattr(hf_config, "model_type", None) not in _PARAM_MODEL_TYPES:
        return

    engine_args.hf_overrides = _param_config_override
    logger.info(
        "Param: serving %s with vLLM's LlamaForCausalLM; normalizing its config.",
        getattr(hf_config, "model_type", "param"),
    )
