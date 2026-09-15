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

"""BharatGen Param adaptations: serve the checkpoint as vLLM's native Llama.

Param ships ``trust_remote_code`` modeling code written against transformers 4.3x.
It cannot reach a backend at all: vLLM's registry rejects it up front because
``ParamBharatGenForCausalLM._supports_attention_backend`` is ``False``, and the flag
is honest: the module hardcodes eager attention, never consults
``config._attn_implementation``, keeps a legacy tuple KV cache and returns a
3-tuple from ``forward``. So neither the native path nor ``model_impl="transformers"``
can serve it as shipped.

It does not need to be. Param *is* Llama: the checkpoint is a Llama state dict
tensor for tensor, and the modeling code is Llama's arithmetic under renamed
classes: RMSNorm (fp32 upcast, ``rsqrt``), rope (``1/base**(arange(0,d,2)/d)``,
``cat((freqs,freqs))``, ``rotate_half``), SiLU ``gate*up`` -> ``down``, and
``1/sqrt(head_dim)`` GQA attention. So the adaptation is a config rewrite before
``ModelConfig`` is built, and from there vLLM's own ``LlamaForCausalLM`` serves it
with no Spyre-specific model code: nothing here subclasses a vLLM module.

Verified on Spyre (fp16) against the fp32 CPU reference from Param's own remote
code: identical logits and token-for-token identical greedy output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# Param config model_types this mapping applies to.
_PARAM_MODEL_TYPES = {"parambharatgen"}


def _param_llama_override(config: Any) -> Any:
    """Rewrite a Param config into the equivalent Llama config, in place.

    Module-level (not a closure) so it survives the pickle to EngineCore.

    Only the architecture and the attributes Param spells differently are touched;
    every field Llama reads that Param already provides under the same name
    (hidden_size, num_key_value_heads, rms_norm_eps, ...) is left exactly as
    loaded, so this cannot drift from the checkpoint.

    Every field beyond ``architectures``/``model_type`` is read defensively, because
    vLLM calls this twice: once on a bare ``PretrainedConfig(architectures=[""],
    model_type=...)`` just to read back the rewritten ``model_type``
    (``vllm/transformers_utils/config.py``), and once on the real config.
    """
    config.architectures = ["LlamaForCausalLM"]
    config.model_type = "llama"

    # Param's remote code derives head_dim as hidden_size // num_attention_heads and
    # its config carries no head_dim; vLLM's Llama would use the same default, but
    # set it explicitly so get_head_size() never depends on that default holding.
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if getattr(config, "head_dim", None) is None and hidden_size and num_heads:
        config.head_dim = hidden_size // num_heads

    # ParamBharatGenMLP prefers custom_mlp_ratio over intermediate_size:
    #     if hasattr(config, "custom_mlp_ratio"):
    #         self.intermediate_size = int(config.hidden_size * config.custom_mlp_ratio)
    # so the ratio, not intermediate_size, is what the weights were trained at
    # whenever the two disagree. Llama has no such attribute and reads only
    # intermediate_size, so fold the ratio in. They agree on Param-1-5B
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

    # transformers >=5 synthesises rope_parameters/rope_scaling defaults of
    # {"rope_type": "default"}; Param's own config.json carries none, and 4.3x-era
    # remote code reads the old rope_scaling["type"] spelling, which is what makes
    # the model raise KeyError: 'type' under transformers 5. "default" means no
    # scaling, so drop it and let vLLM's Llama take its unscaled path. A Param
    # config declaring *real* scaling keeps it: rope_type is then not "default".
    for attr in ("rope_scaling", "rope_parameters"):
        params = getattr(config, attr, None)
        if isinstance(params, dict) and params.get("rope_type", "default") == "default":
            try:
                setattr(config, attr, None)
            except Exception:  # pragma: no cover - config without the alias
                logger.debug("Param: could not clear %s", attr)

    return config


def force_llama_architecture(engine_args: EngineArgs) -> None:
    """Serve a BharatGen Param checkpoint as vLLM's native ``LlamaForCausalLM``.

    Runs before ``create_model_config`` builds the ``ModelConfig``, i.e. before
    vLLM resolves the architecture and rejects Param's remote code. Skipped when
    the user set ``hf_overrides`` of their own.
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

    engine_args.hf_overrides = _param_llama_override
    logger.info(
        "Param: serving %s as native LlamaForCausalLM.",
        getattr(hf_config, "model_type", "param"),
    )
