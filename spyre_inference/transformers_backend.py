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

"""Spyre adaptation of vLLM's Transformers backend.

Upstream's fusers replace HF's linear/norm/GLU modules with vLLM layers, which the Spyre
OOT registrations pick up on their own. Two things are left to HF's module code:

* RoPE — there is no RoPE fuser, so HF's ``rotary_emb`` survives and derives cos/sin
  inside the forward from int64 ``position_ids``, a cast torch-spyre cannot lower.
  Replaced here with a precomputed rotation cache and a matmul-only rotation.
* Models shipping both ``config.json`` and ``params.json`` parse into a bare
  ``PretrainedConfig``, which HF cannot build a model from.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from vllm.logger import init_logger
from vllm.model_executor.models.transformers import (
    TransformersEmbeddingModel,
    TransformersForCausalLM,
    TransformersForSequenceClassification,
)

from spyre_inference.custom_ops.head_pad import original_head_dim

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.pooler.seqwise.heads import ClassifierPoolerHead
    from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
    from vllm.model_executor.layers.pooler.special import DispatchPooler
    from vllm.model_executor.layers.pooler.tokwise.heads import TokenClassifierPoolerHead

logger = init_logger(__name__)


def _build_rotation_cache(
    inv_freq: torch.Tensor,
    scaling: float,
    max_position: int,
    padded_head_dim: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[max_position, 2, 2, head_dim // 2]`` rotation matrices ``[[cos, -sin], [sin, cos]]``.

    Working from ``inv_freq``/``attention_scaling`` inherits whatever rope scaling the
    module being replaced had baked into them. *padded_head_dim* extends the cache with
    identity blocks, so a Q/K padded up to it passes its trailing dimensions through.
    """
    rope_half = inv_freq.shape[0]
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
    cos, sin = torch.cos(freqs) * scaling, torch.sin(freqs) * scaling
    rot = torch.stack([cos, -sin, sin, cos], dim=1).view(max_position, 2, 2, rope_half)

    if padded_head_dim is not None and padded_head_dim // 2 > rope_half:
        identity = torch.zeros(max_position, 2, 2, padded_head_dim // 2 - rope_half)
        identity[:, 0, 0, :] = 1.0
        identity[:, 1, 1, :] = 1.0
        rot = torch.cat([rot, identity], dim=-1)

    return rot.to(dtype)


def _apply_rope_matmul(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, L, D]`` by ``rot`` ``[B, L, 2, 2, D // 2]``.

    Multiply-and-reduce rather than HF's ``rotate_half`` cat: Spyre cannot restickify the
    halves that slicing a stick-aligned head_dim produces.
    """
    b, h, seq, head_dim = x.shape
    pairs = x.transpose(1, 2).reshape(b, seq, h, 2, head_dim // 2)
    out = rot.unsqueeze(2).mul(pairs.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


def _rope_frequencies(original: nn.Module) -> dict[str | None, tuple[torch.Tensor, float]]:
    """``{layer_type: (inv_freq, attention_scaling)}`` for the rope module being replaced.

    Models mixing global and sliding-window attention (Gemma 3, Olmo 3, ...) register one
    ``{layer_type}_inv_freq`` buffer per type and select between them on a third
    ``layer_type`` argument to ``forward``; a single rope is keyed under ``None``, its
    default.
    """
    layer_types = getattr(original, "layer_types", None)
    if not layer_types:
        scaling = float(getattr(original, "attention_scaling", 1.0))
        return {None: (original.get_buffer("inv_freq"), scaling)}

    freqs = {}
    for layer_type in layer_types:
        try:
            inv_freq = original.get_buffer(f"{layer_type}_inv_freq")
        except AttributeError:
            continue  # a layer type that does not rotate registers no buffer
        scaling = float(getattr(original, f"{layer_type}_attention_scaling", 1.0))
        freqs[layer_type] = (inv_freq, scaling)
    return freqs


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for an HF rotary embedding, returning ``(rot, None)`` in place of
    ``(cos, sin)``; the patched ``apply_rotary_pos_emb`` ignores the second element.

    The cache covers ``max_position`` up front rather than growing on demand: sizing it
    from the batch's positions needs an ``.item()``, so a host sync per step and a
    data-dependent guard ``torch.compile`` cannot trace.
    """

    def __init__(
        self,
        original: nn.Module,
        max_position: int,
        padded_head_dim: int | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._cpu_caches = {
            layer_type: _build_rotation_cache(
                inv_freq.to("cpu", torch.float32),
                scaling,
                max_position,
                padded_head_dim,
                dtype,
            )
            for layer_type, (inv_freq, scaling) in _rope_frequencies(original).items()
        }
        self._caches = self._cpu_caches

    def _apply(self, fn, recurse=True):
        # Prime the device caches when the model moves to Spyre, i.e. before compile, so only
        # the index_select is traced. They are not buffers (they are built after weight
        # loading) and there are no children, so super() has nothing to do.
        self._caches = {k: fn(v) for k, v in self._cpu_caches.items()}
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        cache = self._caches[layer_type]
        rot = cache.index_select(0, position_ids.flatten())
        return rot.view(*position_ids.shape, *cache.shape[1:]), None


def _spyre_apply_rotary(q, k, rot, *args, **kwargs):
    """Rotate Q and K by the matrices ``_SpyreRotaryEmbedding`` returned."""
    return _apply_rope_matmul(q, rot), _apply_rope_matmul(k, rot)


def _rope_dispatch(original: Callable) -> Callable:
    """``apply_rotary_pos_emb`` replacement that hands stock HF calls to *original*.

    The patch lands on a modeling module in ``sys.modules`` and is never removed, so an HF
    model built later in the process has to keep working. Spyre's calls are the ones whose
    ``sin`` is the ``None`` standing in for ``_SpyreRotaryEmbedding``'s second return.
    """

    @functools.wraps(original)
    def apply_rotary_pos_emb(q, k, cos, sin=None, *args, **kwargs):
        if sin is None:
            return _spyre_apply_rotary(q, k, cos)
        return original(q, k, cos, sin, *args, **kwargs)

    apply_rotary_pos_emb._spyre_patched = True
    return apply_rotary_pos_emb


class _PreClassifierHead(nn.Module):
    """Head that runs pre_classifier → ReLU → classifier, then squeezes [B,1,C]→[B,C].

    Used by SpyreTransformersForSequenceClassification for models like DistilBERT whose
    ForSequenceClassification adds a pre_classifier layer before the final classifier.
    The inner ``classifier`` attribute holds the original linear so that
    ``_module_has_float32_params`` recurses into it and routes the head to CPU when needed.
    """

    def __init__(self_inner, pre_clf: nn.Module, classifier: nn.Module) -> None:  # noqa: N805
        super().__init__()
        self_inner.pre_clf = pre_clf
        self_inner.classifier = classifier

    def forward(self_inner, x: torch.Tensor) -> torch.Tensor:  # noqa: N805
        x = self_inner.pre_clf(x)
        x = nn.functional.relu(x)
        out = self_inner.classifier(x)
        if out.ndim == 3 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out


class _SqueezeHead(nn.Module):
    """Head that runs classifier and squeezes the spurious [B,1,C]→[B,C] dim.

    ``SequenceClassificationMixin`` wraps ``self.classifier.__class__`` with
    ``ClassifierWithReshape`` which unsqueezes the input to [B,1,hidden] before
    the linear, producing [B,1,num_labels]. ``ClassificationOutput.from_base``
    requires 1-D per-request tensors; this head squeezes the dim back.
    """

    def __init__(self_inner, classifier: nn.Module) -> None:  # noqa: N805
        super().__init__()
        self_inner.classifier = classifier

    def forward(self_inner, x: torch.Tensor) -> torch.Tensor:  # noqa: N805
        out = self_inner.classifier(x)
        if out.ndim == 3 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out


def _stamp_layer_idx(model: nn.Module) -> None:
    """Stamp ``layer_idx`` on every ``*SelfAttention`` module that lacks it.

    vLLM's ``vllm_attention_forward`` looks up the attention instance by
    ``module.layer_idx``. Any model whose ``*SelfAttention`` does not set
    ``layer_idx`` (e.g. DistilBERT, DistilRoBERTa — unlike BERT/RoBERTa which
    accept it as a constructor arg) will crash without this patch. Applies to
    both embedding and classification models.

    Walks ``model.modules()`` in traversal order, which matches the order
    ``create_attention_instances`` uses to assign indices 0..N-1.
    """
    idx = 0
    for module in model.modules():
        if type(module).__name__.endswith("SelfAttention") and not hasattr(module, "layer_idx"):
            module.layer_idx = idx
            idx += 1


def _gather_free_forward(
    self: Any,
    input_ids=None,
    token_type_ids=None,
    position_ids=None,
    inputs_embeds=None,
    past_key_values_length=0,
):
    """Shared gather-free embeddings forward for BERT-family encoder models.

    Replaces the ``forward`` of ``BertEmbeddings``, ``RobertaEmbeddings``, and
    ``XLMRobertaEmbeddings`` to eliminate two integer-tensor ops that fail on Spyre
    when ``token_type_ids`` is ``None`` (the normal vLLM encoder path):

    * ``torch.gather`` on int64 position IDs (used when the ``token_type_ids``
      buffer is present) — ``aten::gather`` has no Spyre kernel.
    * ``torch.zeros(..., dtype=torch.long)`` fallback — int64→int32 downcast
      triggers a ``ReStickifyOpHBM`` crash in Spyre inductor codegen.

    The optimization is valid specifically because the ``None`` branch produces an
    all-zero tensor: ``gather(zero_buffer, position_ids) = zeros``, and
    ``Embedding(zeros) = weight[0]``.  When explicit ``token_type_ids`` are
    supplied, the original embedding lookup is preserved.

    Position ID convention differs between the two families and is handled by
    checking for ``create_position_ids_from_input_ids`` (RoBERTa-style, offsets
    by ``padding_idx``) versus the plain sequential slice used by BERT.
    """
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    else:
        assert inputs_embeds is not None
        batch_size, seq_length = inputs_embeds.shape[:2]

    if position_ids is None:
        if hasattr(self, "create_position_ids_from_input_ids"):
            # RoBERTa / XLM-RoBERTa: position IDs are offset by padding_idx+1
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds, self.padding_idx
                )
        else:
            # BERT: sequential slice from the pre-built position_ids buffer
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    if token_type_ids is None:
        # Optimization: the None branch always produces all-zero IDs so the
        # embedding lookup always returns weight[0]. Bypass the integer gather.
        token_type_embeddings = (
            self.token_type_embeddings.weight[0].view(1, 1, -1).expand(batch_size, seq_length, -1)
        )
    else:
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embeddings = inputs_embeds + token_type_embeddings
    embeddings = embeddings + self.position_embeddings(position_ids)
    embeddings = self.LayerNorm(embeddings)
    return self.dropout(embeddings)


def _patch_encoder_gather(model: nn.Module) -> None:
    """Bind ``_gather_free_forward`` onto encoder embedding modules in *model*.

    Covers ``BertEmbeddings``, ``RobertaEmbeddings``, and
    ``XLMRobertaEmbeddings``.  Uses instance-level method binding so the
    original class is not mutated and other model instances are unaffected.
    No-op when none of these classes appear in *model*.
    """
    import contextlib
    import importlib

    target_classes: list[type] = []
    for mod_path, cls_name in [
        ("transformers.models.bert.modeling_bert", "BertEmbeddings"),
        ("transformers.models.roberta.modeling_roberta", "RobertaEmbeddings"),
        ("transformers.models.xlm_roberta.modeling_xlm_roberta", "XLMRobertaEmbeddings"),
    ]:
        with contextlib.suppress(ImportError, AttributeError):
            target_classes.append(getattr(importlib.import_module(mod_path), cls_name))

    for name, module in model.named_modules():
        if type(module) in target_classes:
            module.forward = _gather_free_forward.__get__(module, type(module))
            logger.debug("patched %s with gather-free forward", name or "model")


def _apply_spyre_encoder_patches(model: nn.Module) -> None:
    """Apply all Spyre encoder-model patches after weights are loaded.

    Called from ``load_weights`` in both encoder backend classes.  Each
    sub-patch guards itself: ``_stamp_layer_idx`` skips modules that already
    carry ``layer_idx``; ``_patch_encoder_gather`` only fires when a supported
    embedding class is present.  Both are no-ops on models that do not need them.
    """
    _stamp_layer_idx(model)
    _patch_encoder_gather(model)


def _rope_at_original_head_dim(cfg, rope: nn.Module, orig_head_dim: int) -> nn.Module:
    """Rebuild *rope* at the pre-pad head_dim.

    HF derived ``inv_freq`` from the widened ``config.head_dim``, giving one frequency
    per padded pair instead of per real pair.
    """
    padded = cfg.head_dim
    cfg.head_dim = orig_head_dim
    try:
        return type(rope)(config=cfg)
    finally:
        cfg.head_dim = padded


class SpyreTransformersForCausalLM(TransformersForCausalLM):
    """Transformers backend with the Spyre RoPE replacement wired in."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("SpyreTransformersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        self._patch_rope()
        return result

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve the bare PretrainedConfig that vLLM's Mistral parser produces for
        repos shipping both config.json and params.json, which AutoModel.from_config
        rejects, and force HF-format weight loading. ``--config-format hf`` skips it."""
        hf_config = vllm_config.model_config.hf_config
        if type(hf_config) is not PretrainedConfig:
            return

        model_id = vllm_config.model_config.hf_config_path or vllm_config.model_config.model
        try:
            resolved = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.revision,
            )
        except Exception:
            logger.warning("AutoConfig re-resolve failed for %s", model_id, exc_info=True)
            return

        skip = {"model_type", "_name_or_path", "transformers_version", "auto_map", "architectures"}
        for key, val in hf_config.to_dict().items():
            if key not in skip and val is not None:
                setattr(resolved, key, val)

        vllm_config.model_config.hf_config = resolved
        vllm_config.model_config.hf_text_config = resolved.get_text_config()
        if vllm_config.load_config.load_format in ("auto", "mistral"):
            vllm_config.load_config.load_format = "hf"
        logger.debug(
            "Re-resolved config: %s (model_type=%s), load_format=hf",
            type(resolved).__name__,
            resolved.model_type,
        )

    def _patch_rope(self):
        """Swap HF's rotary embedding and ``apply_rotary_pos_emb`` for the Spyre ones.

        Partial rotary dimensions (e.g. Phi-3) are unsupported — the cache would cover
        only the rotated dims — but reach a shape mismatch here rather than a check:
        ``_maybe_pad_head_dim`` already rejects them whenever padding is needed.
        """
        # The text backbone holding rotary_emb; multimodal models nest it one level
        # deeper, at model.model.language_model, and carry the rope config on its own
        # config rather than the top-level one.
        inner = self.model.model if hasattr(self.model, "model") else self.model
        backbone = cast(nn.Module, getattr(inner, "language_model", inner))
        cfg = getattr(backbone, "config", self.model.config)

        # head_dim is already stick-aligned (the platform pads it, and the weight pass
        # pads Q/K interleaved to match), so the rotation only needs the pre-pad
        # frequencies identity-padded back out to the widened width.
        rope_source = backbone.get_submodule("rotary_emb")
        orig_head_dim = original_head_dim(cfg)
        padded_head_dim = None
        if orig_head_dim is not None:
            padded_head_dim = cfg.head_dim
            rope_source = _rope_at_original_head_dim(cfg, rope_source, orig_head_dim)

        spyre_rope = _SpyreRotaryEmbedding(
            rope_source,
            self._max_position,
            padded_head_dim,
            next(self.model.parameters()).dtype,
        )
        backbone.rotary_emb = spyre_rope

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if module is spyre_rope:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding"):
                parent_name, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, attr, spyre_rope)
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope

            # apply_rotary_pos_emb is a module-level function in HF modeling files, so it
            # is patched once per modeling module rather than per layer.
            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            existing = getattr(mod, "apply_rotary_pos_emb", None)
            if existing is None or getattr(existing, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _rope_dispatch(existing)
            patched_mods.add(id(mod))


# using_transformers_backend() compares _ModelInfo.architecture, which is model_cls.__name__,
# against "TransformersForCausalLM", so the subclass has to keep answering to that name.
SpyreTransformersForCausalLM.__name__ = "TransformersForCausalLM"


class SpyreTransformersEmbeddingModel(TransformersEmbeddingModel):
    """Transformers backend for encoder pooling models on Spyre."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        SpyreTransformersForCausalLM._fix_generic_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("SpyreTransformersEmbeddingModel ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        hf_model = self.model.model if hasattr(self.model, "model") else self.model
        _apply_spyre_encoder_patches(hf_model)
        return result


# Same aliasing requirement as SpyreTransformersForCausalLM.
SpyreTransformersEmbeddingModel.__name__ = "TransformersEmbeddingModel"


class SpyreTransformersForSequenceClassification(TransformersForSequenceClassification):
    """Transformers backend for pooling/classify (reranker) models on Spyre."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from transformers import AutoModelForSequenceClassification

        SpyreTransformersForCausalLM._fix_generic_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # SequenceClassificationMixin only extracts `classifier` (or `score`) from the
        # ForSequenceClassification model. Models like DistilBERT have an additional
        # pre_classifier layer with checkpoint weights that the weight loader must be
        # able to place. Register it here so the loader finds it.
        # Weight registration only — inference wiring happens in _install_head via
        # load_weights, after checkpoint weights are loaded.
        # Only register modules that actually have parameters — parameter-free modules
        # like nn.Dropout have no checkpoint weights.
        with torch.device("meta"):
            seq_cls_model = AutoModelForSequenceClassification.from_config(
                self.model.config,
                dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )
        module = getattr(seq_cls_model, "pre_classifier", None)
        if module is not None and not hasattr(self, "pre_classifier") and list(module.parameters()):
            self.pre_classifier = module
            self.init_parameters(module, dtype=self.model_config.head_dtype)

    def _install_head(self) -> None:
        """Replace self.classifier with a head that runs pre_classifier if present.

        Called from load_weights after checkpoint weights are loaded.
        self.classifier must still name the original nn.Linear at load time so the
        weight loader can map 'classifier.*' weights onto it.  We swap it here,
        after loading, and update the pooler's stored reference at the same time.
        """
        # SequenceClassificationMixin builds self.pooler with classifier=self.classifier.
        # ClassifierPoolerHead stores that reference directly, so rebinding
        # self.classifier alone does not update the pooler — both must be updated.
        #
        # vLLM's Base.forward calls self.model (the backbone), not
        # ForSequenceClassification.forward. For DistilBERT that means
        # pre_classifier → ReLU → classifier is never invoked automatically.
        # _PreClassifierHead wires it in. ClassifierWithReshape (added by the mixin)
        # produces a spurious [B,1,num_labels] shape; both heads squeeze it back.
        pre_classifier = getattr(self, "pre_classifier", None)
        if pre_classifier is None:
            new_head = _SqueezeHead(self.classifier)
        elif isinstance(pre_classifier, nn.Linear):
            # self.classifier was materialised at head_dtype (float32 by default for
            # pooling models) by SequenceClassificationMixin.init_parameters.
            # ClassifierPoolerHead casts pooled_data to head_dtype before calling
            # self.classifier, so pre_clf must be at the same dtype. Read the dtype
            # from self.classifier rather than hardcoding float32 so this stays
            # correct if head_dtype changes in the future.
            clf_dtype = next(self.classifier.parameters()).dtype
            pre_classifier = pre_classifier.to(dtype=clf_dtype)
            new_head = _PreClassifierHead(pre_classifier, self.classifier)
        else:
            raise NotImplementedError(
                f"{type(self.model).__name__} has a pre_classifier of type "
                f"{type(pre_classifier).__name__}, which is not a flat nn.Linear. "
                "Extend _PreClassifierHead or add a dedicated head wrapper before "
                "enabling this model on Spyre."
            )

        self.classifier = new_head
        # Remove the top-level pre_classifier registration now that it is nested
        # under classifier.pre_clf — named_parameters() would otherwise report the
        # same Parameter twice (pre_classifier.weight and classifier.pre_clf.weight).
        if hasattr(self, "pre_classifier"):
            del self.pre_classifier

        dispatch = cast("DispatchPooler", self.pooler)
        classify_pooler = dispatch.poolers_by_task.get("classify")
        if classify_pooler is not None:
            seq_pooler = cast("SequencePooler", classify_pooler)
            cast("ClassifierPoolerHead", seq_pooler.head).classifier = new_head
        # token_classify shares the same classifier reference built by
        # DispatchPooler.for_seq_cls.  Update it too so both tasks use the
        # new head (pre_classifier chain + squeeze).
        token_classify_pooler = dispatch.poolers_by_task.get("token_classify")
        if token_classify_pooler is not None:
            cast("TokenClassifierPoolerHead", token_classify_pooler.head).classifier = new_head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        # Install the head wrapper after weights are loaded so that the weight
        # loader can resolve 'classifier.*' against the original nn.Linear.
        self._install_head()
        # _install_head nests weights under new paths:
        #   classifier.*     -> classifier.classifier.*  (both head variants)
        #   pre_classifier.* -> classifier.pre_clf.*     (_PreClassifierHead only)
        # The top-level pre_classifier submodule is deleted by _install_head, but
        # classifier.pre_clf.* is still yielded by named_parameters() via _PreClassifierHead.
        # track_weights_loading audits named_parameters() against this set; add all new paths.
        # Handle both flat nn.Linear (classifier.weight/bias) and multi-layer heads
        # like RobertaClassificationHead (classifier.dense.*, classifier.out_proj.*).
        for key in list(result):
            if key.startswith("classifier."):
                result.add("classifier.classifier." + key[len("classifier.") :])
            elif key.startswith("pre_classifier."):
                result.add("classifier.pre_clf." + key[len("pre_classifier.") :])
        hf_model = self.model.model if hasattr(self.model, "model") else self.model
        _apply_spyre_encoder_patches(hf_model)
        logger.debug(
            "SpyreTransformersForSequenceClassification ready: %s",
            type(self.model).__name__,
        )
        return result


# Same aliasing requirement as SpyreTransformersForCausalLM.
SpyreTransformersForSequenceClassification.__name__ = "TransformersForSequenceClassification"
