# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `spyre_inference/multimodal/siglip.py`.

`patch_siglip_vision_embeddings` is an instance-level patch (not class-level):
it binds a new `forward` directly onto each `SiglipVisionEmbeddings` instance
found in the model. The staleness tripwire, embedding-buffer CPU-pin, and
output-equivalence checks cover the three distinct failure modes:
a vLLM rename (silent no-op), a device leak, and a numeric regression.

Section 4 repeats the numeric check on the card and skips without a device.
"""

import sys

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

siglip = pytest.importorskip("vllm.model_executor.models.siglip")

# SigLIP-SO400M/patch-14-384 dimensions (Granite Vision 4.1 default).
PATCH_SIZE = 14
IMAGE_SIZE = 336
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 576
HIDDEN_SIZE = 1152
IN_CHANNELS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_siglip_embeddings(device: torch.device | None = None) -> nn.Module:
    """Instantiate a real SiglipVisionEmbeddings with small deterministic weights.

    Uses the actual class so the patch target is a genuine instance.
    """
    if device is None:
        device = torch.device("cpu")
    from vllm.model_executor.models.siglip import SiglipVisionConfig

    config = SiglipVisionConfig(
        hidden_size=HIDDEN_SIZE,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=IN_CHANNELS,
    )
    emb = siglip.SiglipVisionEmbeddings(config).to(torch.float16).to(device)
    rng = torch.Generator(device="cpu").manual_seed(0)
    for p in emb.parameters():
        p.data.copy_(torch.empty_like(p.data, device="cpu").normal_(std=0.02, generator=rng))
    return emb


def _make_model_with_siglip(device: torch.device | None = None) -> nn.Module:
    """Wrap a SiglipVisionEmbeddings inside a parent module to exercise the
    `model.modules()` traversal in `patch_siglip_vision_embeddings`."""
    model = nn.Module()
    model.embeddings = _make_siglip_embeddings(device)
    return model


# ---------------------------------------------------------------------------
# 1. Staleness tripwires
# ---------------------------------------------------------------------------


@pytest.mark.siglip
@pytest.mark.parametrize(
    "symbol",
    [
        "SiglipVisionEmbeddings",
    ],
)
def test_patch_target_symbols_still_exist(symbol):
    """Every symbol the patch reaches for must still exist in the vLLM module.
    The `try/except ImportError` path returns silently, so this is the only
    place a rename or removal is caught."""
    assert getattr(siglip, symbol, None) is not None, (
        f"vllm.model_executor.models.siglip.{symbol} is gone — the corresponding "
        "Spyre patch in multimodal/siglip.py is now a silent no-op and must be updated"
    )


@pytest.mark.siglip
def test_siglip_vision_embeddings_has_patch_embedding():
    """SiglipVisionEmbeddings must have a `patch_embedding` attribute — the Conv2d
    whose weight dtype the patched forward reads for dtype promotion."""
    cls = siglip.SiglipVisionEmbeddings
    assert hasattr(cls, "__init__"), "SiglipVisionEmbeddings must be a class"
    emb = _make_siglip_embeddings()
    assert hasattr(emb, "patch_embedding"), (
        "SiglipVisionEmbeddings instance has no `patch_embedding` — "
        "the patched forward reads `self.patch_embedding.weight.dtype`"
    )


@pytest.mark.siglip
def test_siglip_vision_embeddings_has_position_embedding_and_ids():
    """SiglipVisionEmbeddings must have `position_embedding` and `position_ids` —
    both are moved to CPU by the patch."""
    emb = _make_siglip_embeddings()
    assert hasattr(emb, "position_embedding"), (
        "SiglipVisionEmbeddings has no `position_embedding` — "
        "the patch calls `module.position_embedding.to('cpu')`"
    )
    assert hasattr(emb, "position_ids"), (
        "SiglipVisionEmbeddings has no `position_ids` — the patch re-registers it as a CPU buffer"
    )


# ---------------------------------------------------------------------------
# 2. Patch application: CPU pin and forward binding
# ---------------------------------------------------------------------------


@pytest.mark.siglip
def test_patch_moves_position_embedding_to_cpu():
    """`patch_siglip_vision_embeddings` must move `position_embedding` to CPU."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = _make_model_with_siglip()
    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    assert model.embeddings.position_embedding.weight.device.type == "cpu", (
        "position_embedding.weight must be on CPU after patch"
    )


@pytest.mark.siglip
def test_patch_moves_position_ids_to_cpu():
    """`patch_siglip_vision_embeddings` must re-register `position_ids` as a CPU buffer."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = _make_model_with_siglip()
    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    assert model.embeddings.position_ids.device.type == "cpu", (
        "position_ids must be on CPU after patch"
    )


@pytest.mark.siglip
def test_patch_binds_instance_forward():
    """`patch_siglip_vision_embeddings` must bind a new `forward` directly on the
    instance — not the class — so unpatched instances are unaffected."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = _make_model_with_siglip()
    original_class_forward = siglip.SiglipVisionEmbeddings.forward

    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    # Instance has a new forward bound directly on it.
    assert "forward" in model.embeddings.__dict__, (
        "patch must bind forward on the instance, not mutate the class"
    )
    # Class-level forward is untouched — other instances are not affected.
    assert siglip.SiglipVisionEmbeddings.forward is original_class_forward, (
        "patch must not mutate SiglipVisionEmbeddings.forward at class level"
    )


@pytest.mark.siglip
def test_apply_patches_all_siglip_instances_in_model():
    """If a model contains multiple SiglipVisionEmbeddings instances, every one
    must be patched (the loop iterates `model.modules()`)."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    model = nn.Module()
    model.emb1 = _make_siglip_embeddings()
    model.emb2 = _make_siglip_embeddings()

    patch_siglip_vision_embeddings(model, torch.device("cpu"))

    assert "forward" in model.emb1.__dict__, "emb1 must be patched"
    assert "forward" in model.emb2.__dict__, "emb2 must be patched"


# ---------------------------------------------------------------------------
# 3. Numeric equivalence on CPU
# ---------------------------------------------------------------------------


@pytest.mark.siglip
def test_patched_forward_output_matches_stock():
    """The patched forward must produce the same output as the unpatched forward
    on CPU — the only change is routing the position-embed add through CPU."""
    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    rng = torch.Generator(device="cpu").manual_seed(1)
    pixel_values = torch.randn(
        1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16, generator=rng
    )

    # Reference: stock forward on an unpatched instance.
    emb_stock = _make_siglip_embeddings()
    expected = emb_stock(pixel_values)

    # Patched: same weights, patched forward.
    emb_patched = _make_siglip_embeddings()
    model = nn.Module()
    model.embeddings = emb_patched
    patch_siglip_vision_embeddings(model, torch.device("cpu"))
    actual = emb_patched(pixel_values)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. On-card equivalence (skipped without a Spyre device)
# ---------------------------------------------------------------------------


@pytest.mark.siglip
def test_patched_forward_output_matches_cpu_on_spyre():
    """The patched SigLIP embeddings forward on-card must equal the CPU reference.

    The patch routes `position_embedding` + add through CPU; the result is moved
    back to Spyre. A value mismatch means the CPU→Spyre transfer corrupted data.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.siglip import patch_siglip_vision_embeddings

    rng = torch.Generator(device="cpu").manual_seed(3)
    pixel_values = torch.randn(
        1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16, generator=rng
    )

    # CPU reference with patched forward.
    emb_cpu = _make_siglip_embeddings()
    model_cpu = nn.Module()
    model_cpu.embeddings = emb_cpu
    patch_siglip_vision_embeddings(model_cpu, torch.device("cpu"))
    expected = emb_cpu(pixel_values)

    # On-card: model on Spyre, patched forward, pixel_values on Spyre.
    device = torch.device("spyre")
    emb_dev = _make_siglip_embeddings(device)
    model_dev = nn.Module()
    model_dev.embeddings = emb_dev
    patch_siglip_vision_embeddings(model_dev, device)
    actual = emb_dev(pixel_values.to(device))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
