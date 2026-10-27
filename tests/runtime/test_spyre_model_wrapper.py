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

"""Device-free tests for model-wrapper I/O."""

from __future__ import annotations

import torch
import torch.nn as nn

from spyre_inference.v1.worker import spyre_model_runner as mr


def test_wrapper_converts_ints_to_int64(monkeypatch):
    seen: list[torch.dtype | None] = []

    def fake_convert(t, device=None, dtype=None):
        seen.append(dtype)
        return t if dtype is None else t.to(dtype)

    monkeypatch.setattr("spyre_inference.custom_ops.utils.convert", fake_convert)

    class _Capture(nn.Module):
        def forward(self, input_ids=None, positions=None, **kwargs):
            return {"input_ids": input_ids, "positions": positions}

    wrapper = mr._SpyreModelWrapper(
        _Capture(),
        torch.device("cpu"),
        keep_outputs_on_device=True,
        model_dtype=torch.float16,
    )
    out = wrapper(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        positions=torch.tensor([0, 1], dtype=torch.int32),
    )
    assert seen == [torch.int64, torch.int64]
    assert out["input_ids"].dtype == torch.int64
    assert out["positions"].dtype == torch.int64


def test_wrapper_recursively_converts_integer_inputs(monkeypatch):
    """Nested tensor inputs share the same boundary conversion as direct inputs."""
    seen: list[torch.dtype | None] = []

    def fake_convert(t, device=None, dtype=None):
        seen.append(dtype)
        return t if dtype is None else t.to(dtype)

    monkeypatch.setattr("spyre_inference.custom_ops.utils.convert", fake_convert)

    class _Capture(nn.Module):
        def forward(self, nested):
            return nested

    wrapper = mr._SpyreModelWrapper(
        _Capture(), torch.device("cpu"), keep_outputs_on_device=True, model_dtype=torch.float16
    )
    out = wrapper(nested={"positions": [torch.tensor([1], dtype=torch.int32)]})

    assert seen == [torch.int64]
    assert out["positions"][0].dtype == torch.int64


def test_wrapper_recursively_converts_outputs_to_cpu(monkeypatch):
    """The same tree conversion serves model outputs and multimodal inputs."""
    seen: list[object] = []

    def fake_convert(t, device=None, dtype=None):
        seen.append(device)
        return t

    monkeypatch.setattr("spyre_inference.custom_ops.utils.convert", fake_convert)

    class _Capture(nn.Module):
        def forward(self, input_ids):
            return {"nested": [input_ids]}

    wrapper = mr._SpyreModelWrapper(_Capture(), torch.device("spyre"), model_dtype=torch.float16)
    wrapper(input_ids=torch.tensor([1], dtype=torch.int64))

    assert seen == [torch.device("spyre"), "cpu"]


def test_wrapper_casts_multimodal_floats_to_the_model_dtype(monkeypatch):
    """A bfloat16 checkpoint's pixel values must not be narrowed to float16 going in."""
    seen: list[torch.dtype | None] = []

    def fake_convert(t, device=None, dtype=None):
        seen.append(dtype)
        return t if dtype is None else t.to(dtype)

    monkeypatch.setattr(mr, "convert", fake_convert)

    class _Vision(nn.Module):
        def embed_multimodal(self, pixel_values=None, **kwargs):
            return pixel_values

    wrapper = mr._SpyreModelWrapper(_Vision(), torch.device("cpu"), model_dtype=torch.bfloat16)
    out = wrapper.embed_multimodal(pixel_values=torch.zeros(2, 3, dtype=torch.float32))

    assert seen == [torch.bfloat16]
    assert out.dtype == torch.bfloat16


class _FixedBucketer:
    """Minimal `SpyreShapeBucketer` stand-in: every size pads to one fixed bucket."""

    def __init__(self, bucket: int) -> None:
        self._bucket = bucket

    def find_bucket(self, num_tokens: int) -> int:
        return self._bucket


class _Embedder(nn.Module):
    """Returns a distinct value per row so staged/trimmed rows are checkable."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.seen_shapes: list[int] = []

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        self.seen_shapes.append(input_ids.shape[0])
        n = input_ids.shape[0]
        return torch.arange(n * self.hidden, dtype=torch.float32).reshape(n, self.hidden)


def test_embed_input_ids_stages_the_padded_result_into_the_buffer():
    """The result lands in `inputs_embeds_buffer` at the bucket shape, and the
    returned tensor is that buffer's own prefix view -- so upstream's later
    `inputs_embeds.gpu[:n].copy_(...)` is a self-copy, not a `copy_from_d2d`
    recompiled for every distinct token count."""
    hidden, bucket, num_tokens = 4, 8, 3
    model = _Embedder(hidden)
    buf = torch.zeros(bucket, hidden, dtype=torch.float32)
    wrapper = mr._SpyreModelWrapper(
        model,
        torch.device("cpu"),
        model_dtype=torch.float16,
        shape_bucketer=_FixedBucketer(bucket),
        inputs_embeds_buffer=buf,
    )

    out = wrapper.embed_input_ids(torch.zeros(num_tokens, dtype=torch.int32))

    assert model.seen_shapes == [bucket], "the model must embed the padded (bucket) shape"
    assert out.shape == (num_tokens, hidden)
    assert out.data_ptr() == buf.data_ptr(), "must return a view into the buffer, not a copy"

    # Prove it's a live view, not an incidentally-equal copy.
    buf[0, 0] = 999.0
    assert out[0, 0] == 999.0


def test_embed_input_ids_stages_even_when_the_count_already_matches_a_bucket():
    """No padding is needed, but the buffer path must still engage: staging is keyed
    on there being a bucketer and a usable buffer, not on padding having occurred."""
    hidden, bucket = 4, 8
    model = _Embedder(hidden)
    buf = torch.zeros(bucket, hidden, dtype=torch.float32)
    wrapper = mr._SpyreModelWrapper(
        model,
        torch.device("cpu"),
        model_dtype=torch.float16,
        shape_bucketer=_FixedBucketer(bucket),
        inputs_embeds_buffer=buf,
    )

    out = wrapper.embed_input_ids(torch.zeros(bucket, dtype=torch.int32))

    assert out.shape == (bucket, hidden)
    assert out.data_ptr() == buf.data_ptr()


def test_embed_input_ids_falls_back_to_trimming_on_a_dtype_mismatch():
    """A buffer that can't hold the result as-is (dtype here) must be left alone;
    the result is trimmed to a fresh tensor instead of force-fit into it."""
    hidden, bucket, num_tokens = 4, 8, 3
    model = _Embedder(hidden)
    buf = torch.zeros(bucket, hidden, dtype=torch.float16)  # result comes back float32
    wrapper = mr._SpyreModelWrapper(
        model,
        torch.device("cpu"),
        model_dtype=torch.float16,
        shape_bucketer=_FixedBucketer(bucket),
        inputs_embeds_buffer=buf,
    )

    out = wrapper.embed_input_ids(torch.zeros(num_tokens, dtype=torch.int32))

    assert out.shape == (num_tokens, hidden)
    assert out.dtype == torch.float32
    assert torch.count_nonzero(buf) == 0, "a mismatched buffer must be left untouched"


def test_embed_input_ids_trims_padding_when_there_is_no_buffer():
    """No buffer at all (e.g. construction didn't pass one): the pre-fix prefix-slice
    behavior must still work."""
    hidden, bucket, num_tokens = 4, 8, 3
    model = _Embedder(hidden)
    wrapper = mr._SpyreModelWrapper(
        model,
        torch.device("cpu"),
        model_dtype=torch.float16,
        shape_bucketer=_FixedBucketer(bucket),
    )

    out = wrapper.embed_input_ids(torch.zeros(num_tokens, dtype=torch.int32))

    assert model.seen_shapes == [bucket]
    assert out.shape == (num_tokens, hidden)


def test_setattr_keeps_the_wrappers_own_state_off_the_model():
    """``__setattr__`` forwards to the wrapped model, so a write to one of the wrapper's
    own fields (``_model_dtype``) would land on the wrong object."""

    class _Plain(nn.Module):
        pass

    model = _Plain()
    wrapper = mr._SpyreModelWrapper(model, torch.device("cpu"), model_dtype=torch.float16)

    wrapper._model_dtype = torch.bfloat16
    assert wrapper._model_dtype == torch.bfloat16
    assert not hasattr(model, "_model_dtype")

    wrapper.some_model_flag = 7
    assert model.some_model_flag == 7
