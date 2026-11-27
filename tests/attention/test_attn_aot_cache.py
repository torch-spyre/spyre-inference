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

"""Tests for the attention AOT kernel cache (SPYRE_ATTN_AOT_CACHE), on CPU."""

import math
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.attention import aot_kernel_cache as akc

pytestmark = pytest.mark.attention


def kernel(*, a, b):
    # Traced with equal sizes, Dynamo ties the two dynamic dims and guards on it.
    if a.shape[0] == b.shape[0]:
        return a + b
    return a.sum() - b.sum()


def other_kernel(*, a, b):
    return a * b


def scaled_kernel(*, a, s):
    return a * s


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _dyn(*shape):
    t = torch.randn(*shape)
    torch._dynamo.mark_dynamic(t, 0)
    return t


class _Plain:
    """The plain torch.compile path, counting its calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def _files(k, suffix=".bin"):
    return sorted(f for f in os.listdir(k._dir) if f.endswith(suffix)) if k._dir else []


def test_key_tells_singleton_strides_apart():
    x = torch.randn(4, 1, 8)
    y = x.as_strided((4, 1, 8), (8, 1, 1))
    assert x.is_contiguous() and y.is_contiguous()
    assert akc._key((), {"a": x}) != akc._key((), {"a": y})


def test_key_tells_scalars_apart_by_type_and_value():
    values = (1, True, 1.0, 0.1, math.nextafter(0.1, 1.0), None, "1")
    assert len({akc._key((), {"s": v}) for v in values}) == len(values)


def test_kernel_never_reuses_a_function_across_keys():
    plain = _Plain(scaled_kernel)
    k = akc.AotKernel(scaled_kernel, plain)
    x = torch.randn(4, 1, 8)
    y = x.as_strided((4, 1, 8), (8, 1, 1))
    calls = [(x, 2), (y, 2), (x, 2.0), (x, True)]
    for a, s in calls:
        assert torch.equal(k(a=a, s=s), a * s)
    # Same shapes, but a size-one stride or a scalar type differs: one function each.
    assert len(k._loaded) == len(calls)
    assert len({id(fn) for fn, _, _ in k._loaded.values()}) == len(calls)
    assert plain.calls == 0


def test_key_sees_aliasing():
    x, y = torch.randn(8), torch.randn(8)
    assert akc._key((), {"a": x, "b": x}) != akc._key((), {"a": x, "b": y})
    # Two views of one storage, against the same views of two storages.
    assert akc._key((), {"a": x[:4], "b": x[4:]}) != akc._key((), {"a": x[:4], "b": y[4:]})


def test_key_holds_dynamic_dims_except_zero_and_one():
    def key(n):
        return akc._key((), {"a": _dyn(n, 8)})

    assert key(5) == key(7)
    assert len({key(0), key(1), key(5)}) == 3


def test_kernel_saves_then_loads(tmp_path):
    a, b = _dyn(4, 8), _dyn(4, 8)
    plain = _Plain(kernel)
    first = akc.AotKernel(kernel, plain)
    assert torch.equal(first(a=a, b=b), a + b)
    assert len(_files(first)) == 1
    second = akc.AotKernel(kernel, plain)
    assert torch.equal(second(a=a, b=b), a + b)
    assert plain.calls == 0
    ((_, path, _),) = second._loaded.values()
    assert path is not None


def test_failed_shape_guard_takes_plain_path():
    a, b = _dyn(4, 8), _dyn(4, 8)
    plain = _Plain(kernel)
    k = akc.AotKernel(kernel, plain)
    k(a=a, b=b)
    # Unequal sizes: same key (both dynamic), but the tied-dims guard fails.
    a6, b5 = _dyn(6, 8), _dyn(5, 8)
    assert akc._key((), {"a": a6, "b": b5}) == akc._key((), {"a": a, "b": b})
    out = k(a=a6, b=b5)
    assert torch.allclose(out, a6.sum() - b5.sum())
    assert plain.calls == 1
    assert len(k._broken) == 0
    # The cheap check and torch's own bind-then-check agree.
    ((fn, _, check),) = k._loaded.values()
    for x, y in ((a6, b5), (a6, _dyn(6, 8)), (a, b)):
        assert check((), {"a": x, "b": y}) == fn.guard_check(a=x, b=y)


def test_corrupt_file_is_set_aside_and_plain_path_runs():
    a, b = _dyn(4, 8), _dyn(4, 8)
    plain = _Plain(kernel)
    k = akc.AotKernel(kernel, plain)
    k(a=a, b=b)
    (name,) = _files(k)
    with open(os.path.join(k._dir, name), "wb") as f:
        f.write(b"not a compiled function")
    fresh = akc.AotKernel(kernel, plain)
    assert torch.equal(fresh(a=a, b=b), a + b)
    assert plain.calls == 1
    assert _files(fresh, ".bad") == [name + ".bad"]
    assert _files(fresh) == []


def test_file_of_another_kernel_is_not_used(monkeypatch):
    a, b = _dyn(4, 8), _dyn(4, 8)
    mine = akc.AotKernel(kernel, _Plain(kernel))
    mine(a=a, b=b)
    (name,) = _files(mine)
    plain = _Plain(other_kernel)
    other = akc.AotKernel(other_kernel, plain)
    # Even when the file names clash, the loaded code must be this kernel's.
    monkeypatch.setattr(other, "_path", lambda key: os.path.join(mine._dir, name))
    assert torch.equal(other(a=a, b=b), a * b)
    assert plain.calls == 1


def test_kernel_name_is_in_the_file_name():
    k1 = akc.AotKernel(kernel, kernel)
    k2 = akc.AotKernel(other_kernel, other_kernel)
    k2.__name__ = k1.__name__
    assert k1._path(("same",)) != k2._path(("same",))


class _Raises:
    def guard_check(self, *args, **kwargs):
        return True

    def __call__(self, *args, **kwargs):
        raise RuntimeError("broken kernel")


def test_loaded_kernel_that_raises_falls_back(tmp_path):
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    plain = _Plain(kernel)
    k = akc.AotKernel(kernel, plain)
    key = akc._key((), {"a": a, "b": b})
    path = tmp_path / "saved.bin"
    path.write_bytes(b"x")
    k._loaded[key] = (_Raises(), str(path), None)
    assert torch.equal(k(a=a, b=b), a + b)
    assert torch.equal(k(a=a, b=b), a + b)
    assert plain.calls == 2
    assert key in k._broken
    assert not path.exists() and (tmp_path / "saved.bin.bad").exists()


def test_fresh_kernel_that_raises_falls_back(monkeypatch):
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    plain = _Plain(kernel)
    k = akc.AotKernel(kernel, plain)
    raises = _Raises()
    raises.disable_guard_check = lambda: None
    monkeypatch.setattr(k._aot, "aot_compile", lambda inputs: raises, raising=False)
    assert torch.equal(k(a=a, b=b), a + b)
    assert plain.calls == 1
    assert _files(k) == []


def test_a_file_that_cannot_be_saved_keeps_the_kernel_in_memory(monkeypatch, tmp_path):
    a, b = _dyn(4, 8), _dyn(4, 8)
    plain = _Plain(kernel)
    k = akc.AotKernel(kernel, plain)
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")  # a file, so no path below it can be created or opened
    monkeypatch.setattr(k, "_path", lambda key: str(blocked / "kernel.bin"))
    assert torch.equal(k(a=a, b=b), a + b)
    assert torch.equal(k(a=a, b=b), a + b)
    assert plain.calls == 0


def test_nothing_compiled_while_a_pool_is_open_in_another_thread():
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    plain = _Plain(kernel)
    k = akc.AotKernel(kernel, plain)
    opened, done = threading.Event(), threading.Event()

    def pool():
        with akc.compile_only():
            opened.set()
            done.wait()

    t = threading.Thread(target=pool)
    t.start()
    try:
        opened.wait()
        assert torch.equal(k(a=a, b=b), a + b)
    finally:
        done.set()
        t.join()
    assert plain.calls == 1
    assert _files(k) == [] and not k._loaded


def test_nothing_compiled_while_torch_spyre_pool_swaps_inner_compile(monkeypatch):
    def inner_compile(*args, **kwargs):
        pass

    config = SimpleNamespace(sencores=32, get_config_copy=lambda: {"sencores": 32})
    inductor = SimpleNamespace(_spyre_inner_compile=inner_compile, config=config)
    monkeypatch.setitem(sys.modules, "torch_spyre._inductor", inductor)
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    plain = _Plain(kernel)
    k = akc.AotKernel(kernel, plain)
    assert torch.equal(k(a=a, b=b), a + b)
    assert plain.calls == 1
    assert _files(k) == []


def _fake_block_pool(monkeypatch):
    """torch-spyre's pool helper without forking: the traces run inline."""
    import contextlib

    def fork_inner_compiles(workers, timeout_s=0.0, placeholders=False):
        return contextlib.nullcontext(SimpleNamespace(submitted=1, ok=1, failed=0))

    helper = SimpleNamespace(
        SecondPassCheck=lambda expected: None, fork_inner_compiles=fork_inner_compiles
    )
    monkeypatch.setitem(sys.modules, "torch_spyre._inductor.parallel_precompile", helper)
    monkeypatch.setenv("SPYRE_PARALLEL_WARMUP_PRECOMPILE", "2")
    monkeypatch.setattr(torch._dynamo, "reset", lambda: None)


def test_block_warmup_pool_counts_as_open(monkeypatch):
    from spyre_inference.v1.worker.parallel_precompile import precompile_in_workers

    _fake_block_pool(monkeypatch)
    seen = []
    precompile_in_workers(lambda pool: seen.append(akc._pool_open()), lambda: None)
    assert seen == [True]
    assert not akc._pool_open()


def test_stamp_sees_a_content_edit_that_keeps_size_and_mtime(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("X = 1\n")
    st = os.stat(src)
    before = akc._files_stamp(str(tmp_path))
    src.write_text("X = 2\n")
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert akc._files_stamp(str(tmp_path)) != before


def test_flag_off_keeps_plain_kernels(monkeypatch):
    from spyre_inference import envs
    from spyre_inference.v1.attention.backends import spyre_attn

    monkeypatch.delenv("SPYRE_ATTN_AOT_CACHE", raising=False)
    assert envs.SPYRE_ATTN_AOT_CACHE is False
    names = (
        "_page_attn_prefill_compiled",
        "_page_attn_decode_compiled",
        "_batched_decode_compiled",
    )
    for name in names:
        assert not isinstance(getattr(spyre_attn, name), akc.AotKernel)


def test_compiles_as_plain_torch_compile_under_inference_mode(monkeypatch):
    """aot_compile runs with inference mode off and non-inference fake inputs, as a
    plain torch.compile does (so its FX graph cache key is the same); the kernel is
    still keyed, saved, loaded and run in the caller's inference mode."""
    from torch._inductor import compile_fx as cfx

    seen = []
    real = cfx.fx_codegen_and_compile

    def recording(gm, example_inputs, *args, **kwargs):
        seen.append(
            (
                torch.is_inference_mode_enabled(),
                [t.is_inference() for t in example_inputs if isinstance(t, torch.Tensor)],
            )
        )
        return real(gm, example_inputs, *args, **kwargs)

    monkeypatch.setattr(cfx, "fx_codegen_and_compile", recording)
    plain = _Plain(kernel)
    with torch.inference_mode():
        a, b = _dyn(4, 8), _dyn(4, 8)
        assert a.is_inference()
        first = akc.AotKernel(kernel, plain)
        assert torch.equal(first(a=a, b=b), a + b)
        assert seen == [(False, [False, False])]
        second = akc.AotKernel(kernel, plain)
        assert torch.equal(second(a=a, b=b), a + b)
    assert plain.calls == 0 and len(seen) == 1
    assert list(second._loaded) == list(first._loaded)
    ((_, path, _),) = second._loaded.values()
    assert path is not None


def test_aot_compile_traces_a_ranged_count_size_obliviously(monkeypatch):
    """With its range argument set, the kernel traces like spyre_attn's plain path
    (backed_size_oblivious), so its graphs match the precompile pool's."""
    from torch.fx.experimental import _config as fx_config

    def ranged(a, page_range=None):
        return a + 1

    seen = []

    def fake_aot_compile(_inputs):
        seen.append(fx_config.backed_size_oblivious)
        raise RuntimeError("not compiled in this test")

    plain = _Plain(ranged)
    k = akc.AotKernel(ranged, plain, "page_range")
    monkeypatch.setattr(k._aot, "aot_compile", fake_aot_compile)
    a = torch.randn(4)
    assert torch.equal(k(a=a, page_range=(1, 8)), a + 1)
    assert torch.equal(k(a=a, page_range=None), a + 1)
    assert seen == [True, False]
    assert plain.calls == 2
