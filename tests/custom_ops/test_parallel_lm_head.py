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

"""
Test SpyreParallelLMHead custom op correctness against a reference implementation.
"""

import sys
import warnings

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

# FP8 ``_scaled_mm`` fails to compile for K < 256
# (``No system in place for splatting of data formats with <16b``).
_FP8_LM_HEAD_HIDDEN = 256
_FP8_LM_HEAD_HIDDEN_MM = 4096


def reference_lm_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Golden reference: standard F.linear as used by upstream ParallelLMHead."""
    return F.linear(x, weight, bias)


@pytest.mark.parallel_lm_head
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("vocab_size", [64, 128, 49216, 51200])
@pytest.mark.parametrize("embedding_dim", [64, 128])
def test_spyre_parallel_lm_head_matches_reference(tp_group, num_tokens, vocab_size, embedding_dim):
    """SpyreUnquantizedLMHeadMethod.apply output matches a plain F.linear reference.

    Exercises the full padded-weight path: checkpoint values are written into
    layer.weight, padded_weight_t is materialized in process_weights_after_loading,
    and quant_method.apply runs the Spyre matmul and eagerly unpads the logits.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops.parallel_lm_head import SpyreParallelLMHead

    torch.manual_seed(42)

    layer = ParallelLMHead(vocab_size, embedding_dim, params_dtype=torch.float16)
    assert isinstance(layer, SpyreParallelLMHead)

    # Simulate checkpoint loading: copy known values into the existing Parameter.
    loaded = torch.randn(layer.weight.shape, dtype=torch.float16)
    layer.weight.data.copy_(loaded)

    # Materialize padded_weight from the now-populated weight, as the loader would.
    layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, embedding_dim, dtype=torch.float16)
    expected = reference_lm_head(x, layer.weight.data)

    # In production weights live on Spyre after `model.to(spyre_device)`;
    # mirror that here so the H2D + Spyre matmul actually run.
    layer = layer.to("spyre")
    actual = layer.quant_method.apply(layer, x.to("spyre"))

    assert actual.shape == (num_tokens, layer.weight.shape[0])
    # Spyre matmul accumulation order diverges from the CPU reference in fp16;
    # see the "expect numerical differences" warning in
    # SpyreUnquantizedLMHeadMethod.process_weights_after_loading.
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-1, rtol=5e-2)


# ---------------------------------------------------------------------------
# Padding-workaround tests
#
# These tests cover a temporary workaround for a torch-spyre work-division
# limitation: matmul shapes must be a multiple of 64 * (k * 32), where k is
# an integer. Once torch-spyre lifts that restriction, the workaround in
# SpyreUnquantizedLMHeadMethod.process_weights_after_loading and the tests
# below (marked `padding_workaround`) can be removed.
# ---------------------------------------------------------------------------


@pytest.mark.parallel_lm_head
@pytest.mark.padding_workaround
@pytest.mark.parametrize(
    "vocab_size, expect_padding, expect_padded_shape",
    [
        (49216, True, 51200),  # 49216 = 64 * (24.03125 * 32) → needs padding to 51200
        (51200, False, 51200),  # 51200 = 64 * (25 * 32) → already aligned, no padding
    ],
)
def test_padded_weight_reflects_loaded_weight(
    tp_group, vocab_size, expect_padding, expect_padded_shape
):
    """padded_weight_t must hold the loaded checkpoint values, not uninitialized data.

    Regression guard: the padded weight was previously snapshotted in __init__,
    before load_weights ran, so it held whatever torch.empty produced. It is
    now materialized in process_weights_after_loading instead.

    padded_weight_t is stored transposed ([embedding_dim, padded_vocab]) so the
    forward GEMM is the Spyre-fast `x @ A`; the vocab padding lands on the
    trailing columns.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    embedding_dim = 64
    layer = ParallelLMHead(vocab_size, embedding_dim, params_dtype=torch.float16)

    loaded = torch.randn(layer.weight.shape, dtype=torch.float16)
    layer.weight.data.copy_(loaded)

    layer.quant_method.process_weights_after_loading(layer)

    vocab = layer.weight.shape[0]
    if expect_padding:
        assert layer.spyre_row_padding > 0
        assert layer.padded_weight_t.shape == (
            embedding_dim,
            expect_padded_shape,
        )
        # Leading columns mirror the loaded weight (transposed) bit-for-bit.
        torch.testing.assert_close(
            layer.padded_weight_t[:, :vocab],
            layer.weight.t(),
            atol=0.0,
            rtol=0.0,
        )
        # Padding columns are zeros (F.pad default), so they contribute 0 to logits.
        assert torch.all(layer.padded_weight_t[:, vocab:] == 0)
    else:
        # Aligned shape: no padding applied, padded_weight_t is just weightᵀ.
        assert layer.spyre_row_padding == 0
        torch.testing.assert_close(
            layer.padded_weight_t,
            layer.weight.t(),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parallel_lm_head
def test_lm_head_oot_dispatch(tp_group):
    """Verify ParallelLMHead OOT registration: class swap + quant_method swap."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops.parallel_lm_head import (
        SpyreParallelLMHead,
        SpyreUnquantizedLMHeadMethod,
    )

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)

    # OOT class swap: ParallelLMHead.__new__ should produce SpyreParallelLMHead.
    assert isinstance(layer, SpyreParallelLMHead)
    # quant_method swap: unquantized method is replaced with the Spyre-routing one.
    assert isinstance(layer.quant_method, SpyreUnquantizedLMHeadMethod)


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_lm_head_fp8_config_stays_fp16(tp_group):
    """``Fp8Config`` never quantizes the head upstream, so neither does Spyre."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops.parallel_lm_head import (
        SpyreParallelLMHead,
        SpyreUnquantizedLMHeadMethod,
    )

    layer = ParallelLMHead(
        128, _FP8_LM_HEAD_HIDDEN, params_dtype=torch.float16, quant_config=Fp8Config()
    )

    assert isinstance(layer, SpyreParallelLMHead)
    assert type(layer.quant_method) is SpyreUnquantizedLMHeadMethod
    assert layer.weight.dtype == torch.float16


def _apply_tracked(layer):
    """Run `layer._apply` with an identity fn; return the tensors handed to it."""
    seen: list[torch.Tensor] = []
    layer._apply(lambda t: (seen.append(t), t)[1])
    return seen


@pytest.mark.parallel_lm_head
def test_lm_head_apply_moves_weight_before_process(tp_group):
    """Before `process_weights_after_loading`, `padded_weight_t` is absent and `weight`
    is still live, so `_apply` must move it rather than strand it on host."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)
    assert not hasattr(layer, "padded_weight_t")
    assert any(t is layer.weight for t in _apply_tracked(layer))


@pytest.mark.parallel_lm_head
def test_lm_head_apply_skips_dead_weight(tp_group):
    """Once `padded_weight_t` exists, `_apply` moves it and skips the dead `weight`,
    restoring `weight` as the same registered Parameter."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)
    layer.quant_method.process_weights_after_loading(layer)

    original = layer.weight
    seen = _apply_tracked(layer)

    assert any(t is layer.padded_weight_t for t in seen), "padded_weight_t was not moved"
    assert not any(t is original for t in seen), "dead weight should be skipped"
    assert layer.weight is original and "weight" in layer._parameters


@pytest.mark.parallel_lm_head
def test_lm_head_weight_stays_on_cpu_after_to_spyre(tp_group):
    """Real device move: the dead `weight` stays on CPU; `padded_weight_t` goes to Spyre.
    Skipped on CPU-only hosts, where off-device placement is not observable."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)
    layer.quant_method.process_weights_after_loading(layer)

    layer.to("spyre")

    assert layer.weight.device.type == "cpu"
    assert layer.padded_weight_t.device.type == "spyre"


@pytest.mark.parallel_lm_head
@pytest.mark.padding_workaround
def test_non_aligned_weight_is_padded(tp_group):
    """process_weights_after_loading pads weight rows not divisible by ALIGN.

    Part of the padding workaround — remove together with the other
    `padding_workaround` tests once torch-spyre lifts the shape restriction.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    ALIGN = 64 * 32

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)

    original = torch.randn(63, 64, dtype=torch.float16)
    layer.weight = torch.nn.Parameter(original.clone(), requires_grad=False)

    layer.quant_method.process_weights_after_loading(layer)

    expected_padded_rows = ALIGN  # ceil(63 / ALIGN) * ALIGN
    # padded_weight_t is transposed: [embedding_dim, padded_vocab].
    assert layer.padded_weight_t.shape[1] == expected_padded_rows
    assert layer.spyre_row_padding == expected_padded_rows - 63
    # Original values preserved in the leading columns (transposed)
    torch.testing.assert_close(layer.padded_weight_t[:, :63], original.t(), atol=0.0, rtol=0.0)
    # Padding columns are zeros
    assert torch.all(layer.padded_weight_t[:, 63:] == 0)


@pytest.mark.parallel_lm_head
@pytest.mark.padding_workaround
def test_padded_matmul_and_unpad_slice_run_on_device(spyre_or_cpu_device):
    """The transposed matmul and the un-pad slice run on-device eagerly.

    SpyreUnquantizedLMHeadMethod.apply does `x @ weight_t` on a padding-aligned
    output dim then slices off the trailing pad columns. Post torch-spyre #3578
    the un-pad slice lowers on-device in eager mode (the storage offset is
    honored), so no torch.compile is needed. This isolates that primitive pair
    (matmul + trailing-slice) from the full layer path.
    """
    ALIGN = 64 * 32
    vocab = 32000
    padding = (-vocab) % ALIGN
    hidden = torch.randn(32, 4096, dtype=torch.float16) * 0.01
    weight_t = torch.randn(4096, vocab + padding, dtype=torch.float16) * 0.01

    def project(x, w):
        out = torch.matmul(x, w)
        return out[:, :-padding]

    expected = torch.matmul(hidden, weight_t)[:, :-padding]
    actual = project(hidden.to(spyre_or_cpu_device), weight_t.to(spyre_or_cpu_device))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parallel_lm_head
@pytest.mark.parametrize("scale", [1.0, 1.0 / 6.0, 2.0])
def test_spyre_logits_processor_scaling(tp_group, spyre_or_cpu_device, scale):
    """SpyreLogitsProcessor matches upstream reference for logits_scaling.

    Granite 3.3 sets logits_scaling, so LogitsProcessor.forward runs an in-place
    `logits *= self.scale` — on the host, as SpyreLogitsProcessor returns CPU logits.
    """

    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

    torch.manual_seed(42)

    vocab_size = 32000
    embedding_dim = 4096
    num_tokens = 8

    torch.manual_seed(43)
    # Small random values keep logits in a range where fp16 accumulation-order
    # differences between CPU and Spyre matmuls do not dominate the tolerance.
    weight = torch.randn(vocab_size, embedding_dim, dtype=torch.float16) * 0.01

    # Minimal fake LM head: just a linear weight with the right interface.
    class FakeLMHead:
        def __init__(self, weight_tensor):
            self.weight = weight_tensor
            # Upstream _get_logits reads lm_head.tp_size to decide whether to
            # gather across TP ranks; single-rank test, so no gather.
            self.tp_size = 1
            self.shard_indices = type(
                "SI", (), {"num_org_vocab_padding": 0, "org_vocab_start_index": 0}
            )()
            self.quant_method = type(
                "QM",
                (),
                {"apply": lambda self, layer, x, bias=None: F.linear(x, layer.weight, bias)},
            )()

    weight_device = weight.to(spyre_or_cpu_device)
    fake_head = FakeLMHead(weight_device)

    processor = LogitsProcessor(
        vocab_size=vocab_size,
        org_vocab_size=vocab_size,
        scale=scale,
    )
    assert isinstance(processor, SpyreLogitsProcessor)

    torch.manual_seed(44)
    hidden = torch.randn(num_tokens, embedding_dim, dtype=torch.float16) * 0.01
    hidden_spyre = hidden.to(spyre_or_cpu_device)

    # Reference: upstream logic on CPU.
    logits_ref = F.linear(hidden, weight)
    logits_ref = logits_ref[..., :vocab_size]
    logits_ref = logits_ref * scale

    # Spyre path.
    logits_out = processor(fake_head, hidden_spyre, embedding_bias=None)
    assert logits_out is not None

    torch.testing.assert_close(logits_out.cpu().float(), logits_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parallel_lm_head
@pytest.mark.parametrize("vocab_size", [128, 49216])
def test_tied_head_promoted_on_first_logits_call(tp_group, vocab_size):
    """An embedding used as the head gains a padded `Wᵀ`, leaving `weight` untouched."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod
    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    embedding_dim = 64
    torch.manual_seed(42)

    embed = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    assert not isinstance(embed.quant_method, SpyreUnquantizedLMHeadMethod)

    loaded = torch.randn(embed.weight.shape, dtype=torch.float16)
    embed.weight.data.copy_(loaded)
    embed.quant_method.process_weights_after_loading(embed)
    assert not hasattr(embed, "padded_weight_t")

    weight_before = embed.weight
    prepared_method = embed._spyre_lm_head_method
    promote_tied_lm_head(embed)

    assert isinstance(embed.quant_method, SpyreUnquantizedLMHeadMethod)
    assert embed.quant_method is prepared_method
    assert embed.weight is weight_before
    torch.testing.assert_close(embed.weight.data, loaded, atol=0.0, rtol=0.0)

    padded_vocab = vocab_size + embed.spyre_row_padding
    assert padded_vocab % (64 * 32) == 0
    assert embed.padded_weight_t.shape == (embedding_dim, padded_vocab)
    torch.testing.assert_close(
        embed.padded_weight_t[:, :vocab_size], loaded.t(), atol=0.0, rtol=0.0
    )


@pytest.mark.parallel_lm_head
def test_promotion_is_idempotent(tp_group):
    """`_apply_head` runs every decode step, so a second promotion must be a no-op."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    embed = VocabParallelEmbedding(49216, 64, params_dtype=torch.float16)
    embed.quant_method.process_weights_after_loading(embed)

    promote_tied_lm_head(embed)
    method, weight_t = embed.quant_method, embed.padded_weight_t
    promote_tied_lm_head(embed)

    assert embed.quant_method is method
    assert embed.padded_weight_t is weight_t


@pytest.mark.parallel_lm_head
def test_tied_head_projection_matches_reference(tp_group):
    """The promoted head's logits match F.linear, and its gather still works."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    vocab_size, embedding_dim, num_tokens = 49216, 64, 7
    torch.manual_seed(42)

    embed = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    embed.weight.data.normal_(std=0.02)
    embed.quant_method.process_weights_after_loading(embed)

    x = torch.randn(num_tokens, embedding_dim, dtype=torch.float16)
    input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
    logits_ref = reference_lm_head(x, embed.weight.data)
    gather_ref = F.embedding(input_ids, embed.weight)

    # Promote after the device move, as the first logits call does.
    embed = embed.to("spyre")
    promote_tied_lm_head(embed)
    assert embed.padded_weight_t.device.type == "spyre"

    logits = embed.quant_method.apply(embed, x.to("spyre"))
    gather = embed(input_ids.to("spyre"))

    assert logits.shape == (num_tokens, vocab_size)
    # Spyre matmul accumulation order diverges from the CPU reference in fp16.
    torch.testing.assert_close(logits.cpu().float(), logits_ref.float(), atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(gather.cpu().float(), gather_ref.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parallel_lm_head
def test_real_head_is_left_alone(tp_group):
    """A real ParallelLMHead already owns the projection; promotion must not touch it."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    embed = VocabParallelEmbedding(49216, 64, params_dtype=torch.float16)
    head = ParallelLMHead(49216, 64, params_dtype=torch.float16).tie_weights(embed)
    assert head.weight is embed.weight

    embed.weight.data.normal_(std=0.02)
    head.quant_method.process_weights_after_loading(head)
    method, weight_t = head.quant_method, head.padded_weight_t

    promote_tied_lm_head(head)

    assert head.quant_method is method
    assert head.padded_weight_t is weight_t
    # The tied table is only ever gathered from, so it never grows a projection.
    assert not hasattr(embed, "padded_weight_t")


@pytest.mark.parallel_lm_head
def test_gather_only_tables_are_never_promoted(tp_group):
    """Only the table handed to the logits processor is promoted.

    Gemma 3n builds a second, gather-only `embed_tokens_per_layer` under the same
    tied config; duplicating that table transposed would be pure waste.
    """
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod

    torch.manual_seed(42)
    embed_tokens = VocabParallelEmbedding(49216, 64, params_dtype=torch.float16)
    per_layer = VocabParallelEmbedding(2048, 64, params_dtype=torch.float16)
    for table in (embed_tokens, per_layer):
        table.weight.data.normal_(std=0.02)
        table.quant_method.process_weights_after_loading(table)

    processor = LogitsProcessor(vocab_size=49216, org_vocab_size=49216)
    processor(embed_tokens, torch.randn(4, 64, dtype=torch.float16), embedding_bias=None)

    assert isinstance(embed_tokens.quant_method, SpyreUnquantizedLMHeadMethod)
    assert not isinstance(per_layer.quant_method, SpyreUnquantizedLMHeadMethod)
    assert not hasattr(per_layer, "padded_weight_t")


@pytest.mark.parallel_lm_head
def test_projection_compiles_itself_and_matches_eager(tp_group, monkeypatch, spyre_or_cpu_device):
    """No graph encloses the lm_head, so it compiles its own and must not drift.

    ``fullgraph=True`` also asserts no graph break: a break would silently fall back
    to eager per step.
    """
    import types

    from vllm.config import CompilationMode
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops import lazy_compile
    from spyre_inference.custom_ops.linear import SpyreTransposedWeightMethod
    from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod

    torch.manual_seed(0)
    vocab_size, embedding_dim, rows = 51200, 64, 8

    layer = ParallelLMHead(vocab_size, embedding_dim, params_dtype=torch.float16)
    layer.weight.data.copy_(torch.randn(layer.weight.shape, dtype=torch.float16))
    layer.quant_method.process_weights_after_loading(layer)
    layer = layer.to(spyre_or_cpu_device)
    x = torch.randn(rows, embedding_dim, dtype=torch.float16).to(spyre_or_cpu_device)

    eager = SpyreTransposedWeightMethod.apply(layer.quant_method, layer, x)

    monkeypatch.setattr(
        lazy_compile,
        "get_cached_compilation_config",
        lambda: types.SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE),
    )
    compiled_method = SpyreUnquantizedLMHeadMethod()
    assert compiled_method.spyre_compiled_kernel is None

    actual = compiled_method.apply(layer, x)

    assert compiled_method.spyre_compiled_kernel is not None, "projection did not compile"
    assert actual.shape == (rows, vocab_size)
    torch.testing.assert_close(actual.cpu().float(), eager.cpu().float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parallel_lm_head
def test_projection_stays_eager_inside_an_enclosing_graph(tp_group, monkeypatch):
    """Blocks already compile their own linears; the wrapper must not nest a graph."""
    import types

    from vllm.config import CompilationMode

    from spyre_inference.custom_ops import lazy_compile
    from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod

    monkeypatch.setattr(
        lazy_compile,
        "get_cached_compilation_config",
        lambda: types.SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE),
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        torch, "compile", lambda *a, **k: pytest.fail("compiled inside an enclosing graph")
    )

    method = SpyreUnquantizedLMHeadMethod()
    layer = torch.nn.Module()
    layer.padded_weight_t = torch.randn(8, 16, dtype=torch.float16)
    layer.spyre_row_padding = 0

    out = method.apply(layer, torch.randn(3, 8, dtype=torch.float16))

    assert out.shape == (3, 16)
    assert method.spyre_compiled_kernel is None


def _cpu_gather_worker(rank: int, world_size: int, port: int, shards: list) -> None:
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps
    from vllm.distributed.parallel_state import GroupCoordinator

    from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    ps._TP = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="tp",
    )
    gathered = SpyreLogitsProcessor._cpu_all_gather_logits(shards[rank])
    torch.testing.assert_close(gathered, torch.cat(shards, dim=-1), atol=0, rtol=0)
    dist.destroy_process_group()


@pytest.mark.parallel_lm_head
def test_cpu_all_gather_logits_concatenates_the_vocab_shards_in_rank_order():
    import socket

    import torch.multiprocessing as mp

    world_size = 2
    torch.manual_seed(0)
    shards = [torch.randn(4, 64, dtype=torch.float16) for _ in range(world_size)]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    mp.spawn(_cpu_gather_worker, args=(world_size, port, shards), nprocs=world_size)


@pytest.fixture
def spyre_or_cpu_device():
    """Use Spyre if available, otherwise CPU."""
    try:
        torch.randn(1, device=torch.device("spyre"))
        return torch.device("spyre")
    except Exception:
        return torch.device("cpu")


def _fp8_args(strategy: str = "channel", **kwargs):
    from compressed_tensors.quantization import QuantizationArgs

    return QuantizationArgs(num_bits=8, type="float", strategy=strategy, **kwargs)


def _ct_config(target_scheme_map: dict, ignore: list[str] | None = None):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    return CompressedTensorsConfig(
        target_scheme_map=target_scheme_map, ignore=ignore or [], quant_format="float-quantized"
    )


def _w8a8(weights="channel", *, static_input: bool = False):
    acts = _fp8_args("tensor", dynamic=False) if static_input else _fp8_args("token", dynamic=True)
    return {"weights": _fp8_args(weights), "input_activations": acts}


def _ct_body_only():
    """Granite FP8 checkpoints: ``targets: ["Linear"]`` and ``ignore: ["lm_head"]``."""
    return _ct_config({"Linear": _w8a8()}, ignore=["lm_head"])


def _ct_fp8_head(weights="channel", *, static_input: bool = False):
    """A compressed-tensors checkpoint that also ships its head in FP8."""
    return _ct_config({"Linear": _w8a8(), "lm_head": _w8a8(weights, static_input=static_input)})


def _make_head(quant_config, vocab_size: int = 128, embedding_dim: int = 64):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops.fp8_linear_kernel import register_spyre_fp8_linear_kernel

    register_spyre_fp8_linear_kernel()
    return ParallelLMHead(
        vocab_size,
        embedding_dim,
        params_dtype=torch.float16,
        quant_config=quant_config,
        prefix="lm_head",
    )


def _compile_mode(monkeypatch, compiled: bool) -> None:
    from types import SimpleNamespace

    from vllm.config import CompilationMode

    from spyre_inference.custom_ops import lazy_compile

    mode = CompilationMode.STOCK_TORCH_COMPILE if compiled else CompilationMode.NONE
    monkeypatch.setattr(
        lazy_compile, "get_cached_compilation_config", lambda: SimpleNamespace(mode=mode)
    )


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
@pytest.mark.parametrize(
    ("make_config", "compiled", "expected"),
    [
        (lambda: None, True, "SpyreUnquantizedLMHeadMethod"),
        (_ct_body_only, True, "SpyreUnquantizedLMHeadMethod"),
        (lambda: _ct_config({"Linear": _w8a8()}), True, "SpyreUnquantizedLMHeadMethod"),
        (_ct_fp8_head, True, "SpyreFp8LMHeadMethod"),
        (lambda: _ct_fp8_head("tensor"), True, "SpyreFp8LMHeadMethod"),
        (lambda: _ct_fp8_head(static_input=True), True, "SpyreFp8LMHeadMethod"),
        (_ct_fp8_head, False, "SpyreFp8LMHeadMethod"),
    ],
    ids=[
        "no-quant",
        "ct-head-ignored",
        "ct-head-not-targeted",
        "ct-fp8-head",
        "ct-fp8-head-per-tensor",
        "ct-fp8-head-static-input",
        "ct-fp8-head-enforce-eager",
    ],
)
def test_construction_follows_the_checkpoint(
    tp_group, monkeypatch, make_config, compiled, expected
):
    """The head is FP8 only when the checkpoint ships it so, as upstream; no card needed."""
    _compile_mode(monkeypatch, compiled)
    head = _make_head(make_config())
    assert type(head.quant_method).__name__ == expected
    fp8 = expected == "SpyreFp8LMHeadMethod"
    assert (head.weight.dtype == torch.float8_e4m3fn) is fp8


def _scheme(name: str):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
        CompressedTensorsW8A16Fp8,
    )

    if name == "w8a16":
        return CompressedTensorsW8A16Fp8(_fp8_args(), is_static_input_scheme=False)
    weights = _fp8_args("block", block_structure=[128, 128]) if name == "block" else _fp8_args(name)
    return CompressedTensorsW8A8Fp8(weights, is_static_input_scheme=False)


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
@pytest.mark.parametrize(
    ("scheme", "expected"),
    [("channel", True), ("tensor", True), ("block", False), ("w8a16", False)],
)
def test_only_fp8_w8a8_heads_get_the_fp8_method(tp_group, scheme, expected):
    """The Spyre FP8 head serves FP8 W8A8 with channel or tensor weight scales.

    Asked of the scheme alone: building a head for any other scheme already fails in
    upstream's kernel selection, which has no Spyre kernel for it.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )

    from spyre_inference.custom_ops.parallel_lm_head import ships_fp8_head

    method = CompressedTensorsLinearMethod(_ct_config({}))
    head = SimpleNamespace(quant_method=method, scheme=_scheme(scheme))
    assert ships_fp8_head(head) is expected
    assert not ships_fp8_head(SimpleNamespace(quant_method=None, scheme=_scheme(scheme)))


def _load_checkpoint_head(head, *, seed: int = 0):
    """Fill the checkpoint parameters as the loader would: an FP8 ``weight`` with
    per-row scales, zero past the vocab, and return the FP16 weight they encode."""
    torch.manual_seed(seed)
    vocab = head.org_vocab_size
    w = torch.randn(vocab, head.weight.shape[1], dtype=torch.float32) * 0.01
    scale = w.abs().amax(dim=1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max
    q = (w / scale).to(torch.float8_e4m3fn)
    head.weight.data.zero_()
    head.weight.data[:vocab].copy_(q)
    head.weight_scale.data.zero_()
    head.weight_scale.data[:vocab].copy_(scale.reshape(head.weight_scale.data[:vocab].shape))
    return (q.float() * scale).to(torch.float16)


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
@pytest.mark.parametrize("weights", ["channel", "tensor"])
def test_checkpoint_scales_are_kept_as_loaded(tp_group, monkeypatch, weights):
    """One scale per row, per-tensor broadcast, values untouched: the weight was quantized
    against them, so a scale below float16's normal range must not be raised."""
    from spyre_inference.custom_ops.parallel_lm_head import checkpoint_fp8_scales

    _compile_mode(monkeypatch, True)
    head = _make_head(_ct_fp8_head(weights), vocab_size=100)
    head.weight_scale.data.fill_(1e-5)
    if weights == "channel":
        head.weight_scale.data[100:] = 0
    scale = checkpoint_fp8_scales(head)
    assert scale.dtype == torch.float32
    assert scale.shape == (head.weight.shape[0],)
    torch.testing.assert_close(scale[:100], torch.full((100,), 1e-5))
    if weights == "channel":
        assert bool((scale[100:] == 0).all())


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
@pytest.mark.parametrize("static_input", [False, True])
def test_fp8_head_keeps_the_checkpoint_weight(tp_group, monkeypatch, static_input):
    """Loading pads and transposes the checkpoint's FP8 bytes unchanged; nothing is
    requantized, and the checkpoint parameters are left for a reload."""
    from spyre_inference.custom_ops import parallel_lm_head

    _compile_mode(monkeypatch, True)
    # The DMA needs a card; the layout under test does not.
    monkeypatch.setattr(parallel_lm_head, "_dma_checkpoint_qfp8wt", lambda w: w.contiguous())
    head = _make_head(_ct_fp8_head(static_input=static_input))
    _load_checkpoint_head(head)
    if static_input:
        head.input_scale.data.copy_(torch.tensor([0.25]))
    loaded = head.weight.data.clone()
    head.quant_method.process_weights_after_loading(head)

    rows = loaded.shape[0]
    padding = head.spyre_row_padding
    assert head.padded_weight_t.shape == (loaded.shape[1], rows + padding)
    head_bytes = head.padded_weight_t.view(torch.uint8)
    assert torch.equal(head_bytes[:, :rows], loaded.t().view(torch.uint8))
    assert bool((head_bytes[:, rows:] == 0).all())
    scale = head.padded_weight_scale
    assert scale.dtype == torch.float16 and scale.shape == (1, rows + padding)
    assert bool((scale[:, rows:] == 0).all())
    assert torch.equal(head.weight.data.view(torch.uint8), loaded.view(torch.uint8))
    if static_input:
        assert head.padded_input_scale.tolist() == [0.25]
    else:
        assert not hasattr(head, "padded_input_scale")


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_fp8_head_compiles_under_enforce_eager(tp_group, monkeypatch):
    """``spyre.scaled_mm`` has no eager kernel, so the projection compiles regardless."""
    from spyre_inference.custom_ops import lazy_compile

    _compile_mode(monkeypatch, False)
    head = _make_head(_ct_fp8_head())
    compiled = []
    monkeypatch.setattr(
        lazy_compile.torch,
        "compile",
        lambda fn, **kwargs: compiled.append(fn) or (lambda *args, **kw: "compiled"),
    )
    assert head.quant_method._project(head, torch.zeros(2, 64)) == "compiled"
    assert len(compiled) == 1


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_checkpoint_parameters_stay_on_host(tp_group, monkeypatch):
    """Once loaded, the checkpoint's FP8 weight and scales are runtime-dead: ``.to`` skips
    them, so a reload writes host tensors, while the runtime scale follows the GEMM."""
    from spyre_inference.custom_ops import parallel_lm_head

    _compile_mode(monkeypatch, True)
    monkeypatch.setattr(parallel_lm_head, "_dma_checkpoint_qfp8wt", lambda w: w.contiguous())
    head = _make_head(_ct_fp8_head())
    _load_checkpoint_head(head)
    head.quant_method.process_weights_after_loading(head)
    moved = []
    head._apply(lambda t: moved.append(t) or t)
    moved_ids = {id(t) for t in moved}
    assert id(head.padded_weight_scale) in moved_ids
    for name in ("weight", "weight_scale"):
        assert id(getattr(head, name)) not in moved_ids, name
        assert name in head._parameters


def _build_fp8_lm_head(vocab_size: int, embedding_dim: int, *, seed: int = 42):
    """A checkpoint FP8 head, loaded: qfp8wt ``padded_weight_t`` on Spyre and
    ``padded_weight_scale``. Returns the head and the FP16 weight it encodes."""
    if not spyre_available():
        pytest.skip("Loading DMAs the head into qfp8wt, which needs Spyre")
    from spyre_inference.custom_ops.parallel_lm_head import SpyreFp8LMHeadMethod

    layer = _make_head(_ct_fp8_head(), vocab_size, embedding_dim)
    assert isinstance(layer.quant_method, SpyreFp8LMHeadMethod)
    encoded = _load_checkpoint_head(layer, seed=seed)
    layer.quant_method.process_weights_after_loading(layer)
    return layer, encoded


def _fp8_lm_head_apply(layer, x):
    """Call SpyreFp8LMHeadMethod.apply on Spyre; error on silent CPU fallback."""
    from torch_spyre.ops.fallbacks import FallbackWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        actual = layer.quant_method.apply(layer, x)
    assert actual.device.type == "spyre", actual.device
    return actual


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_fp8_loading_dmas_qfp8wt(tp_group):
    """Loading leaves a qfp8wt head on Spyre and a host per-column scale."""
    layer, _ = _build_fp8_lm_head(128, _FP8_LM_HEAD_HIDDEN)
    padded_vocab = layer.padded_weight_t.shape[1]
    assert layer.padded_weight_t.device.type == "spyre"
    assert layer.padded_weight_t.dtype == torch.float8_e4m3fn
    assert layer.padded_weight_scale.shape == (1, padded_vocab)
    assert layer.padded_weight_scale.device.type == "cpu"

    layer.to("spyre")
    assert layer.padded_weight_scale.device.type == "spyre"
    assert layer.weight.device.type == "cpu"
    assert layer.weight_scale.device.type == "cpu"


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_fp8_head_dma_round_trips_the_checkpoint_weight(tp_group):
    """The qfp8wt head read back is the checkpoint's FP8 bytes, padding zero."""
    layer, _ = _build_fp8_lm_head(49216, _FP8_LM_HEAD_HIDDEN)
    rows = layer.weight.shape[0]
    read_back = layer.padded_weight_t.cpu()
    assert torch.equal(read_back[:, :rows].view(torch.uint8), layer.weight.t().view(torch.uint8))
    assert bool((read_back[:, rows:].view(torch.uint8) == 0).all())


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_fp8_reload_after_the_device_move_still_projects(tp_group):
    """A reload after ``model.to("spyre")`` rebuilds the head from the host checkpoint
    parameters and keeps the new scale beside the qfp8wt head."""
    embedding_dim = _FP8_LM_HEAD_HIDDEN_MM
    layer, encoded = _build_fp8_lm_head(49216, embedding_dim)
    layer = layer.to("spyre")
    layer.weight_scale.data.mul_(2)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.padded_weight_scale.device.type == "spyre"

    x = torch.randn(4, embedding_dim, dtype=torch.float16) * 0.01
    expected = reference_lm_head(x, encoded * 2)
    actual = _fp8_lm_head_apply(layer, x.to("spyre"))
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=0.02, rtol=0.01)


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_fp8_lm_head_apply_rejects_bias(tp_group):
    """bias is not supported; raise before any Spyre GEMM."""
    layer, _ = _build_fp8_lm_head(128, _FP8_LM_HEAD_HIDDEN)
    x = torch.randn(2, _FP8_LM_HEAD_HIDDEN, dtype=torch.float16)
    bias = torch.zeros(128, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="embedding_bias"):
        layer.quant_method.apply(layer, x, bias)


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
@pytest.mark.parametrize("num_tokens", [1, 4, 7])
@pytest.mark.parametrize("vocab_size", [49216, 51200])
def test_fp8_lm_head_apply_matches_reference(tp_group, num_tokens, vocab_size):
    """SpyreFp8LMHeadMethod.apply matches F.linear on the weight the checkpoint encodes.

    Hidden=4096 so torch-spyre can compile FP8 ``scaled_mm``. Vocab 49216/51200 hits the
    ``spyre_row_padding`` unpad, and the whole padded vocab is one GEMM. The activation is
    quantized per token, so the tolerance covers its e4m3 error only.
    """
    embedding_dim = _FP8_LM_HEAD_HIDDEN_MM
    layer, encoded = _build_fp8_lm_head(vocab_size, embedding_dim)
    x = torch.randn(num_tokens, embedding_dim, dtype=torch.float16) * 0.01
    expected = reference_lm_head(x, encoded)

    layer = layer.to("spyre")
    actual = _fp8_lm_head_apply(layer, x.to("spyre"))

    assert actual.dtype == torch.float16
    assert actual.shape == (num_tokens, vocab_size)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=0.01, rtol=0.01)


@pytest.mark.parallel_lm_head
@pytest.mark.fp8
def test_fp8_lm_head_apply_rejects_3d(tp_group):
    """3-D input is rejected with NotImplementedError.

    Column-slice of a ``scaled_mm`` output followed by ``reshape`` reads garbage on
    Spyre (torch-spyre bug — the 3-D branch is unreachable in serving).
    """
    layer, _ = _build_fp8_lm_head(49216, _FP8_LM_HEAD_HIDDEN_MM)
    x = torch.randn(2, 3, _FP8_LM_HEAD_HIDDEN_MM, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="2-D"):
        layer.quant_method.apply(layer, x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
