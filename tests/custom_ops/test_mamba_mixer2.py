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

"""Spyre ``MambaMixer2`` kernels against an fp32 token-by-token recurrence.

The kernels read and write the state slots in-graph, so each case checks the slots it
updated and that it left every other slot alone. The recurrence is vLLM's reference
math (``causal_conv1d_ref`` / ``selective_state_update_ref``) unrolled per token, so a
prefill that runs it in chunks has to agree with it at every chunk boundary. ``dt_bias``
is drawn as Nemotron's checkpoints have it, ``softplus(dt)`` in ``[1e-3, 0.1]``, which
puts most inputs where the Spyre softplus primitive alone used to return 0.

Shapes are Nemotron-3-Nano's per-card ones at TP4: 16 heads of 64, 2 groups of state
128, conv width 4.
"""

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

pytestmark = pytest.mark.skipif(not spyre_available(), reason="needs a Spyre card")

H, P, G, N, K = 16, 64, 2, 128, 4
INNER, GS = H * P, G * N
CD = INNER + 2 * GS
WIDTH = INNER + CD + H
SLOTS = 9
# fp16 inputs and state on device: relative L2 error against fp32 stays near 1e-2.
_RTOL = 2e-2


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.cpu().float() - expected).norm() / expected.norm()).item()


@pytest.fixture()
def params():
    gen = torch.Generator().manual_seed(0)
    rnd = lambda *s, sc=0.5: (torch.randn(*s, generator=gen) * sc).half()  # noqa: E731
    dt = torch.empty(H).uniform_(1e-3, 0.1, generator=gen)
    return {
        "conv_w": rnd(CD, K, sc=0.3),
        "conv_b": rnd(CD, sc=0.1),
        "A": -torch.empty(H).uniform_(1, 16, generator=gen),
        "D": rnd(H),
        "dt_bias": dt + torch.log(-torch.expm1(-dt)),
        "rnd": rnd,
    }


def _recurrence(rows, conv_state, ssm_state, p):
    """fp32 reference over ``rows`` (``[T, WIDTH]``) from the given states."""
    rows = rows.float()
    tokens = rows.shape[0]
    window = torch.cat([conv_state.float(), rows[:, INNER : INNER + CD]], 0)
    taps = sum(window[k : k + tokens] * p["conv_w"].float()[:, k] for k in range(K))
    x, b, c = F.silu(taps + p["conv_b"].float()).split([INNER, GS, GS], -1)
    dt = F.softplus(rows[:, INNER + CD :] + p["dt_bias"])
    b = b.view(tokens, G, N).repeat_interleave(H // G, 1)
    c = c.view(tokens, G, N).repeat_interleave(H // G, 1)
    state = ssm_state.float().clone()
    ys = []
    for t in range(tokens):
        xt = x[t].view(H, P)
        state = state * torch.exp(dt[t] * p["A"])[:, None, None]
        state = state + (dt[t][:, None] * xt)[..., None] * b[t][:, None, :]
        ys.append(torch.einsum("hpn,hn->hp", state, c[t]) + p["D"].float()[:, None] * xt)
    return torch.stack(ys).reshape(tokens, INNER), window[tokens:], state


class _Mixer:
    """The attributes ``SpyreMambaMixer2``'s ``_decode`` / ``_prefill`` read, without
    building the vLLM layer (which needs a TP group, a model config and weight loading).
    """

    tp_size = 1
    num_heads, head_dim, n_groups, ssm_state_size = H, P, G, N
    conv_kernel_size = K
    tped_intermediate_size, tped_conv_size, tped_dt_size = INNER, CD, H
    groups_ssm_state_size = GS

    def __init__(self, p, chunk, conv_cache, ssm_cache, max_num_seqs=4):
        from spyre_inference.custom_ops import mamba_mixer2 as m

        cls = m.SpyreMambaMixer2
        for name in (
            "_consts",
            "_pad_rows",
            "_out_staging",
            "_flush_output",
            "_dims",
            "_decode",
            "_prefill",
            "warm_kernels",
        ):
            setattr(self, name, getattr(cls, name).__get__(self))
        self.chunk_size = chunk
        self.decode_buckets = m.decode_buckets(max_num_seqs)
        self._spyre_consts = self._spyre_staging = self._spyre_out = None
        self.A, self.D, self.dt_bias = p["A"], p["D"].float(), p["dt_bias"]
        self.conv1d = torch.nn.Module()
        self.conv1d.weight = torch.nn.Parameter(p["conv_w"].float()[:, None, :])
        self.conv1d.bias = torch.nn.Parameter(p["conv_b"].float())
        self.kv_cache = (conv_cache.to("spyre"), ssm_cache.to("spyre"))


@pytest.mark.parametrize("batch", [1, 3, 4])
def test_decode_step_matches_the_recurrence(params, batch):
    """A decode batch, padded to the next bucket, through the layer's own dispatch.

    ``projected_states`` carries more rows than the batch (the body bucket), as in serving.
    """
    rnd = params["rnd"]
    conv_cache, ssm_cache = rnd(SLOTS, K - 1, CD), rnd(SLOTS, H, P, N, sc=0.1)
    slots = torch.tensor([3, 7, 1, 5][:batch])
    rows = rnd(8, WIDTH)
    rows[:, INNER + CD :] = rnd(8, H, sc=1.0)
    expected = [
        _recurrence(rows[i : i + 1], conv_cache[s], ssm_cache[s], params)
        for i, s in enumerate(slots.tolist())
    ]
    mixer = _Mixer(params, 128, conv_cache, ssm_cache)
    out = torch.zeros(8, INNER, dtype=torch.float16).to("spyre")
    mixer._decode(rows.to("spyre"), batch, slots, out)
    mixer._flush_output(out)
    y, d_conv, d_ssm = out.cpu(), mixer.kv_cache[0].cpu(), mixer.kv_cache[1].cpu()
    for i, s in enumerate(slots.tolist()):
        assert _rel(y[i : i + 1], expected[i][0]) < _RTOL
        assert _rel(d_conv[s], expected[i][1]) < _RTOL
        assert _rel(d_ssm[s], expected[i][2]) < _RTOL
    # Slot 0 is the null block that padded rows update.
    untouched = [s for s in range(1, SLOTS) if s not in slots.tolist()]
    # Only the fp16 host/device round trip: dl16 drops one bit of fp16.
    torch.testing.assert_close(d_ssm[untouched], ssm_cache[untouched], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(d_conv[untouched], conv_cache[untouched], atol=4e-3, rtol=1e-3)


@pytest.mark.parametrize("tokens", [37, 128, 2 * 128 + 37])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_prefill_chunks_match_the_recurrence(params, tokens, has_initial_state):
    """Chunked from a state slot and back, a partial last chunk padded to the kernel.

    The sequence starts at a non-zero row of a larger ``projected_states``, as a
    prefill does behind a step's decode rows.
    """
    rnd = params["rnd"]
    slot, start = 6, 3
    rows = rnd(start + tokens + 5, WIDTH)
    rows[:, INNER + CD :] = rnd(rows.shape[0], H, sc=1.0)
    conv_cache, ssm_cache = rnd(SLOTS, K - 1, CD), rnd(SLOTS, H, P, N, sc=0.1)
    init_conv = conv_cache[slot] if has_initial_state else torch.zeros(K - 1, CD)
    init_ssm = ssm_cache[slot] if has_initial_state else torch.zeros(H, P, N)
    y_ref, conv_ref, ssm_ref = _recurrence(
        rows[start : start + tokens], init_conv, init_ssm, params
    )

    mixer = _Mixer(params, 128, conv_cache, ssm_cache)
    out = torch.zeros(rows.shape[0], INNER, dtype=torch.float16).to("spyre")
    mixer._prefill(rows.to("spyre"), start, start + tokens, slot, has_initial_state, out)
    mixer._flush_output(out)
    assert _rel(out.cpu()[start : start + tokens], y_ref) < _RTOL
    assert _rel(mixer.kv_cache[0].cpu()[slot], conv_ref) < _RTOL
    assert _rel(mixer.kv_cache[1].cpu()[slot], ssm_ref) < _RTOL
