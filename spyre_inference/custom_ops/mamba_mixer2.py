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

"""Spyre OOT ``MambaMixer2``: upstream's layer, with a Spyre body for its conv + SSM op.

Upstream implements ``conv_ssm_forward`` only with Triton and x86 C++ kernels. Everything
else is upstream's: construction, TP sharding, weight loaders, state shapes and cache
spec, ``in_proj`` / ``out_proj``, the gated RMSNorm and the opaque ``mamba_mixer2`` op
the block graph calls. The body is vLLM's reference math (``causal_conv1d_ref``,
``selective_state_update_ref`` and the ``ssd_minimal_discrete`` chunk scan in
``tests/kernels/mamba``), written in forms torch-spyre lowers:

* the depthwise conv is a sum of ``K`` shifted taps over a tap-major window, the
  layout of vLLM's default ``SD`` conv state;
* B and C go from groups to heads through a constant one-hot ``[H, G]`` matmul;
* per-head ``A``, ``D`` and ``dt_bias`` are expanded along head_dim once, at first use;
* cumulative and segment sums are matmuls with constant triangular matrices.

Two compiled kernels do the work, each reading and writing the state slots in-graph
(eager indexing has no Spyre kernel): one per decode batch, padded to a power of two,
and one per prefill chunk of ``chunk_size`` tokens, so prefill compiles once.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2, Mixer2RMSNormGated
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.worker import compile_guard

logger = init_logger(__name__)

# softplus(-30) is 9e-14, an exact zero in fp16: a padded prefill row with this dt and
# zero x/B/C leaves the SSM state untouched.
_PAD_DT = -30.0


def _split(rows: torch.Tensor, intermediate: int, conv_dim: int):
    """``in_proj`` rows are ``[gate | xBC | dt]``; the gate is applied by the norm."""
    _, xbc, dt = rows.split([intermediate, conv_dim, rows.shape[-1] - intermediate - conv_dim], -1)
    return xbc, dt


def _decode_step(
    projected: torch.Tensor,
    pad: torch.Tensor,
    row_ids: torch.Tensor,
    out_ids: torch.Tensor,
    out: torch.Tensor,
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    slots: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    E: torch.Tensor,
    intermediate: int,
    group_state: int,
    head_dim: int,
) -> None:
    """One token for each of ``B`` sequences: conv-state shift, conv, state update.

    Rows ``row_ids`` [B] of ``cat(projected, pad)`` are the step's in_proj outputs
    ``[I + conv_dim + H]``, written to rows ``out_ids`` of ``out`` ``[T + 1, I]``, whose last
    row takes the padded ones; conv_cache [S, K-1, conv_dim], ssm_cache [S, H, P, N],
    slots [B]; conv_w [K, conv_dim], dt_bias / D [H, P], A [H, P, N], E [H, G].
    """
    rows = torch.cat([projected, pad], 0).index_select(0, row_ids)
    bs = rows.shape[0]
    heads, groups = E.shape
    n = group_state // groups
    xbc, dt = _split(rows, intermediate, conv_cache.shape[-1])
    window = torch.cat([conv_cache[slots], xbc.unsqueeze(1)], 1)
    xbc = F.silu((window * conv_w).sum(1) + conv_b)
    x, b, c = xbc.split([intermediate, group_state, group_state], -1)
    x = x.view(bs, heads, head_dim)
    # The [B, G*N] -> [B, G, N] view must be materialised before the matmul.
    b = torch.matmul(E, b.view(bs, groups, n) * 1)
    c = torch.matmul(E, c.view(bs, groups, n) * 1)
    dt = F.softplus(dt[:, :, None].expand(bs, heads, head_dim) + dt_bias)
    state = ssm_cache[slots] * torch.exp(dt[..., None] * A)
    state = state + dt[..., None] * b[:, :, None, :] * x[..., None]
    y = (state * c[:, :, None, :]).sum(-1) + x * D
    conv_cache[slots] = window[:, 1:]
    ssm_cache[slots] = state
    out[out_ids] = y.reshape(bs, intermediate)


def _prefill_chunk(
    projected: torch.Tensor,
    pad: torch.Tensor,
    row_ids: torch.Tensor,
    out_ids: torch.Tensor,
    out: torch.Tensor,
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    slot: torch.Tensor,
    init: torch.Tensor,
    conv_rows: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    E: torch.Tensor,
    tril: torch.Tensor,
    stril: torch.Tensor,
    striu: torch.Tensor,
    intermediate: int,
    group_state: int,
    head_dim: int,
) -> None:
    """One SSD chunk of ``L`` tokens of one sequence, from and back to its state slot.

    Rows ``row_ids`` [L] of ``cat(projected, pad)`` are the chunk's tokens, padded with
    rows that leave the states unchanged, and are written to rows ``out_ids`` of
    ``out`` [T + 1, I], whose last row takes the padded ones; slot [1], init [1] (0 starts from zero
    states), conv_rows [K-1] (window rows that become the conv state); dt_bias / A [H],
    D [H, P], tril / stril / striu [L, L].
    """
    rows = torch.cat([projected, pad], 0).index_select(0, row_ids)
    length = rows.shape[0]
    taps = conv_w.shape[0]
    heads, groups = E.shape
    n = group_state // groups
    xbc, dt = _split(rows, intermediate, conv_cache.shape[-1])
    xp = torch.cat([conv_cache[slot][0] * init, xbc], 0)
    acc = xp[0:length] * conv_w[0]
    for k in range(1, taps):
        acc = acc + xp[k : k + length] * conv_w[k]
    xbc = F.silu(acc + conv_b)
    x, b, c = xbc.split([intermediate, group_state, group_state], -1)
    # One restickify: x is read token-major below and head-major in the matmuls.
    x = x.view(length, heads, head_dim).clone()
    dt = F.softplus(dt + dt_bias)
    da = dt * A
    cs = torch.matmul(tril, da)
    seg = torch.exp(torch.matmul(tril, da.t()[:, :, None] * stril)) * tril
    bt = torch.matmul(E, b.view(length, groups, n) * 1).transpose(0, 1)
    ct = torch.matmul(E, c.view(length, groups, n) * 1).transpose(0, 1)
    xdt = (x * dt[..., None]).transpose(0, 1)
    state0 = ssm_cache[slot][0] * init
    y = torch.matmul(torch.matmul(ct, bt.transpose(1, 2)) * seg, xdt)
    y = y + torch.matmul(ct, state0.transpose(1, 2)) * torch.exp(cs.t())[..., None]
    y = (y + x.transpose(0, 1) * D[:, None, :]).transpose(0, 1)
    decay_to_end = torch.exp(torch.matmul(striu, da))
    state = state0 * torch.exp(cs[length - 1 : length]).reshape(heads, 1, 1)
    state = state + torch.matmul((x * (dt * decay_to_end)[..., None]).permute(1, 2, 0), bt)
    conv_cache[slot] = xp.index_select(0, conv_rows).unsqueeze(0)
    ssm_cache[slot] = state.unsqueeze(0)
    out[out_ids] = y.reshape(length, intermediate)


_decode_compiled = torch.compile(_decode_step, dynamic=False)
_prefill_compiled = torch.compile(_prefill_chunk, dynamic=False)
compile_guard.watch(_decode_step, "Mamba2 decode kernel")
compile_guard.watch(_prefill_chunk, "Mamba2 prefill chunk kernel")


def decode_buckets(max_num_seqs: int) -> list[int]:
    """Decode batch sizes the kernel compiles at: powers of two, and ``max_num_seqs``."""
    sizes = {max_num_seqs}
    size = 1
    while size < max_num_seqs:
        sizes.add(size)
        size *= 2
    return sorted(sizes)


@Mixer2RMSNormGated.register_oot(name="Mixer2RMSNormGated")
class SpyreMixer2RMSNormGated(Mixer2RMSNormGated):
    """Upstream's norm with both operands materialised in fp32 before the gate product.

    Upstream promotes the fp16 mixer output through its product with the fp32 gate.
    torch-spyre cannot lower that mixed-dtype product right after an opaque op
    (``No FX node for bufN``); it lowers once both operands are fp32. The gate is a
    column slice of ``in_proj``'s output, whose width need not be a multiple of the
    64-element stick (Nemotron-H-4B: 16496); upcasting such a slice directly returns
    wrong values, so it is copied out first.
    """

    def forward_native(self, x: torch.Tensor, gate: torch.Tensor):
        return super().forward_native(x.to(torch.float32), gate.clone()).to(x.dtype)


@MambaMixer2.register_oot(name="MambaMixer2")
class SpyreMambaMixer2(MambaMixer2):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_replayssm or self.num_spec > 0:
            raise NotImplementedError(
                "Spyre MambaMixer2 supports neither ReplaySSM nor speculative decoding."
            )
        if self.activation not in ("silu", "swish"):
            raise NotImplementedError(
                f"Spyre MambaMixer2 does not support activation {self.activation!r}."
            )
        # conv1d is a linear layer only to reuse its sharded loaders; its weight is the
        # 3-D depthwise kernel, which the Spyre GEMM method must not transpose.
        self.conv1d.quant_method = UnquantizedLinearMethod()
        assert self.model_config is not None
        self.chunk_size = self.model_config.get_mamba_chunk_size()
        self.decode_buckets = decode_buckets(
            get_current_vllm_config().scheduler_config.max_num_seqs
        )
        self._spyre_consts: dict[str, torch.Tensor] | None = None
        self._spyre_staging: dict[str, torch.Tensor] | None = None
        self._spyre_out: torch.Tensor | None = None

    def bind_kv_cache(self, kv_cache: Any) -> None:
        # Spyre allocates each state as its own tensor (kv_cache_setup), not as a raw
        # page that upstream would re-view per state.
        if isinstance(kv_cache, tuple):
            self.kv_cache = tuple(kv_cache)
            return
        super().bind_kv_cache(kv_cache)

    def _consts(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        if self._spyre_consts is not None:
            return self._spyre_consts
        heads = self.num_heads // self.tp_size
        groups = self.n_groups // self.tp_size
        head_dim, state = self.head_dim, self.ssm_state_size

        def host(t: torch.Tensor) -> torch.Tensor:
            return t.detach().to("cpu", torch.float32)

        A, D, dt_bias = host(self.A), host(self.D), host(self.dt_bias)
        # A = -exp(A_log) can exceed fp16 (Nemotron-H-4B: -2e6) and would become -inf,
        # then NaN in the segment sums. Such a head forgets its state within a step, which
        # the fp16 maximum still does: exp(dt * -65504) is 0 for any dt above 2e-4.
        A = A.clamp(min=torch.finfo(dtype).min)
        E = torch.zeros(heads, groups)
        E[torch.arange(heads), torch.arange(heads) // (heads // groups)] = 1
        ones = torch.ones(self.chunk_size, self.chunk_size)
        bias = self.conv1d.bias
        consts = {
            # conv1d.weight is [conv_dim, 1, K]; the kernels read taps first.
            "conv_w": host(cast(torch.Tensor, self.conv1d.weight))[:, 0, :].t().contiguous(),
            "conv_b": host(bias) if bias is not None else torch.zeros(self.tped_conv_size),
            "dt_bias_d": dt_bias[:, None].expand(heads, head_dim).contiguous(),
            "A_d": A[:, None, None].expand(heads, head_dim, state).contiguous(),
            "D_d": D[:, None].expand(heads, head_dim).contiguous(),
            "dt_bias": dt_bias,
            "A": A,
            "E": E,
            "tril": torch.tril(ones),
            "stril": torch.tril(ones, diagonal=-1),
            "striu": torch.triu(ones, diagonal=1),
        }
        self._spyre_consts = {k: convert(v.to(dtype), device=device) for k, v in consts.items()}
        return self._spyre_consts

    def _pad_rows(self, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Rows that leave both states unchanged: zero x/B/C, and a dt softplus maps to 0.

        Gathered in-graph next to the step's rows, so the kernels compile once per
        bucket and never copy a row range of ``projected_states``, whose width need not
        be stick-aligned.
        """
        if self._spyre_staging is None or self._spyre_staging["pad"].shape[1] != width:
            pad = torch.zeros(self.chunk_size, width, dtype=dtype)
            pad[:, self.tped_intermediate_size + self.tped_conv_size :] = _PAD_DT
            self._spyre_staging = {"pad": convert(pad, device=device)}
        return self._spyre_staging["pad"]

    def _out_staging(self, rows: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``[rows + 1, I]``: the kernels write their rows here in-graph, and padded rows to
        the extra last one. Copied whole into the op's output once per step, so serving
        does no eager row-range copy, whose shape would change with every request."""
        staging = self._spyre_out
        if staging is None or staging.shape[0] != rows + 1:
            staging = torch.zeros(rows + 1, self.tped_intermediate_size, dtype=dtype, device=device)
            self._spyre_out = staging
        return staging

    def _flush_output(self, output: torch.Tensor) -> None:
        if self._spyre_out is not None and self._spyre_out.shape[0] == output.shape[0] + 1:
            output.copy_(self._spyre_out[: output.shape[0]])

    def _dims(self) -> tuple[int, int, int]:
        return (
            self.tped_intermediate_size,
            self.groups_ssm_state_size // self.tp_size,
            self.head_dim,
        )

    def conv_ssm_forward(self, projected_states: torch.Tensor, output: torch.Tensor):
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.prefix)
        if attn_metadata is None:
            # Profile and dummy runs: no state to read or write.
            return
        assert isinstance(attn_metadata, Mamba2AttentionMetadata)
        assert not is_conv_state_dim_first(), "Spyre MambaMixer2 needs the SD conv state layout."
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        if num_decodes > 0:
            assert attn_metadata.state_indices_tensor_d is not None
            slots = attn_metadata.state_indices_tensor_d[:num_decodes, 0]
            self._decode(projected_states, num_decode_tokens, slots, output)
        if attn_metadata.num_prefills > 0:
            assert attn_metadata.query_start_loc_p is not None
            assert attn_metadata.has_initial_states_p is not None
            assert attn_metadata.state_indices_tensor_p is not None
            qsl = attn_metadata.query_start_loc_p.cpu().tolist()
            has_init = attn_metadata.has_initial_states_p.cpu().tolist()
            slots = attn_metadata.state_indices_tensor_p.cpu().tolist()
            for i, slot in enumerate(slots):
                start, end = num_decode_tokens + qsl[i], num_decode_tokens + qsl[i + 1]
                self._prefill(projected_states, start, end, int(slot), bool(has_init[i]), output)
        self._flush_output(output)

    def _decode(
        self, projected_states: torch.Tensor, bs: int, slots: torch.Tensor, output: torch.Tensor
    ) -> None:
        """The first ``bs`` rows of ``projected_states`` are one decode token each."""
        padded = next(b for b in self.decode_buckets if b >= bs)
        device, dtype = projected_states.device, projected_states.dtype
        consts = self._consts(device, dtype)
        pad = self._pad_rows(projected_states.shape[1], device, dtype)
        # Padded rows read the first pad row, write the scratch output row, and update
        # the null block, which no request reads.
        row_ids = convert(
            torch.cat([torch.arange(bs), torch.full((padded - bs,), projected_states.shape[0])]),
            device=device,
        )
        host = torch.full((padded,), NULL_BLOCK_ID, dtype=torch.int64)
        host[:bs] = slots.to(torch.int64)
        intermediate, group_state, head_dim = self._dims()
        _decode_compiled(
            projected_states,
            pad,
            row_ids,
            row_ids,
            self._out_staging(projected_states.shape[0], device, dtype),
            self.kv_cache[0],
            self.kv_cache[1],
            convert(host, device=device),
            consts["conv_w"],
            consts["conv_b"],
            consts["dt_bias_d"],
            consts["A_d"],
            consts["D_d"],
            consts["E"],
            intermediate,
            group_state,
            head_dim,
        )

    def _prefill(
        self,
        projected_states: torch.Tensor,
        start: int,
        end: int,
        slot: int,
        has_init: bool,
        output: torch.Tensor,
    ) -> None:
        device, dtype = projected_states.device, projected_states.dtype
        consts = self._consts(device, dtype)
        pad = self._pad_rows(projected_states.shape[1], device, dtype)
        total = projected_states.shape[0]
        chunk, taps = self.chunk_size, self.conv_kernel_size - 1
        intermediate, group_state, head_dim = self._dims()
        slot_dev = convert(torch.tensor([slot], dtype=torch.int64), device=device)
        for offset in range(start, end, chunk):
            n = min(chunk, end - offset)
            real = torch.arange(offset, offset + n)
            row_ids = torch.cat([real, total + torch.arange(n, chunk)])
            out_ids = torch.cat([real, torch.full((chunk - n,), total)])
            init = 1.0 if has_init or offset > start else 0.0
            _prefill_compiled(
                projected_states,
                pad,
                convert(row_ids, device=device),
                convert(out_ids, device=device),
                self._out_staging(total, device, dtype),
                self.kv_cache[0],
                self.kv_cache[1],
                slot_dev,
                convert(torch.tensor([init], dtype=dtype), device=device),
                convert(torch.arange(n, n + taps, dtype=torch.int64), device=device),
                consts["conv_w"],
                consts["conv_b"],
                consts["dt_bias"],
                consts["A"],
                consts["D_d"],
                consts["E"],
                consts["tril"],
                consts["stril"],
                consts["striu"],
                intermediate,
                group_state,
                head_dim,
            )

    def warm_kernels(self, body_buckets: list[int]) -> None:
        """Compile the kernels at every shape serving can present, on the null block.

        A kernel's input is the whole padded ``projected_states``, so each body bucket
        is its own shape: decode at every decode bucket that fits in it, prefill once.
        Writing only the null block leaves every request's state untouched.
        """
        device = self.kv_cache[0].device
        dtype = self.kv_cache[0].dtype
        null = torch.full((self.decode_buckets[-1],), NULL_BLOCK_ID, dtype=torch.int64)
        hidden = self.in_proj.input_size
        for tokens in body_buckets:
            # Through in_proj, like serving: the kernels guard on the device layout of
            # ``projected_states``, which a host-built tensor of the same shape lacks.
            x = convert(torch.zeros(tokens, hidden, dtype=dtype), device=device)
            rows, _ = self.in_proj(x)
            out = torch.empty(tokens, self.tped_intermediate_size, dtype=dtype, device=device)
            for bs in self.decode_buckets:
                if bs <= tokens:
                    self._decode(rows, bs, null[:bs], out)
            self._prefill(rows, 0, min(tokens, self.chunk_size), NULL_BLOCK_ID, False, out)
            self._flush_output(out)
