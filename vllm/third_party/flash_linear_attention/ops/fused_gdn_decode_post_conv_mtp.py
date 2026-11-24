# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton port of the fused GDN MTP decode kernel (``fused_gdn_decode_post_conv_mtp``).

The CUDA kernel in csrc only builds for CUDA 13, so on ROCm every Qwen3.5 GDN layer's
speculative-decode step runs the unfused path: ~8 small copy/cat/fill kernels around the
recurrent kernel plus the gated RMSNorm. This does the post-conv work in two launches.
The update kernel reads q/k/v and a/b in place from the projections and runs the gated
delta-rule update over each request's tokens (same state reads and writes as
``fused_sigmoid_gating_delta_rule_update``: start from the state of the last accepted token,
store the state after every token, slot index <= 0 means no state). Like that kernel it tiles
the value dimension, so a small batch still fills the GPU; every tile writes its outputs and
their sum of squares. The epilogue then applies the gated RMSNorm, the only step that needs
the whole value dimension, and writes the out_proj input.

Deferred state commit (``seq_lens`` given): the first verify token is always accepted,
so only its state is stored in full. For every later token the update kernel writes a
record of its rank-1 update (raw k, the corrected value rows, the decay) into the first
rows of that token's state slot, and the next step, when the draft was accepted, loads
the first token's state and replays the accepted records before its own tokens. That
cuts the state traffic of a K=1 verify from one read and two writes to one read and
one write. Each 32-row chunk of a head holds its own copy of the record, so every
program reads and writes only the rows it owns, whatever the value tiling of either
step. In Mamba ``align`` mode the copies at a block boundary read the slot of an
accepted token as a full state, so near a boundary (a multiple of the block size within
2 x the slot count of the computed length) the kernel stores full states as before.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units


@triton.jit
def _gdn_mtp_update_kernel(
    mixed_qkv,
    b,
    a,
    A_log,
    dt_bias,
    state,
    state_indices,
    cu_seqlens,
    num_accepted_tokens,
    seq_lens,
    out,
    sum_sq,
    stride_qkv_row,
    stride_b_row,
    stride_a_row,
    stride_state_slot,
    stride_indices_row,
    num_indices_per_row,
    mamba_block_size,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    DEFER: tl.constexpr,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV: tl.constexpr = V // BV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    num_tokens = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if num_tokens <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_out = out + (bos * HV + i_hv) * V + o_v
    p_sum_sq = sum_sq + (bos * HV + i_hv) * NV + i_v
    accepted = tl.load(num_accepted_tokens + i_n).to(tl.int64)
    src_col = accepted - 1
    if DEFER:
        # Record layout per 32-row chunk of a head, in the state's bf16 elements:
        # raw k [0, K), then as fp32 the corrected values [K/2, K/2 + CH) and the
        # decay at K/2 + CH. A chunk is CH * K elements, CH * K / 2 fp32 words.
        CH: tl.constexpr = 32
        p_row = state_indices + i_n * stride_indices_row
        computed = tl.load(seq_lens + i_n).to(tl.int64) - num_tokens
        prev_computed = computed - accepted
        window = 2 * num_indices_per_row
        block = tl.maximum(mamba_block_size, 1)
        no_align = mamba_block_size <= 0
        defer_now = no_align | ((computed + window) // block == computed // block)
        pending = (
            (prev_computed >= 0)
            & (accepted >= 2)
            & (accepted <= num_indices_per_row)
            & (no_align | ((prev_computed + window) // block == prev_computed // block))
        )
        src_col = tl.where(pending, 0, src_col)
    p_src = state_indices + i_n * stride_indices_row + src_col
    has_src = (accepted > 0) & (accepted <= num_indices_per_row)
    src = tl.load(p_src, mask=has_src, other=0).to(tl.int64)
    if src <= 0:
        for t in range(num_tokens):
            tl.store(p_out + t * HV * V, tl.zeros([BV], dtype=out.dtype.element_ty))
            tl.store(p_sum_sq + t * HV * NV, 0.0)
        return

    p_state = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(state + src * stride_state_slot + p_state).to(tl.float32)
    if DEFER:
        o_rec = (o_v // CH) * (CH * K // 2) + K // 2 + o_v % CH
        chunk0 = i_v * (BV // CH)
        if pending:
            # Replay the accepted draft tokens of the previous step.
            for j in range(1, accepted):
                rec_slot = tl.load(p_row + j).to(tl.int64)
                if rec_slot > 0:
                    p_rec = state + rec_slot * stride_state_slot + i_hv * V * K
                    p_rec32 = p_rec.to(tl.pointer_type(tl.float32))
                    r_k = tl.load(p_rec + chunk0 * CH * K + o_k).to(tl.float32)
                    r_k = r_k * tl.rsqrt(tl.sum(r_k * r_k) + 1e-6)
                    r_v = tl.load(p_rec32 + o_rec)
                    r_decay = tl.load(p_rec32 + chunk0 * (CH * K // 2) + K // 2 + CH)
                    b_h = b_h * r_decay + r_v[:, None] * r_k[None, :]
    neg_A = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    bias = tl.load(dt_bias + i_hv).to(tl.float32)
    for t in range(num_tokens):
        row = bos + t
        p_qkv = mixed_qkv + row * stride_qkv_row
        b_q = tl.load(p_qkv + i_h * K + o_k).to(tl.float32)
        b_k_raw = tl.load(p_qkv + (H + i_h) * K + o_k)
        b_k = b_k_raw.to(tl.float32)
        b_v = tl.load(p_qkv + 2 * H * K + i_hv * V + o_v).to(tl.float32)
        x = tl.load(a + row * stride_a_row + i_hv).to(tl.float32) + bias
        decay = tl.exp(neg_A * tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x))
        beta = tl.sigmoid(tl.load(b + row * stride_b_row + i_hv).to(tl.float32))

        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= decay
        b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * beta
        b_h += b_v[:, None] * b_k[None, :]
        # Rounded like the unfused path, whose recurrent kernel stores it first.
        b_o = tl.sum(b_h * b_q[None, :], 1).to(out.dtype.element_ty)
        tl.store(p_out + t * HV * V, b_o)
        b_o = b_o.to(tl.float32)
        tl.store(p_sum_sq + t * HV * NV, tl.sum(b_o * b_o))

        dst = tl.load(state_indices + i_n * stride_indices_row + t).to(tl.int64)
        if DEFER:
            if dst > 0:
                if (t > 0) & defer_now:
                    p_rec = state + dst * stride_state_slot + i_hv * V * K
                    p_rec32 = p_rec.to(tl.pointer_type(tl.float32))
                    tl.store(p_rec32 + o_rec, b_v)
                    for j in tl.static_range(BV // CH):
                        p_chunk = p_rec + (chunk0 + j) * CH * K
                        tl.store(p_chunk + o_k, b_k_raw.to(state.dtype.element_ty))
                        p_chunk32 = p_rec32 + (chunk0 + j) * (CH * K // 2)
                        tl.store(p_chunk32 + K // 2 + CH, decay)
                else:
                    p_dst = state + dst * stride_state_slot + p_state
                    tl.store(p_dst, b_h.to(state.dtype.element_ty))
        elif dst > 0:
            p_dst = state + dst * stride_state_slot + p_state
            tl.store(p_dst, b_h.to(state.dtype.element_ty))


@triton.jit
def _gdn_mtp_gated_rmsnorm_kernel(
    out,
    sum_sq,
    z,
    norm_weight,
    cu_seqlens,
    stride_z_row,
    eps,
    HV: tl.constexpr,
    V: tl.constexpr,
    NV: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
):
    i_n, i_hv, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    if t >= tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos:
        return

    row = bos + t
    o_v = tl.arange(0, V)
    p_out = out + (row * HV + i_hv) * V + o_v
    b_o = tl.load(p_out).to(tl.float32)
    p_sum_sq = sum_sq + (row * HV + i_hv) * NV + tl.arange(0, NV)
    rstd = tl.rsqrt(tl.sum(tl.load(p_sum_sq)) / V + eps)
    b_z = tl.load(z + row * stride_z_row + i_hv * V + o_v).to(tl.float32)
    if SIGMOID_GATE:
        gate = tl.sigmoid(b_z)
    else:
        gate = b_z * tl.sigmoid(b_z)
    w = tl.load(norm_weight + o_v).to(tl.float32)
    tl.store(p_out, (b_o * rstd * w * gate).to(out.dtype.element_ty))


def _gdn_mtp_update(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    seq_lens: torch.Tensor | None,
    mamba_block_size: int,
) -> tuple[torch.Tensor, int]:
    """Launch the update kernel; returns the per-tile sums of squares and BV."""
    HV, V = out.shape[-2:]
    K = state.shape[-1]
    H = (mixed_qkv.shape[-1] - HV * V) // (2 * K)
    num_tokens = mixed_qkv.shape[0]
    num_requests = cu_seqlens.numel() - 1
    # Layouts the kernels index with (as the CUDA op checks).
    assert K == triton.next_power_of_2(K) and V == triton.next_power_of_2(V) >= 64
    assert H > 0 and HV % H == 0 and mixed_qkv.shape[-1] == 2 * H * K + HV * V
    assert mixed_qkv.dim() == 2 and mixed_qkv.stride(1) == 1
    assert a.shape == b.shape == (num_tokens, HV) and a.stride(1) == b.stride(1) == 1
    assert A_log.is_contiguous() and A_log.numel() == HV
    assert dt_bias.is_contiguous() and dt_bias.numel() == HV
    assert state.dim() == 4 and state.shape[1:3] == (HV, V)
    assert state.stride(1) == V * K and state.stride(2) == K and state.stride(3) == 1
    assert state_indices.dim() == 2 and state_indices.shape[0] == num_requests
    assert state_indices.stride(1) == 1 and cu_seqlens.is_contiguous()
    assert num_accepted_tokens.is_contiguous()
    assert num_accepted_tokens.numel() == num_requests
    assert out.is_contiguous() and out.shape == (num_tokens, HV, V)
    defer = seq_lens is not None
    if defer:
        # A record (raw k, CH fp32 values, an fp32 decay) fits in a 32-row chunk
        # of a bf16 head; the model's k is bf16, so storing it raw is exact.
        assert state.dtype == torch.bfloat16 and mixed_qkv.dtype == torch.bfloat16
        assert V % 32 == 0 and K % 2 == 0 and K * 2 + 33 * 4 <= 32 * K * 2
        assert seq_lens.is_contiguous() and seq_lens.numel() >= num_requests
    # Tile V only while the GPU needs more programs: 32 rows per program while there are fewer
    # (request, value head) pairs than CUs, 64 up to 4x the CUs, else one tile per value head
    # (tuned on MI300X and MI350X).
    heads, num_cus = num_requests * HV, num_compute_units(out.device.index)
    BV = 32 if heads < num_cus else 64 if heads <= 4 * num_cus else V
    sum_sq = torch.empty(
        (out.shape[0], HV, V // BV), dtype=torch.float32, device=out.device
    )
    _gdn_mtp_update_kernel[(V // BV, num_requests, HV)](
        mixed_qkv,
        b,
        a,
        A_log,
        dt_bias,
        state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        seq_lens if defer else num_accepted_tokens,
        out,
        sum_sq,
        mixed_qkv.stride(0),
        b.stride(0),
        a.stride(0),
        state.stride(0),
        state_indices.stride(0),
        state_indices.shape[1],
        mamba_block_size,
        scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        DEFER=defer,
        num_warps=4,
    )
    return sum_sq, BV


def fused_gdn_mtp_update_triton(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    seq_lens: torch.Tensor | None = None,
    mamba_block_size: int = 0,
) -> torch.Tensor:
    """The recurrent update alone (no gated RMSNorm): the core attention output
    of the speculative rows in ``out`` [tokens, HV, V], as
    ``fused_sigmoid_gating_delta_rule_update`` with ``num_accepted_tokens``
    computes it. ``seq_lens`` (per request, its length after this step) turns on
    the deferred state commit; ``mamba_block_size`` is the ``align`` mode block
    size, 0 otherwise.
    """
    _gdn_mtp_update(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        state,
        out,
        scale,
        seq_lens,
        mamba_block_size,
    )
    return out


def fused_gdn_decode_post_conv_mtp_triton(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    norm_eps: float,
    output_gate_activation: str,
    seq_lens: torch.Tensor | None = None,
    mamba_block_size: int = 0,
) -> torch.Tensor:
    """Same arguments and result as ``ops.fused_gdn_decode_post_conv_mtp``.

    ``out`` and ``output_gate`` are [tokens, HV, V]; ``state`` is [slots, HV, V, K].
    ``seq_lens`` and ``mamba_block_size`` as in ``fused_gdn_mtp_update_triton``.
    """
    HV, V = out.shape[-2:]
    num_requests = cu_seqlens.numel() - 1
    assert output_gate.shape == out.shape
    assert output_gate.stride(2) == 1 and output_gate.stride(1) == V
    assert norm_weight.is_contiguous() and norm_weight.numel() == V
    assert output_gate_activation in ("silu", "sigmoid")
    sum_sq, BV = _gdn_mtp_update(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        state,
        out,
        scale,
        seq_lens,
        mamba_block_size,
    )
    _gdn_mtp_gated_rmsnorm_kernel[(num_requests, HV, state_indices.shape[1])](
        out,
        sum_sq,
        output_gate,
        norm_weight,
        cu_seqlens,
        output_gate.stride(0),
        norm_eps,
        HV=HV,
        V=V,
        NV=V // BV,
        SIGMOID_GATE=output_gate_activation == "sigmoid",
        num_warps=2,
    )
    return out
