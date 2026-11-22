# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact per-row top-k over 16-bit logits in two reads of the row.

Each 16-bit value maps to an order-preserving 16-bit key. The row is cut into
`NUM_CHUNKS` chunks and the k-th largest chunk maximum, T, bounds the k-th
largest value from below (k distinct elements are >= T). A second pass
collects every element with key >= T (about 1.5-3 k of them on LM-head logits)
into a small per-row buffer and a sort picks the k best: key descending, token
id ascending among equal keys, so the result is a pure function of the row. A
row with more than `cap` candidates (a very flat row) takes an exact bisection
over the key space instead.

Returns the k column indices of each row, best first.
"""

import torch

from vllm.triton_utils import tl, triton

NUM_CHUNKS = 1024
DEFAULT_CAP = 512
MAX_K = 128


@triton.jit
def _key(x):
    b = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return tl.where(b >= 0x8000, 0xFFFF - b, b | 0x8000)


@triton.jit
def _chunk_max_kernel(
    x_ptr,
    stride,
    maxes_ptr,
    cnt_ptr,
    CH,
    NCH: tl.constexpr,
    CHP: tl.constexpr,
    G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    c = g * G + tl.arange(0, G)
    e = tl.arange(0, CHP)
    m = e[None, :] < CH
    offs = c[:, None] * CH + e[None, :]
    x = tl.load(x_ptr + row * stride + offs, mask=m, other=0.0)
    key = tl.where(m, _key(x), -1)
    tl.store(maxes_ptr + row * NCH + c, tl.max(key, axis=1))
    if g == 0:
        tl.store(cnt_ptr + row, 0)


@triton.jit
def _threshold_kernel(maxes_ptr, t_ptr, NCH: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    mx = tl.load(maxes_ptr + row * NCH + tl.arange(0, NCH))
    lo = 0
    hi = 65535
    for _ in range(16):
        mid = (lo + hi + 1) // 2
        c = tl.sum((mx >= mid).to(tl.int32), axis=0)
        if c >= K:
            lo = mid
        else:
            hi = mid - 1
    tl.store(t_ptr + row, lo)


@triton.jit
def _collect_kernel(
    x_ptr,
    stride,
    t_ptr,
    ck_ptr,
    ci_ptr,
    cnt_ptr,
    V,
    BLK: tl.constexpr,
    CAP: tl.constexpr,
    W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    thr = tl.load(t_ptr + row)
    r = tl.arange(0, BLK // W)
    c = tl.arange(0, W)
    offs = tl.program_id(1) * BLK + r[:, None] * W + c[None, :]
    m = offs < V
    x = tl.load(x_ptr + row * stride + offs, mask=m, other=0.0)
    key = _key(x)
    sel = ((key >= thr) & m).to(tl.int32)
    row_cnt = tl.sum(sel, axis=1)
    n = tl.sum(row_cnt, axis=0)
    if n > 0:
        base = tl.atomic_add(cnt_ptr + row, n, sem="relaxed")
        pos = base + (tl.cumsum(row_cnt, axis=0) - row_cnt)[:, None] + tl.cumsum(sel, axis=1) - 1
        st = (sel > 0) & (pos < CAP)
        tl.store(ck_ptr + row * CAP + pos, key, mask=st)
        tl.store(ci_ptr + row * CAP + pos, offs, mask=st)


@triton.jit
def _select_kernel(
    x_ptr,
    stride,
    ck_ptr,
    ci_ptr,
    cnt_ptr,
    out_ptr,
    V,
    K: tl.constexpr,
    CAP: tl.constexpr,
    BLK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(cnt_ptr + row)
    if n <= CAP:
        i = tl.arange(0, CAP)
        valid = i < n
        key = tl.load(ck_ptr + row * CAP + i, mask=valid, other=-1).to(tl.int64)
        idx = tl.load(ci_ptr + row * CAP + i, mask=valid, other=0).to(tl.int64)
        packed = tl.where(valid, (key << 32) | (0x7FFFFFFF - idx), -1)
        srt = tl.sort(packed, 0)
        best_first = CAP - 1 - i
        tl.store(
            out_ptr + row * K + best_first,
            (0x7FFFFFFF - (srt & 0x7FFFFFFF)).to(tl.int32),
            mask=best_first < K,
        )
    else:
        lo = 0
        hi = 65535
        for _ in range(16):
            mid = (lo + hi + 1) // 2
            c = 0
            for s in range(0, V, BLK):
                offs = s + tl.arange(0, BLK)
                m = offs < V
                x = tl.load(x_ptr + row * stride + offs, mask=m, other=0.0)
                c += tl.sum(((_key(x) >= mid) & m).to(tl.int32), axis=0)
            if c >= K:
                lo = mid
            else:
                hi = mid - 1
        need = K
        for s in range(0, V, BLK):
            offs = s + tl.arange(0, BLK)
            m = offs < V
            x = tl.load(x_ptr + row * stride + offs, mask=m, other=0.0)
            need -= tl.sum(((_key(x) > lo) & m).to(tl.int32), axis=0)
        base = 0
        eq_seen = 0
        for s in range(0, V, BLK):
            offs = s + tl.arange(0, BLK)
            m = offs < V
            x = tl.load(x_ptr + row * stride + offs, mask=m, other=0.0)
            key = _key(x)
            eq = ((key == lo) & m).to(tl.int32)
            take_eq = (eq > 0) & (eq_seen + tl.cumsum(eq, axis=0) <= need)
            sel = ((key > lo) & m) | take_eq
            s32 = sel.to(tl.int32)
            pos = base + tl.cumsum(s32, axis=0) - 1
            tl.store(out_ptr + row * K + pos, offs.to(tl.int32), mask=sel & (pos < K))
            base += tl.sum(s32, axis=0)
            eq_seen += tl.sum(eq, axis=0)


_scratch: dict[tuple[torch.device, int], tuple[torch.Tensor, ...]] = {}


def topk_bf16_supported(x: torch.Tensor, k: int) -> bool:
    return (
        x.dtype in (torch.bfloat16, torch.float16)
        and x.is_cuda
        and x.dim() == 2
        and x.stride(-1) == 1
        and x.shape[1] % NUM_CHUNKS == 0
        and 0 < k <= MAX_K
    )


def topk_bf16_select(
    x: torch.Tensor, k: int, cap: int = DEFAULT_CAP
) -> torch.Tensor:
    """Indices (int32, [rows, k], best first) of the k largest entries of each row."""
    rows, vocab = x.shape
    assert topk_bf16_supported(x, k)
    chunk = vocab // NUM_CHUNKS
    sk = (x.device, cap)
    buf = _scratch.get(sk)
    if buf is None or buf[0].shape[0] < rows:
        n = max(rows, 256)
        buf = (
            torch.empty(n, NUM_CHUNKS, dtype=torch.int32, device=x.device),
            torch.empty(n, cap, dtype=torch.int32, device=x.device),
            torch.empty(n, cap, dtype=torch.int32, device=x.device),
            torch.empty(n, dtype=torch.int32, device=x.device),
            torch.empty(n, dtype=torch.int32, device=x.device),
        )
        _scratch[sk] = buf
    maxes, cand_key, cand_idx, cnt, thr = buf
    out = torch.empty(rows, k, dtype=torch.int32, device=x.device)
    chunks_per_prog = 64
    _chunk_max_kernel[(rows, NUM_CHUNKS // chunks_per_prog)](
        x,
        x.stride(0),
        maxes,
        cnt,
        chunk,
        NUM_CHUNKS,
        triton.next_power_of_2(chunk),
        chunks_per_prog,
        num_warps=8,
    )
    _threshold_kernel[(rows,)](maxes, thr, NUM_CHUNKS, k, num_warps=4)
    blk = 4096
    _collect_kernel[(rows, triton.cdiv(vocab, blk))](
        x, x.stride(0), thr, cand_key, cand_idx, cnt, vocab, blk, cap, 8, num_warps=4
    )
    _select_kernel[(rows,)](
        x, x.stride(0), cand_key, cand_idx, cnt, out, vocab, k, cap, 4096, num_warps=8
    )
    return out
