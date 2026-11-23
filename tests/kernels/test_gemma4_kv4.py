# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.layers.fusion.gemma4_qkv_glue import qkv_glue
from vllm.v1.attention.ops.rocm_kv4_attention import (
    kv4_decode_attention,
    kv4_dequant_to_fp8,
)

HD, BS = 512, 64
EPS = 1e-6
NQ, NKV = 8, 2


def _rms(x, w):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if w is not None:
        out = out * w.float()
    return out.to(torch.bfloat16)


def _rope(x, cos_sin, pos, rot_pairs):
    cos, sin = cos_sin.index_select(0, pos).float().chunk(2, dim=-1)
    half = x.shape[-1] // 2
    c = torch.ones(cos.shape[0], half, device=x.device)
    s = torch.zeros_like(c)
    c[:, :rot_pairs], s[:, :rot_pairs] = cos[:, :rot_pairs], sin[:, :rot_pairs]
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = c.unsqueeze(1), s.unsqueeze(1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1).to(torch.bfloat16)


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _deq(nib):
    """int tensor of e2m1 codes -> float values."""
    mag = torch.tensor(E2M1, device=nib.device)[(nib & 7).long()]
    return mag * (1 - 2 * ((nib >> 3) & 1).float())


def _unpack(packed):
    """[..., HD/2] uint8 -> [..., HD] float (low nibble = channel j, high = j + HD/2)."""
    p = packed.int()
    return torch.cat((_deq(p & 15), _deq(p >> 4)), -1)


def _quant(x, scale):
    y = x.float() / scale
    a = y.abs()
    thr = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = (a[..., None] > thr).sum(-1).int() | ((y < 0).int() << 3)
    return code


@pytest.mark.parametrize("num_tokens", [1, 37, 300])
def test_kv4_writer(num_tokens: int):
    torch.manual_seed(0)
    dev = "cuda"
    rot_pairs = 64
    qkv = (torch.randn(num_tokens, (NQ + 2 * NKV) * HD, device=dev) * 3).to(torch.bfloat16)
    pos = torch.randint(0, 4096, (num_tokens,), device=dev)
    cos_sin = torch.randn(4096, HD, device=dev).to(torch.bfloat16)
    q_w = (torch.randn(HD, device=dev) * 0.05 + 0.06).to(torch.bfloat16)
    k_w = (torch.randn(HD, device=dev) * 0.05 + 0.06).to(torch.bfloat16)
    ks, vs = 0.035, 0.5
    scales = [torch.tensor(s, device=dev) for s in (1.0, ks, vs)]
    nblocks = num_tokens // BS + 3
    kv = torch.zeros(nblocks, NKV, BS, HD, dtype=torch.uint8, device=dev)  # K | V rows of HD/2 bytes
    kc, vc = kv.transpose(1, 2).split(HD // 2, dim=-1)
    slots = torch.randperm(nblocks * BS, device=dev)[:num_tokens]
    if num_tokens > 1:
        slots[::5] = -1
    qkv_glue(qkv, pos, cos_sin, q_w, k_w, *scales, kc, vc, slots, NQ, NKV, HD, rot_pairs, EPS, kv4=True)
    k = qkv[:, NQ * HD : (NQ + NKV) * HD].view(-1, NKV, HD)
    v = qkv[:, (NQ + NKV) * HD :].view(-1, NKV, HD)
    k_ref = _rope(_rms(k, k_w), cos_sin, pos, rot_pairs)
    v_ref = _rms(v, None)
    ok = slots >= 0
    blk, off = slots[ok] // BS, slots[ok] % BS
    for cache, ref, s in ((kc, k_ref, ks), (vc, v_ref, vs)):
        code = _quant(ref[ok], s)
        want = (code[..., : HD // 2] | (code[..., HD // 2 :] << 4)).to(torch.uint8)
        got = cache[blk, off]
        assert (got != want).float().mean() < 2e-3
        # nothing else written
    untouched = torch.ones(nblocks * BS, dtype=torch.bool, device=dev)
    untouched[slots[ok]] = False
    idx = torch.arange(nblocks * BS, device=dev)[untouched]
    assert not kc[idx // BS, idx % BS].any() and not vc[idx // BS, idx % BS].any()


def _make_cache(nblocks, seqlens, device, BS=64):
    nb = (seqlens + BS - 1) // BS
    perm = torch.randperm(nblocks, device=device)
    bt = torch.zeros(len(seqlens), int(nb.max()), dtype=torch.int32, device=device)
    p = 0
    for i in range(len(seqlens)):
        bt[i, : nb[i]] = perm[p : p + int(nb[i])].int()
        p += int(nb[i])
    cache = torch.randint(0, 256, (nblocks, NKV, BS, HD), dtype=torch.uint8, device=device)
    return cache, bt


@pytest.mark.parametrize("BS", [64, 128])
@pytest.mark.parametrize("lens", [[1, 63, 64, 65, 700], [3000] * 4, [8300, 9100, 777]])
def test_kv4_decode(lens, BS):
    torch.manual_seed(1)
    dev = "cuda"
    fp8 = current_platform.fp8_dtype()
    seqlens = torch.tensor(lens, dtype=torch.int32, device=dev)
    ns = len(lens)
    cache, bt = _make_cache(sum((x + BS - 1) // BS for x in lens) + 5, seqlens, dev, BS)
    kc, vc = cache.transpose(1, 2).split(HD // 2, dim=-1)
    q = (torch.randn(ns, NQ, HD, device=dev) * 0.5).to(fp8)
    out = torch.empty(ns, NQ, HD, dtype=torch.bfloat16, device=dev)
    ks, vs = 0.035, 0.5
    qd = torch.ones(1, device=dev)
    cu = torch.arange(ns + 1, dtype=torch.int32, device=dev)
    kv4_decode_attention(q, kc, vc, out, cu, seqlens, bt, 1.0, qd, ks, vs)
    ref = torch.zeros(ns, NQ, HD, device=dev)
    for s in range(ns):
        n = lens[s]
        blocks = bt[s, : (n + BS - 1) // BS].long()
        rows = cache[blocks].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :n]  # [H, n, K|V bytes]
        kf = _unpack(rows[..., : HD // 2]) * ks
        vf = _unpack(rows[..., HD // 2 :]) * vs
        for h in range(NQ):
            g = h // (NQ // NKV)
            p = torch.softmax(q[s, h].float() @ kf[g].T, -1)
            ref[s, h] = p @ vf[g]
    err = (out.float() - ref).abs().mean() / ref.abs().mean()
    assert err < 0.03, err


@pytest.mark.parametrize("BS", [64, 128])
def test_kv4_dequant_roundtrip(BS):
    torch.manual_seed(2)
    dev = "cuda"
    fp8 = current_platform.fp8_dtype()
    lens = [100, 1, 700]
    seqlens = torch.tensor(lens, dtype=torch.int32, device=dev)
    cache, bt = _make_cache(40, seqlens, dev, BS)
    k, v, bt2 = kv4_dequant_to_fp8(cache, bt, seqlens, max(lens))
    for s, n in enumerate(lens):
        rows = cache[bt[s, : (n + BS - 1) // BS].long()].permute(1, 0, 2, 3).reshape(NKV, -1, HD)[:, :n]
        kref = _unpack(rows[..., : HD // 2])
        vref = _unpack(rows[..., HD // 2 :])
        for b in range((n + 63) // 64):
            sb = int(bt2[s, b])
            t = min(64, n - b * 64)
            assert torch.equal(k[sb].permute(1, 0, 2)[:, :t].float(), kref[:, b * 64 : b * 64 + t])
            assert torch.equal(v[sb].permute(1, 0, 2)[:, :t].float(), vref[:, b * 64 : b * 64 + t])
