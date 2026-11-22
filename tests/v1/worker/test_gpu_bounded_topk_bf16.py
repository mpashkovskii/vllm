# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the 16-bit top-k selector and the bounded sampler's deferred soft cap."""

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.v1.worker.gpu.sample.topk_bf16 import topk_bf16_select
from vllm.v1.worker.gpu.sample.topk_sample import (
    bounded_topk_available,
    bounded_topk_sample,
)

DEVICE = current_platform.device_type
VOCAB = 262144


def _key(x: torch.Tensor) -> torch.Tensor:
    b = x.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(b >= 0x8000, 0xFFFF - b, b | 0x8000).long()


def _reference(x: torch.Tensor, k: int) -> torch.Tensor:
    ids = torch.arange(x.shape[1], device=x.device)[None, :]
    packed = (_key(x) << 32) | (0x7FFFFFFF - ids)
    return packed.topk(k, dim=1).indices.int()


def _cases():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    randn = lambda *s: torch.randn(*s, device=DEVICE, generator=g)  # noqa: E731
    yield "gaussian", (randn(16, VOCAB) * 4).to(torch.bfloat16)
    yield "few_values", torch.randint(-3, 4, (4, VOCAB), device=DEVICE).to(torch.bfloat16)
    yield "all_equal", torch.zeros(3, VOCAB, device=DEVICE, dtype=torch.bfloat16)
    x = torch.zeros(3, VOCAB, device=DEVICE, dtype=torch.bfloat16)
    x[:, ::3] = -0.0
    yield "signed_zeros", x
    x = (randn(4, VOCAB) * 3).to(torch.bfloat16)
    x[:, 1000:1100] = 50
    x[:, 5000:5040] = 50
    yield "tie_at_the_cut", x
    x = (randn(4, VOCAB) * 3).to(torch.bfloat16)
    x[:, 100000:101500] = (randn(4, 1500) + 20).to(torch.bfloat16)
    yield "cluster_over_capacity", x
    x = (randn(4, VOCAB) * 3).to(torch.bfloat16)
    x[:, 7] = float("inf")
    x[:, 9] = -float("inf")
    yield "infinities", x
    yield "fp16", (randn(4, VOCAB) * 3).to(torch.float16)


@pytest.mark.parametrize("k", [1, 8, 64, 128])
def test_selector_is_exact(k):
    for name, x in _cases():
        out = topk_bf16_select(x, k)
        ref = _reference(x, k)
        assert torch.equal(out.sort(dim=1).values, ref.sort(dim=1).values), name
    # Within capacity the result is also ordered best first.
    x = (torch.randn(8, VOCAB, device=DEVICE) * 4).to(torch.bfloat16)
    assert torch.equal(topk_bf16_select(x, k), _reference(x, k))


def test_selector_small_capacity_takes_the_exact_fallback():
    x = (torch.randn(8, VOCAB, device=DEVICE) * 4).to(torch.bfloat16)
    out = topk_bf16_select(x, 64, cap=16)
    assert torch.equal(out.sort(dim=1).values, _reference(x, 64).sort(dim=1).values)


def _sampler_inputs(n: int, top_p: float):
    mapping = torch.arange(n, device=DEVICE, dtype=torch.int32)
    k = torch.full((n,), 64, device=DEVICE, dtype=torch.int32)
    p = torch.full((n,), top_p, device=DEVICE, dtype=torch.float32)
    t = torch.ones(n, device=DEVICE, dtype=torch.float32)
    seeds = torch.randint(-(2**60), 2**60, (n,), device=DEVICE)
    pos = torch.arange(n, device=DEVICE, dtype=torch.int64) + 100
    return mapping, k, p, t, seeds, pos


@pytest.mark.skipif(not bounded_topk_available(), reason="aiter topk_select unavailable")
@pytest.mark.parametrize("top_p", [0.95, 1.0])
def test_deferred_soft_cap_matches_capped_logits(top_p):
    n, cap = 256, 30.0
    g = torch.Generator(device=DEVICE).manual_seed(1)
    raw = (torch.randn(n, VOCAB, device=DEVICE, generator=g) * 6).to(torch.bfloat16)
    capped = torch.tanh(raw / cap) * cap
    mapping, k, p, t, seeds, pos = _sampler_inputs(n, top_p)
    pp = p if top_p < 1.0 else None
    expected = bounded_topk_sample(capped.float(), 64, mapping, k, pp, t, seeds, pos)
    actual = bounded_topk_sample(raw, 64, mapping, k, pp, t, seeds, pos, soft_cap=cap)
    # The two selectors may keep different tokens among those tied at the cut.
    assert (actual == expected).float().mean().item() >= 0.97
    greedy = bounded_topk_sample(
        raw, 64, mapping, k, pp, torch.zeros_like(t), seeds, pos, soft_cap=cap
    )
    assert torch.equal(greedy, capped.float().argmax(dim=-1))


@pytest.mark.skipif(not bounded_topk_available(), reason="aiter topk_select unavailable")
def test_deferred_soft_cap_saturated_logits_stay_in_support():
    n, cap = 128, 30.0
    g = torch.Generator(device=DEVICE).manual_seed(2)
    raw = (torch.randn(n, VOCAB, device=DEVICE, generator=g) * 20).to(torch.bfloat16)
    capped = (torch.tanh(raw / cap) * cap).float()
    mapping, k, p, t, seeds, pos = _sampler_inputs(n, 1.0)
    sampled = bounded_topk_sample(raw, 64, mapping, k, None, t, seeds, pos, soft_cap=cap)
    kth = capped.topk(64, dim=1).values[:, -1:]
    assert torch.all(capped.gather(1, sampled[:, None]) >= kth)
