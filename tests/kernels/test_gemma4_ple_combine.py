# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm import ir
from vllm.model_executor.layers.fusion.gemma4_ple_combine import gemma4_ple_combine

D = 256
EPS = 1e-6


def _reference(proj, raw, weight, s_proj, s_in):
    tokens, layers, _ = raw.shape
    x = (proj * s_proj).reshape(tokens, layers, D)
    x = ir.ops.rms_norm.impls["native"].impl_fn(x, weight, EPS, None)
    return (x + raw) * s_in


@pytest.mark.parametrize("tokens", [1, 17, 300, 4099])
@pytest.mark.parametrize("layers", [18, 24])
def test_ple_combine(tokens, layers):
    torch.manual_seed(0)
    dt = torch.bfloat16
    proj = torch.randn(tokens, layers * D, device="cuda", dtype=dt) * 3
    raw = torch.randn(tokens, layers, D, device="cuda", dtype=dt)
    weight = torch.randn(D, device="cuda", dtype=dt) * 0.5 + 1
    s_proj = torch.tensor(2560**-0.5, device="cuda", dtype=dt)
    s_in = torch.rsqrt(torch.tensor(2.0, device="cuda", dtype=dt))
    out = gemma4_ple_combine(proj, raw, weight, s_proj, s_in, EPS)
    ref = _reference(proj, raw, weight, s_proj, s_in)
    assert out.shape == ref.shape and out.dtype == dt
    # fp32 reduction order differs from the unfused chain: a rare 1-ulp bf16 difference
    torch.testing.assert_close(out, ref, atol=0.07, rtol=0.02)
    assert (out != ref).float().mean().item() < 1e-4
