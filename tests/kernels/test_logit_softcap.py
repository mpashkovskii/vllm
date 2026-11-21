# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.layers.fusion.logit_softcap import (
    _verified,
    fused_logit_softcap,
)


def _eager(x, cap):
    return torch.tanh(x / cap) * cap


@pytest.mark.parametrize("cap", [30.0, 50.0, 15.0])
def test_all_bf16_inputs(cap):
    assert _verified(cap, torch.bfloat16, torch.device("cuda"))


@pytest.mark.parametrize("shape", [(1, 262144), (7, 1000), (256, 262144)])
@pytest.mark.parametrize("out_dtype", [None, torch.float32])
def test_matches_eager(shape, out_dtype):
    torch.manual_seed(0)
    x = (torch.randn(*shape, device="cuda") * 25).to(torch.bfloat16)
    out = fused_logit_softcap(x, 30.0, out_dtype)
    ref = _eager(x, 30.0)
    assert out.dtype == (out_dtype or torch.bfloat16)
    assert torch.equal(out, ref.to(out.dtype))


def test_falls_back_when_not_applicable():
    x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    assert fused_logit_softcap(x, 30.0) is None
    y = torch.randn(4, 64, device="cuda").to(torch.bfloat16).t()
    assert fused_logit_softcap(y, 30.0) is None
