# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 per-layer-input combine as one row-wise Triton kernel.

  out[t, l, :] = (rmsnorm(proj[t, l*D:(l+1)*D] * s_proj) * w + raw[t, l, :]) * s_in

replaces the scale multiply, the projection RMSNorm and the add-multiply of
``project_per_layer_inputs`` (three passes over [T, L, D]) with one pass that reads
``proj`` and ``raw`` once and writes ``out`` once. The bf16 rounding points of the unfused
chain are kept (scaled projection, norm output, sum, scaled sum).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _ple_combine_kernel(
    proj_ptr,
    raw_ptr,
    w_ptr,
    s_proj_ptr,
    s_in_ptr,
    out_ptr,
    stride_proj,
    stride_raw,
    stride_out,
    eps,
    L: tl.constexpr,
    D: tl.constexpr,
    BL: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    lb = tl.program_id(1)
    ls = lb * BL + tl.arange(0, BL)
    cols = tl.arange(0, D)
    lmask = (ls < L)[:, None]
    offs = ls[:, None] * D + cols[None, :]
    x = tl.load(proj_ptr + t * stride_proj + offs, mask=lmask, other=0.0)
    s_proj = tl.load(s_proj_ptr).to(tl.float32)
    x = (x.to(tl.float32) * s_proj).to(x.dtype)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=1) / D
    xn = xf * tl.rsqrt(var + eps)[:, None]
    w = tl.load(w_ptr + cols)
    y = (xn.to(x.dtype).to(tl.float32) * w.to(tl.float32)[None, :]).to(x.dtype)
    raw = tl.load(raw_ptr + t * stride_raw + offs, mask=lmask, other=0.0)
    z = (y.to(tl.float32) + raw.to(tl.float32)).to(x.dtype)
    s_in = tl.load(s_in_ptr).to(tl.float32)
    o = (z.to(tl.float32) * s_in).to(x.dtype)
    tl.store(out_ptr + t * stride_out + offs, o, mask=lmask)


def _gemma4_ple_combine_impl(
    proj: torch.Tensor,
    raw: torch.Tensor,
    weight: torch.Tensor,
    proj_scale: torch.Tensor,
    in_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    T, L, D = raw.shape
    assert proj.shape == (T, L * D) and proj.stride(1) == 1 and raw.stride(2) == 1
    assert raw.stride(1) == D and D == 256
    out = torch.empty((T, L, D), dtype=proj.dtype, device=proj.device)
    BL = 4
    _ple_combine_kernel[(T, triton.cdiv(L, BL))](
        proj,
        raw,
        weight,
        proj_scale,
        in_scale,
        out,
        proj.stride(0),
        raw.stride(0),
        out.stride(0),
        eps,
        L=L,
        D=D,
        BL=BL,
        num_warps=4,
    )
    return out


def _gemma4_ple_combine_fake(
    proj: torch.Tensor,
    raw: torch.Tensor,
    weight: torch.Tensor,
    proj_scale: torch.Tensor,
    in_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty(raw.shape, dtype=proj.dtype, device=proj.device)


direct_register_custom_op(
    op_name="gemma4_ple_combine",
    op_func=_gemma4_ple_combine_impl,
    mutates_args=[],
    fake_impl=_gemma4_ple_combine_fake,
)


def gemma4_ple_combine(
    proj: torch.Tensor,
    raw: torch.Tensor,
    weight: torch.Tensor,
    proj_scale: torch.Tensor,
    in_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.ops.vllm.gemma4_ple_combine(
        proj, raw, weight, proj_scale, in_scale, eps
    )
