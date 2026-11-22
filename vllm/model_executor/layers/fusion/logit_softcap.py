# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-pass final-logit soft cap, ``tanh(x / cap) * cap``, with the output in fp32 or the input dtype.

The eager chain (``x / cap``, ``tanh``, ``* cap``) is three launches over the whole [tokens, vocab]
logits, each rounding to the model dtype, and the bounded top-k sampler then copies the result to fp32.
This kernel keeps the three roundings (so every output element equals the eager chain's) and writes the
fp32 selector input directly.

The tanh is ``(1 - e) / (1 + e)`` with ``e = exp2(-2 |t| log2 e)``, a hardware reciprocal, and ``t``
itself below 0.03. It is not correctly rounded in fp32, but ``t`` is a 16-bit value and the result is
rounded to 16 bits, so the whole function has only 65536 inputs: `_verified` compares the kernel with the
eager chain on all of them once per (cap, dtype, device) and the fused path is used only if every output
is bit-identical.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_LOG2E_X2 = tl.constexpr(2.8853900817779268)
_SMALL_T = 0.03
_verdicts: dict[tuple[float, torch.dtype, str], bool] = {}


@triton.jit
def _rcp(v):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [v], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def softcap_f32(x, inv_cap, cap, IN_DTYPE: tl.constexpr, SMALL_T: tl.constexpr):
    """The eager chain's three roundings on an fp32 view of 16-bit values; fp32 result."""
    t = (x * inv_cap).to(IN_DTYPE).to(tl.float32)
    a = tl.abs(t)
    e = tl.exp2(a * -_LOG2E_X2)
    r = (1.0 - e) * _rcp(1.0 + e)
    r = tl.where(a < SMALL_T, a, r)
    r = tl.where(a > 10.0, 1.0, r)
    r = (
        r.to(tl.int32, bitcast=True)
        | (t.to(tl.int32, bitcast=True) & -2147483648)
    ).to(tl.float32, bitcast=True)
    u = r.to(IN_DTYPE).to(tl.float32)
    return (u * cap).to(IN_DTYPE).to(tl.float32)


@triton.jit
def _softcap_kernel(
    x_ptr,
    y_ptr,
    n,
    inv_cap,
    cap,
    IN_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
    SMALL_T: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = softcap_f32(x, inv_cap, cap, IN_DTYPE, SMALL_T)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


_TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def _launch(x: torch.Tensor, y: torch.Tensor, cap: float) -> None:
    n = x.numel()
    block = 2048 if n > (1 << 21) else 1024
    _softcap_kernel[(triton.cdiv(n, block),)](
        x,
        y,
        n,
        1.0 / cap,
        cap,
        _TL_DTYPES[x.dtype],
        block,
        _SMALL_T,
        num_warps=2,
    )


def _verified(cap: float, dtype: torch.dtype, device: torch.device) -> bool:
    key = (cap, dtype, str(device))
    ok = _verdicts.get(key)
    if ok is None:
        bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
        x = bits.view(dtype)
        ref = torch.tanh(x / cap) * cap
        out = torch.empty_like(x)
        _launch(x, out, cap)
        same = (out.view(torch.int16) == ref.view(torch.int16)) | (
            torch.isnan(out) & torch.isnan(ref)
        )
        ok = bool(same.all().item())
        _verdicts[key] = ok
        if not ok:
            logger.warning(
                "Fused logit soft cap differs from the eager chain for cap=%s %s; "
                "using the eager chain.",
                cap,
                dtype,
            )
    return ok


def softcap_exact(cap: float, dtype: torch.dtype, device: torch.device) -> bool:
    """Whether `softcap_f32` reproduces the eager chain bit for bit for this cap and dtype."""
    return dtype in _TL_DTYPES and _verified(cap, dtype, device)


def eager_soft_cap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    capped = fused_logit_softcap(logits, cap)
    return capped if capped is not None else torch.tanh(logits / cap) * cap


def fused_logit_softcap(
    logits: torch.Tensor, cap: float, out_dtype: torch.dtype | None = None
) -> torch.Tensor | None:
    """``tanh(logits / cap) * cap`` in one pass, or None when the eager chain must be used."""
    if (
        logits.dtype not in _TL_DTYPES
        or not logits.is_cuda
        or not logits.is_contiguous()
        or logits.numel() == 0
        or not _verified(cap, logits.dtype, logits.device)
    ):
        return None
    out = torch.empty(logits.shape, dtype=out_dtype or logits.dtype, device=logits.device)
    _launch(logits, out, cap)
    return out
