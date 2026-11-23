# SPDX-License-Identifier: Apache-2.0
"""Pipelined/chunked TP all-reduce with stream overlap (ROCm).

The per-layer [tokens, hidden] all-reduce above the custom-AR window falls
back to PYNCCL and is strictly serial between its producer (o-proj /
MoE-combine GEMM) and its consumers (fused add+rmsnorm, router GEMM, qkv
GEMM). All of those consumers are token-local, so the reduced buffer can be
consumed chunk-wise along the token dimension.

Behind `VLLM_ROCM_PIPELINED_AR_K` (0=off) this module provides:

* the AR split into K token chunks issued back-to-back on a dedicated comm
  stream (stream order serializes NCCL collectives on one communicator; the
  chunks are issued from a single thread, so there is no NCCL concurrency
  hazard);
* each chunk's consumers wait only that chunk's completion event, so chunk
  k's comm overlaps chunk k-1's consumer compute;
* a join event at the end of each op: the compute stream waits every chunk
  event before the op returns, so tensor blocks handed back to the caching
  allocator can never race outstanding comm-stream work (and the pattern is
  a proper fork/join, which is what CUDA-graph capture requires).

Two composite custom ops wrap the stock kernels unchanged (same kernels,
same per-token numerics; only the AR is chunk-reordered, which changes ring
reduction order within bf16 associativity):

* `vllm.rocm_pipelined_post_attn` (o-proj AR + per-chunk fused add+rmsnorm +
  per-chunk router GEMM through the layer's GateLinear);
* `vllm.rocm_pipelined_post_moe` (MoE-output AR + next-layer input
  fused add+rmsnorm + per-chunk qkv GEMM).

Both fall back to the exact stock op sequence (plain AR and full-width
consumers) below `VLLM_ROCM_PIPELINED_AR_MIN_TOKENS` tokens, so decode-size
steps are numerically identical to stock.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Registry of module callables used inside the composite ops (custom-op
# arguments must be tensors/scalars, so modules are passed by integer key).
# key -> {"gate": GateLinear}
_CONSUMERS: dict[int, dict] = {}

_GROUP_STATE: dict[str, dict] = {}


def register_gate(key: int, gate) -> None:
    """Register a decoder layer's GateLinear for the post-attn op."""
    _CONSUMERS[key] = {"gate": gate}


def enabled() -> int:
    """The chunk count K when pipelined AR is enabled on this platform,
    else 0."""
    if not current_platform.is_rocm():
        return 0
    return max(int(envs.VLLM_ROCM_PIPELINED_AR_K), 0)


_MASTER_K: int | None = None


def master_k() -> int:
    """Run-wide pipelined-AR chunk count (0 disables), resolved once.

    Model init code must use this (not `enabled`) so that the decision to
    defer the all-reduces in o_proj/FusedMoE agrees with the decision to
    pipeline them in the decoder layers.
    """
    global _MASTER_K
    if _MASTER_K is None:
        k = enabled()
        if k >= 2:
            try:
                from vllm.config import get_current_vllm_config
                from vllm.distributed import (
                    get_pp_group,
                    get_tensor_model_parallel_world_size,
                )

                if (
                    get_tensor_model_parallel_world_size() <= 1
                    or get_pp_group().world_size > 1
                ):
                    k = 0
                vllm_config = get_current_vllm_config()
                if getattr(vllm_config, "speculative_config", None) is not None:
                    k = 0
            except Exception:
                k = 0
        _MASTER_K = k
        logger.info_once("pipelined_ar: master_k resolved to %d", k)
    return _MASTER_K


def _coordinator(group_name: str):
    from vllm.distributed.parallel_state import _groups

    ref = _groups.get(group_name)
    group = ref() if ref is not None else None
    if group is None:
        raise ValueError(f"Group {group_name} is not found.")
    return group


def _comm_stream(group_name: str) -> torch.cuda.Stream:
    st = _GROUP_STATE.get(group_name)
    if st is None:
        st = {}
        _GROUP_STATE[group_name] = st
    if st.get("comm_stream") is None:
        st["comm_stream"] = torch.cuda.Stream()
        logger.info_once(
            "pipelined_ar: created dedicated comm stream for group %s",
            group_name,
        )
    return st["comm_stream"]


def _chunk_bounds(num_tokens: int, k: int):
    base, rem = divmod(num_tokens, k)
    bounds = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _chunked_ar_start(inp: torch.Tensor, out: torch.Tensor, group_name: str,
                      bounds: list) -> tuple[list, torch.cuda.Event]:
    """Issue the chunk all-reduces for `inp` into `out` on the comm stream.

    Returns (per-chunk completion events, ready event). The caller waits each
    completion event before consuming that chunk on the compute stream, and
    waits the final event (or the whole comm stream) before the op returns,
    which also covers reuse of `inp`. Chunks mirror the production dispatch:
    AITER custom AR for in-window chunks, PYNCCL otherwise.
    """
    group = _coordinator(group_name)
    dc = group.device_communicator
    comm = _comm_stream(group_name)
    current = torch.cuda.current_stream()

    ready = torch.cuda.Event()
    ready.record(current)
    comm.wait_event(ready)

    pynccl_comm = getattr(dc, "pynccl_comm", None) if dc is not None else None
    if pynccl_comm is None or pynccl_comm.disabled:
        raise RuntimeError(
            "pipelined_ar: pynccl communicator unavailable; cannot run the "
            "chunked all-reduce. Set VLLM_ROCM_PIPELINED_AR_K=0."
        )
    aiter_comm = None
    if (
        envs.VLLM_ROCM_PIPELINED_AR_CUSTOM
        and dc is not None
        and getattr(dc, "use_aiter_allreduce", False)
        and getattr(dc, "aiter_ar_comm", None) is not None
        and not dc.aiter_ar_comm.disabled
    ):
        aiter_comm = dc.aiter_ar_comm

    events: list[torch.cuda.Event] = []
    for start, end in bounds:
        x = inp[start:end]
        y = out[start:end]
        if aiter_comm is not None and aiter_comm.should_custom_ar(x):
            with torch.cuda.stream(comm):
                z = aiter_comm.custom_all_reduce(x)
                y.copy_(z)
        else:
            pynccl_comm.all_reduce(x, y, stream=comm)
        ev = torch.cuda.Event()
        ev.record(comm)
        events.append(ev)
    return events, ready


def _plain_ar(inp: torch.Tensor, out: torch.Tensor, group_name: str):
    """The stock out-of-place all-reduce on the current stream."""
    group = _coordinator(group_name)
    out.copy_(group.all_reduce(inp))


def _use_pipelined(num_tokens: int, k: int) -> bool:
    return (
        _runtime_k(k) >= 2
        and current_platform.is_rocm()
        and num_tokens >= envs.VLLM_ROCM_PIPELINED_AR_MIN_TOKENS
    )


_CTRL_CACHE: dict = {"mtime": -1.0, "path": None, "on": True}


def _runtime_k(k: int) -> int:
    """Runtime on/off override, for same-server A/B/A measurements.

    `VLLM_ROCM_PIPELINED_AR_CTRL` (default /tmp/vllm_pipelined_ar.enabled)
    may name a control file containing "1"/"0". Missing file = on. The file
    mtime is checked at most once per second; a removed/missing file means
    enabled. (Under CUDA-graph replay the op body does not re-run, so this
    only controls eager execution paths.)
    """
    if k < 2:
        return 0
    import os
    import time

    path = os.environ.get(
        "VLLM_ROCM_PIPELINED_AR_CTRL", "/tmp/vllm_pipelined_ar.enabled"
    )
    cache = _CTRL_CACHE
    now = time.time()
    if path != cache["path"] or now - cache["mtime"] > 1.0:
        try:
            with open(path) as f:
                cache["on"] = f.read().strip() not in ("0", "false", "off", "")
        except OSError:
            cache["on"] = True
        cache["mtime"] = now
        cache["path"] = path
    return k if cache["on"] else 0


# ---------------------------------------------------------------------------
# post-attention: chunked o-proj AR + per-chunk post_attention_layernorm
# (fused add+rmsnorm) + per-chunk router GEMM.
#   x:          [T, H]  o-proj partial output (unreduced)
#   residual:   [T, H]  residual *before* this layer's attention add
#   rms_weight: [H]     post_attention_layernorm weight
# returns:
#   moe_input:      [T, H]        rmsnorm(residual + AR(x))
#   router_logits:  [T, E]        gate(moe_input) (gate's own out dtype)
#   new_residual:   [T, H]        residual + AR(x)
# ---------------------------------------------------------------------------


def _post_attn_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    group_name: str,
    k: int,
    gate_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import vllm._custom_ops as ops

    assert residual.shape == x.shape, (
        f"rocm_pipelined_post_attn: residual {residual.shape} must match "
        f"x {x.shape}"
    )
    gate = _CONSUMERS[gate_key]["gate"]
    T = x.shape[0]
    normed = torch.empty_like(x)

    if not _use_pipelined(T, k):
        _plain_ar(x, normed, group_name)
        ops.fused_add_rms_norm(normed, residual, rms_weight, eps)
        router_logits, _ = gate(normed)
        return normed, router_logits, residual

    bounds = _chunk_bounds(T, k)
    events, _ = _chunked_ar_start(x, normed, group_name, bounds)
    current = torch.cuda.current_stream()
    chunks = []
    for i, (start, end) in enumerate(bounds):
        current.wait_event(events[i])
        ops.fused_add_rms_norm(
            normed[start:end], residual[start:end], rms_weight, eps
        )
        logits_chunk, _ = gate(normed[start:end])
        chunks.append(logits_chunk)
    router_logits = torch.cat(chunks, dim=0)
    # join (see module docstring): every chunk event is awaited before return
    current.wait_event(events[-1])
    return normed, router_logits, residual


def _post_attn_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    group_name: str,
    k: int,
    gate_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = _CONSUMERS[gate_key]["gate"]
    normed = torch.empty_like(x)
    router_logits = torch.empty(
        x.shape[0], gate.output_size, dtype=gate.out_dtype, device=x.device
    )
    return normed, router_logits, residual


# ---------------------------------------------------------------------------
# post-MoE: chunked MoE-output AR + next-layer input fused add+rmsnorm +
# per-chunk qkv GEMM.
#   x:          [T, H]              MoE partial output (unreduced)
#   residual:   [T, H]              residual *before* the MoE add
#   rms_weight: [H]                 next layer's input_layernorm weight
#   qkv_weight: [(q+2*kv)*hd, H]    next layer's qkv_proj weight (unquantized)
# returns:
#   qkv:           [T, (q+2*kv)*hd]
#   new_residual:  [T, H]           residual + AR(x)
# ---------------------------------------------------------------------------


def _post_moe_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    qkv_weight: torch.Tensor,
    group_name: str,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import vllm._custom_ops as ops

    assert residual.shape == x.shape, (
        f"rocm_pipelined_post_moe: residual {residual.shape} must match "
        f"x {x.shape}"
    )
    T = x.shape[0]
    normed = torch.empty_like(x)

    if not _use_pipelined(T, k):
        _plain_ar(x, normed, group_name)
        ops.fused_add_rms_norm(normed, residual, rms_weight, eps)
        qkv = torch.matmul(normed, qkv_weight.t())
        return qkv, residual

    bounds = _chunk_bounds(T, k)
    events, _ = _chunked_ar_start(x, normed, group_name, bounds)
    current = torch.cuda.current_stream()
    qkv = torch.empty(T, qkv_weight.shape[0], dtype=x.dtype, device=x.device)
    for i, (start, end) in enumerate(bounds):
        current.wait_event(events[i])
        ops.fused_add_rms_norm(
            normed[start:end], residual[start:end], rms_weight, eps
        )
        torch.matmul(normed[start:end], qkv_weight.t(), out=qkv[start:end])
    current.wait_event(events[-1])
    return qkv, residual


def _post_moe_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    qkv_weight: torch.Tensor,
    group_name: str,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qkv = torch.empty(
        x.shape[0], qkv_weight.shape[0], dtype=x.dtype, device=x.device
    )
    return qkv, residual


def _register():
    if not (current_platform.is_rocm() or current_platform.is_cuda()):
        return
    direct_register_custom_op(
        op_name="rocm_pipelined_post_attn",
        op_func=_post_attn_impl,
        mutates_args=["residual"],
        fake_impl=_post_attn_fake,
    )
    direct_register_custom_op(
        op_name="rocm_pipelined_post_moe",
        op_func=_post_moe_impl,
        mutates_args=["residual"],
        fake_impl=_post_moe_fake,
    )


_register()
