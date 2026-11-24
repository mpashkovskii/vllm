# SPDX-License-Identifier: Apache-2.0
"""Cascade (shared-prefix) decode attention glue for ROCm backends.

Sibling requests in agentic traffic share long block-aligned prefixes.
Prefix caching stores that KV once; a plain decode kernel still streams
it once per request, and decode attention is bound by those streams.
Cascade attention streams each shared block once per group and merges
in log-sum-exp space -- the same identity split-KV decode relies on.
The kernels and the planner live in AITER
(`aiter.ops.flydsl.pa_decode_cascade{,_plan}`); this module owns the
per-batch-bucket workspaces, the plain-4D layout mapping from vLLM
paged caches, and the decode-row routing for mixed steps.

Enabled with ``VLLM_ROCM_CASCADE_DECODE=1`` (pure-decode steps) and
``VLLM_ROCM_CASCADE_MIXED=1`` (decode rows of mixed prefill/decode
steps). Both default to off while the path is validated. Nothing runs
-- at all, including buffer allocation -- with the flags off.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_PLANNER_TILE_LIMIT = 4096


def _cascade_aiter():
    try:
        from aiter.ops.flydsl.pa_decode_cascade import (  # noqa: F401
            pa_decode_cascade,
            plan_pa_decode_cascade,
        )
        from aiter.ops.flydsl.pa_decode_cascade_plan import (  # noqa: F401
            fill_pa_decode_cascade_plan,
            plan_pa_decode_cascade_groups,
        )
    except ImportError:
        return None
    return (
        pa_decode_cascade,
        plan_pa_decode_cascade,
        fill_pa_decode_cascade_plan,
        plan_pa_decode_cascade_groups,
    )


class CascadeState:
    """Per (geometry, batch bucket) cascade workspace and planner."""

    def __init__(
        self,
        device,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        table_width: int,
        max_reqs: int,
    ):
        fns = _cascade_aiter()
        assert fns is not None
        (
            self.pa_decode_cascade,
            plan,
            self.fill,
            self.plan_planner,
        ) = fns
        self.device = device
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        # One item table row at most per pair of requests; singles cover all.
        max_items = max(1, max_reqs // 2)
        self.ws = plan(
            device,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_items=max_items,
            max_singles=max_reqs,
            table_width=table_width,
            kv_layout="plain4d",
        )
        self.pl = self.plan_planner(
            device,
            max_rows=max_reqs,
            table_width=table_width,
            max_b_rows=self.ws.max_b_rows,
        )
        self.max_reqs = max_reqs
        self.table_width = table_width

    def run(self, output, query, k_cache, v_cache, seq_lens, block_table, scale, sinks):
        n = seq_lens.shape[0]
        self.fill(self.ws, self.pl, block_table, seq_lens)
        self.pa_decode_cascade(
            output,
            query,
            k_cache,
            v_cache,
            seq_lens,
            block_table,
            scale,
            sinks=sinks,
            workspace=self.ws,
        )


_STATES: dict[tuple, CascadeState] = {}
_GATE_REASON: str | None = None


def _log_once(level: int, msg: str, *args) -> None:
    if not getattr(_log_once, f"_{msg}", False):
        setattr(_log_once, f"_{msg}", True)
        logger.info_once(msg, *args, scope="local")


def eligible(
    impl,
    attn_metadata,
    *,
    mixed: bool,
) -> bool:
    """Whether a cascade call may handle this layer and step."""
    global _GATE_REASON
    if not (envs.VLLM_ROCM_CASCADE_DECODE or envs.VLLM_ROCM_CASCADE_MIXED):
        return False
    if _cascade_aiter() is None:
        _GATE_REASON = "aiter cascade ops are not importable"
        return False
    if impl.sliding_window not in (None, -1, (-1, -1)):
        _GATE_REASON = "cascade is off for sliding-window layers"
        return False
    if impl.alibi_slopes is not None:
        _GATE_REASON = "cascade is off for ALiBi"
        return False
    if impl.logits_soft_cap:
        _GATE_REASON = "cascade is off for logits soft cap"
        return False
    if impl.kv_cache_dtype not in (
        "auto",
        "bfloat16",
        "fp16",
        torch.bfloat16,
        torch.float16,
    ):
        _GATE_REASON = f"cascade is off for kv_cache_dtype={impl.kv_cache_dtype}"
        return False
    h = impl.head_size
    if h not in (64, 128, 256, 512):
        _GATE_REASON = f"cascade is off for head_size={h}"
        return False
    if impl.num_kv_heads <= 0 or impl.num_heads % impl.num_kv_heads:
        _GATE_REASON = "cascade needs a GQA group of 1..8 query heads"
        return False
    if impl.num_heads // impl.num_kv_heads > 8:
        _GATE_REASON = "cascade needs a GQA group of 1..8 query heads"
        return False
    if mixed and not envs.VLLM_ROCM_CASCADE_MIXED:
        return False
    if not mixed and not envs.VLLM_ROCM_CASCADE_DECODE:
        return False
    return True


def describe_blocked_call() -> str:
    return _GATE_REASON or "not attempted"


def _state_for(state_key, device, heads, kv_heads, dim, width, max_reqs) -> CascadeState:
    key = (state_key, heads, kv_heads, dim, width, max_reqs)
    st = _STATES.get(key)
    if st is None:
        st = CascadeState(device, heads, kv_heads, dim, width, max_reqs)
        _STATES[key] = st
        logger.info_once(
            "ROCm cascade decode workspace allocated: %s", key, scope="local"
        )
    return st


def decode_rows_maps(query_start_loc: torch.Tensor, seq_lens: torch.Tensor):
    """Decode-row routing for a mixed step (torch ops, eager-only).

    Returns (dec_req, dec_q_map, chk_req, chk_tok, chk_cu_q, chk_len_k):
    request rows with query length 1, their token rows, and the same
    infrastructure for the chunk rows that the unified kernel keeps.
    """
    qsl = query_start_loc
    qlens = qsl[1:] - qsl[:-1]
    is_dec = qlens.eq(1)
    is_chk = ~is_dec
    dec_req = torch.nonzero(is_dec, as_tuple=False).squeeze(1).to(torch.int32)
    chk_req = torch.nonzero(is_chk, as_tuple=False).squeeze(1)
    dec_q_map = qsl[:-1][dec_req.long()].to(torch.int32)
    if chk_req.numel() == 0:
        empty = torch.empty(0, dtype=torch.int32, device=qsl.device)
        return dec_req, dec_q_map, empty, empty, empty.to(qsl.dtype), empty
    chk_qlens = qlens[chk_req]
    chk_cu = torch.cat(
        [
            torch.zeros(1, dtype=qlens.dtype, device=qsl.device),
            torch.cumsum(chk_qlens, 0),
        ]
    ).to(qsl.dtype)
    total_tokens = qsl[-1] - qsl[0]
    tok_of_req = (
        torch.searchsorted(qsl, torch.arange(total_tokens, device=qsl.device), right=True)
        - 1
    )
    chk_tok = torch.nonzero(~is_dec[tok_of_req], as_tuple=False).squeeze(1).to(torch.int32)
    chk_len_k = seq_lens[chk_req.long()]
    return dec_req, dec_q_map, chk_req, chk_tok, chk_cu, chk_len_k
