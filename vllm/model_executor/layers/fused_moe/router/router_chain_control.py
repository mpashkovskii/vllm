# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime on/off control file for the gfx942 router-chain fast paths.

The bf16x3 router GEMM (``bf16x3_router_gemm_rocm`` dispatch) and the fused
routed-scale combine in ``MoERunner.forward_common`` consult this switch so a
same-server A/B/A measurement can flip the whole chain without a server
restart. It is called only from inside custom op bodies, which run eagerly on
every eager step; a plain Python branch in compiled model code would be
frozen at first trace because vLLM drops all Dynamo guards. (For mixed /
prefill steps — where these fast paths engage — nothing is CUDA-graph
captured, so the switch is live exactly where it matters.)

``VLLM_ROCM_ROUTER_CHAIN_CTRL`` (default /tmp/vllm_router_chain.enabled) may
name a control file containing "1"/"0". A missing or removed file means on:
the fast paths stay on when the switch was never wired. The file is re-read
at most once per second.
"""

_CTRL_CACHE: dict = {"mtime": -1.0, "path": None, "on": True}


def router_chain_fastpath_enabled() -> bool:
    """Whether the router-chain fast paths should engage on this call."""
    import os
    import time

    path = os.environ.get(
        "VLLM_ROCM_ROUTER_CHAIN_CTRL", "/tmp/vllm_router_chain.enabled"
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
    return cache["on"]
