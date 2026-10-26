# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The unified K2 MoE config describes every merged mono MoE model, not just
DeepSeek-V4.1. Each registered MoeMonoConfig is cross-checked against the shape
constants its model's own stage module still defines, so the registry cannot
drift from the kernels it describes. (Driving those stage bodies from the config
-- as DeepSeek-V4.1 already is -- lands when the config module is relocated to a
shared package, so the kimi_k3 / minimax_m3 packages need not import it.)

Host-only: ``flydsl`` imports the stage modules, no GPU; the grid is pinned to
the 256-CU target so MiniMax's dependency on DeepSeek-V4.1's common/plan resolves
on any box.
"""

import os

import pytest

os.environ.setdefault("ARCH", "gfx950")
os.environ.setdefault("VLLM_DSV41_MONO_BLOCKS", "256")

pytest.importorskip("flydsl")

try:
    from vllm.models.deepseek_v41.amd.mono.common import config as C
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe as kmoe
    from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn as mffn
except Exception as e:  # pragma: no cover - environment without the toolchain
    pytest.skip(f"the mono stages did not import: {e}", allow_module_level=True)


def test_merged_moe_models_registered():
    for name in ("dsv41", "kimi_k3", "minimax_m3"):
        cfg = C.named(name)
        assert cfg.quant in C.QUANTS
        assert cfg.moe_hidden > 0 and cfg.hidden > 0


def test_kimi_k3_config_matches_stage_constants():
    cfg = C.named("kimi_k3")
    tp = cfg.parallel.tp
    assert cfg.hidden == kmoe.HIDDEN == 7168
    assert cfg.experts == kmoe.E == 896
    assert cfg.topk == kmoe.TOPK == 16
    assert tp == kmoe.TP == 8
    # a latent MoE: the experts read the compressed LAT width, not HIDDEN
    assert cfg.moe_hidden == kmoe.LAT == 3584 < cfg.hidden
    # the config carries full widths; the stage keeps a rank's share
    assert cfg.moe_inter == kmoe.RI * tp
    assert cfg.shared_inter == kmoe.SI * tp
    assert cfg.quant == "mxfp4_mxfp4"


def test_minimax_m3_config_matches_stage_constants():
    cfg = C.named("minimax_m3")
    tp = cfg.parallel.tp
    assert cfg.hidden == mffn.HIDDEN == 6144
    assert cfg.experts == mffn.EXPERTS == 128
    assert cfg.topk == mffn.TOPK == 4
    assert cfg.moe_inter == mffn.INTER * tp  # full = a rank's INTER times TP
    # MiniMax reads the stream directly (no latent compression)
    assert cfg.moe_hidden == cfg.hidden
    assert cfg.quant == "a16w4"


def test_arch_support_differs_by_model():
    mi300x, mi355x, mi350p = (C.arch_named(n) for n in ("MI300X", "MI355X", "MI350P"))
    # Kimi-K3 runs on gfx942 and gfx950 (256 CUs); MiniMax on gfx950 only.
    assert C.arch_gate(C.KIMI_K3, mi300x) == []
    assert C.arch_gate(C.KIMI_K3, mi355x) == []
    assert C.arch_gate(C.MINIMAX_M3, mi300x)  # refused: gfx942 not supported
    assert C.arch_gate(C.MINIMAX_M3, mi355x) == []
    # none of the merged models has the 128-CU (MI350P) tiling yet
    for cfg in (C.DSV41, C.KIMI_K3, C.MINIMAX_M3):
        assert C.arch_gate(cfg, mi350p)


def test_parallelism_gate_applies_to_every_model():
    for cfg in (C.DSV41, C.KIMI_K3, C.MINIMAX_M3):
        from dataclasses import replace

        assert C.gate(cfg) == []  # their canonical TP-only configs serve
        assert C.gate(replace(cfg, parallel=replace(cfg.parallel, pp=2)))
