# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The unified K2 MoE kernel's config layer, on CPU (no GPU, no kernel build).

Two guarantees:

1. **Bit-identical placement.** After the MoE stages were made to read their
   model shape from ``common.config.active()`` instead of module literals, the
   derived task counts, CTA placements, route-table offsets, ``Dims`` widths and
   per-width scratch layouts of DeepSeek-V4.1 must equal the snapshot captured
   from the pre-refactor source (``dsv41_mono_placement_golden.json``). The
   golden is the contract that the config refactor changed no numbers.

2. **The refusal gate**, including pipeline parallelism, which the intra-layer
   kernel does not implement and must refuse rather than misroute.

Host-only: the grid is pinned to the 256-CU target so the snapshot is the same
on any box (a 128-CU MI350P has no DSv4.1 tiling), and ``flydsl`` is only needed
to import the stage modules, not a GPU.
"""

import json
import os
from pathlib import Path

import pytest

# Pin the arch/grid before the stage modules resolve their per-process constants,
# so the snapshot is hardware-independent (see module docstring).
os.environ.setdefault("ARCH", "gfx950")
os.environ.setdefault("VLLM_DSV41_MONO_BLOCKS", "256")

pytest.importorskip("flydsl")

try:
    from vllm.models.deepseek_v41.amd.mono import layer
    from vllm.models.deepseek_v41.amd.mono.common import config as C
    from vllm.models.deepseek_v41.amd.mono.common.plan import BLOCKS
    from vllm.models.deepseek_v41.amd.mono.stages import moe
    from vllm.models.deepseek_v41.amd.mono.stages.dims import Dims
    from vllm.models.deepseek_v41.amd.mono.stages.moe_shape import MoeBuild, route_shape
except Exception as e:  # pragma: no cover - environment without the toolchain
    pytest.skip(f"the mono stages did not import: {e}", allow_module_level=True)

if BLOCKS != 256:  # pragma: no cover - a box whose grid did not pin to the target
    pytest.skip(f"placement golden is for a 256-CTA grid, got {BLOCKS}")

GOLDEN = json.loads(
    (Path(__file__).parent / "dsv41_mono_placement_golden.json").read_text()
)

# The same enumeration the golden was captured with.
_RS_PROPS = [
    "experts", "topk", "tokens", "tp", "ug_groups", "ug_parts", "ug_k_slices",
    "picks", "max_u", "router_groups", "router_tasks", "router0", "route0",
    "shared0", "shared_units", "ug0", "tab_words", "multi", "slot_tiles",
    "max_tiles", "tile_first0", "tile_slot0", "queue0", "expert_slot0",
    "order0", "first0", "mask0", "tile_picks",
]
_DIMS_PROPS = [
    "inter_real", "inter", "sh_inter", "shared_tasks", "mid_words",
    "down_scale_cols",
]
_WIDTHS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 42, 48]
_ROUTINGS = [(384, 6, 2), (128, 6, 1)]  # target layer, DSpark draft layer


def _snap_rs(rs):
    out = {}
    for p in _RS_PROPS:
        v = getattr(rs, p)
        out[p] = list(v) if isinstance(v, tuple) else v
    return out


def test_route_shape_bit_identical():
    for (e, k, ug) in _ROUTINGS:
        for tp in (2, 4, 8):
            for s in _WIDTHS:
                tag = f"s{s}_tp{tp}_e{e}_k{k}_ug{ug}"
                key = MoeBuild(tokens=s, experts=e, topk=k, tp=tp, ug_groups=ug)
                assert _snap_rs(route_shape(key)) == GOLDEN["route_shape"][tag], tag


def test_moe_scratch_bit_identical():
    for (e, k, ug) in _ROUTINGS:
        for tp in (2, 4, 8):
            for s in _WIDTHS:
                tag = f"s{s}_tp{tp}_e{e}_k{k}_ug{ug}"
                key = MoeBuild(tokens=s, experts=e, topk=k, tp=tp, ug_groups=ug)
                got = {n: list(v) for n, v in moe.scratch_layout(key).items()}
                assert got == GOLDEN["moe_scratch"][tag], tag


def test_dims_bit_identical():
    for tp in (2, 4, 8):
        got = {p: getattr(Dims(tp), p) for p in _DIMS_PROPS}
        assert got == GOLDEN["dims"][str(tp)], tp


def test_layer_scratch_bit_identical():
    for tp in (2, 4):
        for s in _WIDTHS:
            got = {n: list(v) for n, v in layer.scratch_layout(s, tp).items()}
            assert got == GOLDEN["layer_scratch"][f"s{s}_tp{tp}"], (s, tp)


def test_build_key_carries_shape():
    key = MoeBuild(tokens=8, tp=4)
    cfg = C.active()
    assert (key.hidden, key.moe_inter, key.shared_inter, key.moe_pad, key.quant) == (
        cfg.hidden, cfg.moe_inter, cfg.shared_inter, cfg.moe_pad, cfg.quant
    )


def test_gate_accepts_supported_tp():
    for tp in (2, 4, 8):
        cfg = C.MoeMonoConfig(
            "t", 5120, 2304, 2304, 384, 6, 128, "mxfp4_mxfp8", C.ParallelConfig(tp=tp)
        )
        assert C.gate(cfg) == []


@pytest.mark.parametrize(
    "parallel, needle",
    [
        (dict(tp=4, pp=2), "pipeline parallelism"),
        (dict(tp=4, dp=2), "data parallelism"),
        (dict(tp=4, ep=2), "expert parallelism"),
        (dict(tp=4, dcp=2), "decode context parallelism"),
        (dict(tp=4, sp=True), "sequence parallelism"),
    ],
)
def test_gate_refuses_unsupported_parallelism(parallel, needle):
    cfg = C.MoeMonoConfig(
        "t", 5120, 2304, 2304, 384, 6, 128, "mxfp4_mxfp8", C.ParallelConfig(**parallel)
    )
    reasons = C.gate(cfg)
    assert any(needle in r for r in reasons), reasons


def test_gate_refuses_unknown_quant():
    cfg = C.MoeMonoConfig(
        "t", 5120, 2304, 2304, 384, 6, 128, "fp16", C.ParallelConfig(tp=4)
    )
    assert any("quant" in r for r in C.gate(cfg))


def test_mi350p_arch_registered():
    mi350p = C.arch_named("MI350P")
    assert (mi350p.gfx, mi350p.cus, mi350p.fnuz) == ("gfx950", 128, False)
    assert mi350p.lds_bytes == 160 * 1024
    # the gfx950 256-CU parts and the gfx942 family are registered too
    for name in ("MI355X", "MI350X", "MI300X", "MI325X"):
        assert C.arch_named(name).cus == 256
    assert C.arch_named("MI300X").fnuz is True


def test_default_active_arch_is_mi355x():
    assert C.active_arch().name == "MI355X"


def test_arch_gate_refuses_mi350p_for_dsv41():
    # DeepSeek-V4.1 is validated on the 256-CU grid only; MI350P's 128-CU grid
    # has no tiling yet, so the arch gate refuses it.
    reasons = C.arch_gate(C.DSV41, C.arch_named("MI350P"))
    assert reasons and "MI350P" in reasons[0]
    for name in ("MI355X", "MI350X", "MI300X", "MI325X"):
        assert C.arch_gate(C.DSV41, C.arch_named(name)) == []


def test_no_shared_expert_collapses_placement():
    # A model with no shared expert (shared_inter == 0) has no shared units, so
    # the ug stage starts where the shared placement would have ended.
    rs = route_shape(MoeBuild(tokens=8, tp=4))
    assert rs.shared_units > 0 and rs.ug0 == rs.shared0 + rs.shared_units
    d0 = Dims(4)
    assert d0.sh_inter > 0
    # the derivation is sh_inter // ROWS; at sh_inter 0 it is 0 units (the muse
    # spark shape, exercised when that model migrates).
