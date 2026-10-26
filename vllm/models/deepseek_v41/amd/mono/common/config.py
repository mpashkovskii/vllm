# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-model configuration of the unified AMD K2 MoE mono kernel: the model
shape, quantization and parallelism one instance describes, the refusal gate
that polices an unsupported deployment, and the process-wide active selection.

The MoE stage sources (``stages.dims``, ``stages.seam``, ``stages.router``,
``stages.moe_shape``) read their model-shape constants from ``active()`` the way
``common.arch`` reads the grid width from the environment: a served process runs
one model, so the shape is a per-process constant resolved once, and the runner
selects it (``VLLM_MONO_K2_MODEL``) before the device modules are traced. The
build key (``stages.moe_shape.MoeBuild``) *also* carries the shape scalars, so
two models never share a FlyDSL cache entry and the CPU tests can read a build's
shape without a global.

This module is model-agnostic but for the registered instances at the bottom;
when the package is lifted to a shared location those instances move to their
model packages and register themselves on import.
"""

import os
from dataclasses import dataclass

# The activation x weight quant a model's routed experts run. DeepSeek-V4.1's
# ``stages.moe`` implements ``mxfp4_mxfp8`` today; the others name a build each
# of the merged mono MoE models (K3 / MuseSpark inline MXFP4, MiniMax A16W4) so
# a config can describe them before the shared stage body gains their path.
QUANTS = ("mxfp4_mxfp8", "mxfp4_mxfp4", "a16w4")

MODEL_ENV = "VLLM_MONO_K2_MODEL"
ARCH_ENV = "VLLM_MONO_K2_ARCH"

# a CU's LDS: 160 KiB on gfx950 (MI355X / MI350X / MI350P), 64 KiB on gfx942.
_LDS_950 = 160 * 1024
_LDS_942 = 64 * 1024


@dataclass(frozen=True)
class ParallelConfig:
    """A deployment's parallelism. Only ``tp`` shapes the K2 kernel (a rank owns
    ``moe_inter // tp`` of every expert); the rest are model-partition axes the
    intra-layer kernel does not implement and ``gate`` refuses (for now)."""

    tp: int
    pp: int = 1  # pipeline parallel: splits layers across ranks
    dp: int = 1  # data parallel
    ep: int = 1  # expert parallel
    dcp: int = 1  # decode context parallel
    sp: bool = False  # sequence parallel (reduce-scatter in place of all-reduce)


@dataclass(frozen=True)
class ArchConfig:
    """A GPU the kernel can be built for. ``cus`` is the grid width (one resident
    CTA per compute unit). gfx950 parts differ only in CU count (MI355X / MI350X
    256, MI350P 128); gfx942 lacks several instructions and uses FNUZ FP8. The
    runtime grid-width authority is ``common.arch`` (``resolve_tiling``); these
    are the named, enumerable configs a model declares support against."""

    name: str  # the SKU, e.g. "MI350P"
    gfx: str  # "gfx950" | "gfx942"
    cus: int  # compute units == grid width (BLOCKS)
    lds_bytes: int  # a CU's LDS
    fnuz: bool  # gfx942 FNUZ FP8 (bias 8, no -0) vs gfx950 OCP e4m3

    @property
    def key(self) -> tuple[str, int]:
        """What a model's ``supports`` is keyed on: a tiling class, not a SKU."""
        return (self.gfx, self.cus)


# the 256-CU gfx950 and gfx942 tiling both V4.1 supports today (MI350P's 128-CU
# grid needs the down / wo_b task decomposition re-derived -- see common.arch).
_CU256 = (("gfx950", 256), ("gfx942", 256))


@dataclass(frozen=True)
class MoeMonoConfig:
    """One model's K2 MoE kernel shape. ``shared_inter == 0`` is a model with no
    shared expert (its shared stages collapse out). Scalars only: the shape
    fields feed ``MoeBuild`` and so the FlyDSL JIT cache key. ``supports`` is the
    ``(gfx, cus)`` tiling classes the model's kernel is validated for."""

    name: str
    hidden: int  # the residual stream width H
    moe_inter: int  # a routed expert's full intermediate width
    shared_inter: int  # the shared expert's full width; 0 => no shared expert
    experts: int  # routed experts
    topk: int  # a token's routed experts
    moe_pad: int  # the loader's pad_align for a rank's intermediate share
    quant: str  # one of QUANTS
    parallel: ParallelConfig
    supports: tuple[tuple[str, int], ...] = _CU256
    # the width the experts actually see: a latent/pooled MoE (K3, MuseSpark)
    # compresses the stream first, so this is below ``hidden``; 0 => == hidden
    # (DeepSeek-V4.1, MiniMax: the experts read the stream directly).
    moe_hidden: int = 0

    def __post_init__(self):
        if self.moe_hidden == 0:
            object.__setattr__(self, "moe_hidden", self.hidden)


def gate(cfg: MoeMonoConfig) -> list[str]:
    """Every reason this deployment cannot run the K2 MoE kernel, or an empty
    list. Fails closed: an axis the kernel does not implement is a refusal with
    one clear reason, not a silent misroute. The pipeline/data/expert/context
    axes are *model-partition* concerns orthogonal to the intra-layer kernel; a
    future reduce-scatter epilogue would lift the ``sp`` refusal behind here."""
    out: list[str] = []
    p = cfg.parallel
    if p.pp != 1:
        out.append(
            f"pipeline parallelism (pp={p.pp}) is unsupported: the mono K2 runs a "
            "whole decoder layer in one launch; PP splits layers across ranks, "
            "which must assign each rank its layer set outside the kernel"
        )
    if p.dp != 1:
        out.append(f"data parallelism (dp={p.dp}) is unsupported")
    if p.ep != 1:
        out.append(
            f"expert parallelism (ep={p.ep}) is unsupported: the kernel reduces "
            "every expert's columns by their TP owner, not by an expert-parallel "
            "group"
        )
    if p.dcp != 1:
        out.append(f"decode context parallelism (dcp={p.dcp}) is unsupported")
    if p.sp:
        out.append(
            "sequence parallelism is unsupported: the o_proj epilogue all-reduces "
            "the residual, it does not reduce-scatter it"
        )
    if cfg.quant not in QUANTS:
        out.append(f"quant {cfg.quant!r} is unknown (one of {QUANTS})")
    return out


def arch_gate(cfg: MoeMonoConfig, arch: "ArchConfig") -> list[str]:
    """Why ``cfg``'s kernel cannot be built for ``arch``, or an empty list. Fails
    closed on a tiling class the model is not validated for (e.g. V4.1 on MI350P's
    128-CU grid, whose down / wo_b stages need re-deriving); ``common.arch`` is
    the runtime authority that refuses it before any kernel is traced."""
    if arch.key not in cfg.supports:
        return [
            f"{arch.name} ({arch.gfx}, {arch.cus} CUs) has no {cfg.name} tiling "
            f"(supported: {', '.join(f'{g}/{c}' for g, c in cfg.supports)})"
        ]
    return []


# -------------------------------------------------------------- the registries
_REGISTRY: dict[str, MoeMonoConfig] = {}
_ARCHS: dict[str, ArchConfig] = {}


def register(cfg: MoeMonoConfig) -> MoeMonoConfig:
    if cfg.name in _REGISTRY and _REGISTRY[cfg.name] != cfg:
        raise ValueError(f"a different K2 MoE config is already {cfg.name!r}")
    _REGISTRY[cfg.name] = cfg
    return cfg


def register_arch(arch: ArchConfig) -> ArchConfig:
    if arch.name in _ARCHS and _ARCHS[arch.name] != arch:
        raise ValueError(f"a different arch is already {arch.name!r}")
    _ARCHS[arch.name] = arch
    return arch


def arch_named(name: str) -> ArchConfig:
    try:
        return _ARCHS[name]
    except KeyError:
        raise ValueError(f"no such arch {name!r} (have {sorted(_ARCHS)})") from None


def active_arch() -> ArchConfig:
    """The GPU this process targets: ``VLLM_MONO_K2_ARCH`` (default MI355X)."""
    return arch_named(os.environ.get(ARCH_ENV, "MI355X"))


def named(name: str) -> MoeMonoConfig:
    """A registered model by name. A model's own stages bind to it directly
    (``named("kimi_k3")``), rather than ``active()``, so every model's shape is
    its own even when several are registered in one process (the CPU tests)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"no such K2 MoE model {name!r} (have {sorted(_REGISTRY)})"
        ) from None


def active() -> MoeMonoConfig:
    """The model this process serves: ``VLLM_MONO_K2_MODEL`` (default the sole
    registered DeepSeek-V4.1 shape). Read at module import by the stages, as the
    grid width is: the runner sets it before the device modules load."""
    name = os.environ.get(MODEL_ENV)
    return DSV41 if name is None else named(name)


# -------------------------------------------------------------- registered archs
# gfx950: MI355X / MI350X at 256 CUs, MI350P at 128. gfx942: MI300X / MI325X at
# the validated 256-CU grid (FNUZ FP8).
MI355X = register_arch(ArchConfig("MI355X", "gfx950", 256, _LDS_950, False))
MI350X = register_arch(ArchConfig("MI350X", "gfx950", 256, _LDS_950, False))
MI350P = register_arch(ArchConfig("MI350P", "gfx950", 128, _LDS_950, False))
MI300X = register_arch(ArchConfig("MI300X", "gfx942", 256, _LDS_942, True))
MI325X = register_arch(ArchConfig("MI325X", "gfx942", 256, _LDS_942, True))

# -------------------------------------------------------------- registered models
# DeepSeek-V4.1-Flash. The ``parallel`` here is the canonical default; the runner
# rebuilds the config with the deployment's ParallelConfig for ``gate``. It is
# validated on the 256-CU gfx950 / gfx942 grid only -- MI350P's 128-CU grid needs
# the down / wo_b task decomposition re-derived (see ``common.arch``), so the
# arch gate refuses it until that lands. (On the move to a shared location this
# instance lives in the deepseek_v41 package.)
DSV41 = register(
    MoeMonoConfig(
        name="dsv41",
        hidden=5120,
        moe_inter=2304,
        shared_inter=2304,
        experts=384,
        topk=6,
        moe_pad=128,
        quant="mxfp4_mxfp8",
        parallel=ParallelConfig(tp=4),
        supports=_CU256,
    )
)

# Kimi-K3 (merged mono MoE, pr60683 + pr61435). A latent MoE: the experts read
# the LAT=3584 compressed stream, not HIDDEN=7168. 896 routed experts, top-16, a
# shared expert; inline MXFP4 x MXFP4. TP8, gfx950 and gfx942. Full intermediates
# are the per-rank RI=384 / SI=768 times TP. (Its stages live in the kimi_k3
# package; this entry moves there on the shared-package relocation.)
KIMI_K3 = register(
    MoeMonoConfig(
        name="kimi_k3",
        hidden=7168,
        moe_hidden=3584,
        moe_inter=384 * 8,
        shared_inter=768 * 8,
        experts=896,
        topk=16,
        moe_pad=128,
        quant="mxfp4_mxfp4",
        parallel=ParallelConfig(tp=8),
        supports=_CU256,
    )
)

# MiniMax-M3 (merged mono MoE). 128 routed experts, top-4, a fused shared expert
# (union slot 0); A16W4 (int4 weight, bf16 activation). The experts read the
# stream directly (no latent compression). TP4, gfx950 (256-CU) only. Full
# intermediates are the per-rank INTER=768 times TP; the shared expert shares the
# routed width.
MINIMAX_M3 = register(
    MoeMonoConfig(
        name="minimax_m3",
        hidden=6144,
        moe_inter=768 * 4,
        shared_inter=768 * 4,
        experts=128,
        topk=4,
        moe_pad=128,
        quant="a16w4",
        parallel=ParallelConfig(tp=4),
        supports=(("gfx950", 256),),
    )
)
