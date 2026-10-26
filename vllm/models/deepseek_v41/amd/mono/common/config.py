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

# The activation x weight MXFP quant a model's routed experts run. Only the
# first is implemented today; ``stages.moe`` raises for the others at build.
QUANTS = ("mxfp4_mxfp8", "mxfp4_mxfp4")

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


def active() -> MoeMonoConfig:
    """The model this process serves: ``VLLM_MONO_K2_MODEL`` (default the sole
    registered DeepSeek-V4.1 shape). Read at module import by the stages, as the
    grid width is: the runner sets it before the device modules load."""
    name = os.environ.get(MODEL_ENV)
    if name is None:
        return DSV41
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"{MODEL_ENV}={name!r}: no such K2 MoE model (have {sorted(_REGISTRY)})"
        ) from None


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
