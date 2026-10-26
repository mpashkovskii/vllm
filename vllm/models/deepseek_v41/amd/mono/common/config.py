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
class MoeMonoConfig:
    """One model's K2 MoE kernel shape. ``shared_inter == 0`` is a model with no
    shared expert (its shared stages collapse out). Scalars only: the shape
    fields feed ``MoeBuild`` and so the FlyDSL JIT cache key."""

    name: str
    hidden: int  # the residual stream width H
    moe_inter: int  # a routed expert's full intermediate width
    shared_inter: int  # the shared expert's full width; 0 => no shared expert
    experts: int  # routed experts
    topk: int  # a token's routed experts
    moe_pad: int  # the loader's pad_align for a rank's intermediate share
    quant: str  # one of QUANTS
    parallel: ParallelConfig


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


# -------------------------------------------------------------- the registry
_REGISTRY: dict[str, MoeMonoConfig] = {}


def register(cfg: MoeMonoConfig) -> MoeMonoConfig:
    if cfg.name in _REGISTRY and _REGISTRY[cfg.name] != cfg:
        raise ValueError(f"a different K2 MoE config is already {cfg.name!r}")
    _REGISTRY[cfg.name] = cfg
    return cfg


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


# -------------------------------------------------------------- registered models
# DeepSeek-V4.1-Flash. The ``parallel`` here is the canonical default; the runner
# rebuilds the config with the deployment's ParallelConfig for ``gate``. (On the
# move to a shared location this instance lives in the deepseek_v41 package.)
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
    )
)
