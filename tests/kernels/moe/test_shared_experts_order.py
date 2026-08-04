# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SharedExperts._determine_shared_experts_order.

These exercise the platform-gating logic that decides whether the shared
experts run overlapped on a separate auxiliary stream. The aux-stream path
must be eligible on any cuda-alike platform (CUDA and ROCm), not just CUDA,
mirroring aux_stream() which allocates the stream via is_cuda_alike().

The tests run on any platform (including CPU) because current_platform and
the stream are patched, so the branch under test does not require a GPU.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.runner import shared_experts as se_mod
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


class _FakePlatform:
    def __init__(self, kind: str):
        # kind is one of "cuda", "rocm", "cpu"
        self._kind = kind

    def is_cuda(self) -> bool:
        return self._kind == "cuda"

    def is_rocm(self) -> bool:
        return self._kind == "rocm"

    def is_cuda_alike(self) -> bool:
        return self._kind in ("cuda", "rocm")


def _make_shared_experts(
    *,
    stream: object | None,
    disable_overlap: bool,
    mk_can_overlap: bool,
) -> SharedExperts:
    """Build a SharedExperts without running its heavy __init__."""
    obj = SharedExperts.__new__(SharedExperts)
    obj._stream = stream
    obj._mk_can_overlap_shared_experts = lambda: mk_can_overlap
    # _disable_shared_experts_overlap is a property that reads _moe_config;
    # override it on the instance via a tiny shim type to force the value.
    obj.__dict__["_forced_disable_overlap"] = disable_overlap
    return obj


@pytest.fixture(autouse=True)
def _patch_disable_overlap(monkeypatch):
    """Force _disable_shared_experts_overlap to read the injected flag."""
    monkeypatch.setattr(
        SharedExperts,
        "_disable_shared_experts_overlap",
        property(lambda self: self.__dict__["_forced_disable_overlap"]),
    )


def _set_platform(monkeypatch, kind: str):
    monkeypatch.setattr(se_mod, "current_platform", _FakePlatform(kind))


@pytest.mark.parametrize("platform_kind", ["cuda", "rocm"])
def test_cuda_alike_uses_aux_stream(monkeypatch, platform_kind):
    """On CUDA and ROCm, a small batch with a live stream overlaps."""
    _set_platform(monkeypatch, platform_kind)
    se = _make_shared_experts(
        stream=object(), disable_overlap=False, mk_can_overlap=False
    )
    hidden_states = torch.empty(1)  # 1 token, well under the threshold
    assert (
        se._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )


def test_cpu_never_uses_aux_stream(monkeypatch):
    """Non-cuda-alike (CPU) never takes the aux-stream path."""
    _set_platform(monkeypatch, "cpu")
    se = _make_shared_experts(
        stream=object(), disable_overlap=False, mk_can_overlap=False
    )
    hidden_states = torch.empty(1)
    assert (
        se._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.NO_OVERLAP
    )


@pytest.mark.parametrize("platform_kind", ["cuda", "rocm"])
def test_no_stream_falls_back_to_no_overlap(monkeypatch, platform_kind):
    """Even cuda-alike falls back when no aux stream was allocated."""
    _set_platform(monkeypatch, platform_kind)
    se = _make_shared_experts(
        stream=None, disable_overlap=False, mk_can_overlap=False
    )
    hidden_states = torch.empty(1)
    assert (
        se._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.NO_OVERLAP
    )


@pytest.mark.parametrize("platform_kind", ["cuda", "rocm"])
def test_batch_over_threshold_no_overlap(monkeypatch, platform_kind):
    """Batches above the token threshold do not overlap."""
    _set_platform(monkeypatch, platform_kind)
    se = _make_shared_experts(
        stream=object(), disable_overlap=False, mk_can_overlap=False
    )
    over = envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD + 1
    hidden_states = torch.empty(over)
    assert (
        se._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.NO_OVERLAP
    )


def test_disable_overlap_short_circuits(monkeypatch):
    """_disable_shared_experts_overlap wins regardless of platform/stream."""
    _set_platform(monkeypatch, "cuda")
    se = _make_shared_experts(
        stream=object(), disable_overlap=True, mk_can_overlap=False
    )
    hidden_states = torch.empty(1)
    assert (
        se._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.NO_OVERLAP
    )


def test_mk_overlap_takes_precedence(monkeypatch):
    """MK internal overlap is chosen before the aux-stream check."""
    _set_platform(monkeypatch, "cuda")
    se = _make_shared_experts(
        stream=object(), disable_overlap=False, mk_can_overlap=True
    )
    hidden_states = torch.empty(1)
    assert (
        se._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
    )
