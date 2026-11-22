# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group

_ubatching = None


def _get_ubatching():
    """Lazily import the DBO ubatch helpers.

    Imported on first TP collective inside a DBO forward rather than at module
    import time: ``vllm.v1.worker.ubatching`` pulls in ``forward_context``,
    which is too heavy to order against ``vllm.distributed``'s own imports.
    """
    global _ubatching
    if _ubatching is None:
        from vllm.v1.worker import ubatching as _ub

        _ubatching = _ub
    return _ubatching


def _dbo_collective(fn, input_: torch.Tensor) -> torch.Tensor:
    """Run a TP-group collective on the DBO comm stream, yield-wrapped.

    Both ubatch threads pass through the same matched sequence of these, so
    the cooperative yields keep the collective issue order identical on every
    rank while the other ubatch's compute overlaps the transfer. The sync
    variants record/wait the compute_done/comm_done events, which is the GPU
    side of the data dependency; ``record_stream`` covers the caching
    allocator side for tensors produced on the compute stream and consumed
    here (and vice versa for the output).
    """
    ub = _get_ubatching()
    ub.dbo_yield_and_switch_from_compute_to_comm()
    out = fn(input_)
    stream = torch.cuda.current_stream()
    input_.record_stream(stream)
    if out is not input_:
        out.record_stream(stream)
    ub.dbo_yield_and_switch_from_comm_to_compute()
    return out


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if _get_ubatching().dbo_enabled():
        return _dbo_collective(get_tp_group().all_reduce, input_)
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    if _get_ubatching().dbo_enabled():
        return _dbo_collective(
            lambda t: get_tp_group().all_gather(t, dim), input_
        )
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
