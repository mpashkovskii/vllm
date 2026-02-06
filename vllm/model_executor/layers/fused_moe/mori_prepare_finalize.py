# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Feature flag to enable dispatch-free optimization for TP+EP
_DISPATCH_FREE_ENABLED = os.environ.get("VLLM_MORI_EP_DISPATCH_FREE", "1") == "1"


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MoRI kernels.
    
    Automatically detects TP+EP mode and skips dispatch when both are enabled,
    since all GPUs have identical data after TP all-reduce.
    """

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        # Optional parameters for TP+EP dispatch-free optimization
        num_local_experts: int | None = None,
        rank_expert_offset: int | None = None,
        ep_group: torch.distributed.ProcessGroup | None = None,
        enable_dispatch_free: bool = True,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        
        # TP+EP dispatch-free optimization
        self.use_dispatch_free = (
            _DISPATCH_FREE_ENABLED
            and enable_dispatch_free
            and num_local_experts is not None
            and rank_expert_offset is not None
            and ep_group is not None
        )
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = rank_expert_offset
        self.ep_group = ep_group
        
        # State for dispatch-free combine phase
        self._local_token_indices: torch.Tensor | None = None
        self._original_batch_size: int = 0
        
        if self.use_dispatch_free:
            logger.info(
                "MORI-EP dispatch-free optimization enabled. "
                "Dispatch All-to-All will be skipped for ~45%% communication savings."
            )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """
        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        
        if self.use_dispatch_free:
            # TP+EP mode: skip dispatch, filter locally
            return self._prepare_dispatch_free(
                a1, topk_weights, topk_ids, quant_config
            )
        
        # Standard mode: use MORI dispatch
        a1, scale = self._quantize_if_needed(a1, quant_config)

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self.use_dispatch_free:
            # TP+EP mode: use all-reduce instead of MORI combine
            self._finalize_dispatch_free(output, fused_expert_output)
        else:
            # Standard mode: use MORI combine
            num_token = output.shape[0]
            result = self.mori_op.combine(
                fused_expert_output,
                None,
                topk_ids,
            )[0]
            output.copy_(result[:num_token])
    
    def _quantize_if_needed(
        self,
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply FP8 quantization if enabled."""
        scale = None
        if self.use_fp8_dispatch:
            from aiter import QuantType, get_hip_quant

            if quant_config.is_block_quantized:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
            elif quant_config.is_per_act_token:
                quant_func = get_hip_quant(QuantType.per_Token)
                a1, scale = quant_func(a1, quant_dtype=current_platform.fp8_dtype())
        return a1, scale
    
    def _prepare_dispatch_free(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """Dispatch-free prepare for TP+EP mode."""
        # 1. Identify tokens needing ANY local expert
        assert self.rank_expert_offset is not None
        assert self.num_local_experts is not None
        
        local_end = self.rank_expert_offset + self.num_local_experts
        is_local_expert = (topk_ids >= self.rank_expert_offset) & (
            topk_ids < local_end
        )
        token_needs_local = is_local_expert.any(dim=1)

        # 2. Filter to relevant tokens
        local_token_indices = token_needs_local.nonzero(as_tuple=True)[0]
        num_local_tokens = local_token_indices.shape[0]

        if num_local_tokens == 0:
            # No tokens need our experts
            empty_a1 = torch.empty((0, a1.shape[1]), dtype=a1.dtype, device=a1.device)
            empty_ids = torch.empty((0, topk_ids.shape[1]), dtype=torch.int32, device=a1.device)
            empty_weights = torch.empty((0, topk_weights.shape[1]), dtype=topk_weights.dtype, device=a1.device)
            empty_recv = torch.zeros(self.num_local_experts, dtype=torch.int32, device=a1.device)
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=empty_recv, expert_num_tokens_cpu=None
            )
            self._local_token_indices = local_token_indices
            self._original_batch_size = a1.shape[0]
            return (empty_a1, None, expert_tokens_meta, empty_ids, empty_weights)

        # 3. Extract local tokens
        local_a1 = a1[local_token_indices]
        local_topk_ids = topk_ids[local_token_indices]
        local_topk_weights = topk_weights[local_token_indices]

        # 4. Convert global → local expert IDs
        local_expert_ids = local_topk_ids - self.rank_expert_offset
        is_local = (local_expert_ids >= 0) & (local_expert_ids < self.num_local_experts)
        local_topk_weights = local_topk_weights * is_local.float()
        local_expert_ids = local_expert_ids.clamp(0, self.num_local_experts - 1)

        # 5. Quantize if needed
        local_a1, scale = self._quantize_if_needed(local_a1, quant_config)

        # 6. Count tokens per expert
        expert_counts = torch.zeros(self.num_local_experts, dtype=torch.int32, device=a1.device)
        for i in range(self.num_local_experts):
            expert_counts[i] = ((local_expert_ids == i) & is_local[:, :]).sum()

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_counts, expert_num_tokens_cpu=None
        )

        # 7. Store state for finalize
        self._local_token_indices = local_token_indices
        self._original_batch_size = a1.shape[0]

        return (local_a1, scale, expert_tokens_meta, local_expert_ids, local_topk_weights)
    
    def _finalize_dispatch_free(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
    ) -> None:
        """Dispatch-free finalize using all-reduce."""
        assert self.ep_group is not None
        
        local_contribution = torch.zeros_like(output)
        if (
            fused_expert_output.numel() > 0
            and self._local_token_indices is not None
            and len(self._local_token_indices) > 0
        ):
            local_contribution[self._local_token_indices] = fused_expert_output[
                : len(self._local_token_indices)
            ]

        torch.distributed.all_reduce(
            local_contribution,
            op=torch.distributed.ReduceOp.SUM,
            group=self.ep_group,
        )
        output.copy_(local_contribution)
