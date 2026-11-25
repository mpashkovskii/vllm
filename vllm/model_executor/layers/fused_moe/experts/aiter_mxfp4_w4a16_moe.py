# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.triton_utils import triton

logger = init_logger(__name__)

__all__ = [
    "AiterW4A16ExpertsMonolithic",
    "aiter_triton_kernel_w4a16_moe_forward",
]

# iter-14 (DeepSeek-V4.1-Flash uplift): opt-in env flag routing the W4A16
# GEMM2 (down-proj) through AITER's iter-10 FlyDSL a8w4 kernel instead of the
# triton moe_gemm_a16w4. The flag is a server-start switch (A/B): the W2 ->
# e4m3fnuz requant is built once in process_weights_after_loading and the
# packed mxfp4 W2 is dropped when the flag is on.
_FLYDSL_A8W4_G2_ENV = "VLLM_FLYDSL_A8W4_G2"
_flydsl_a8w4_g2_cache: bool | None = None


# iter-22 (DeepSeek-V4.1-Flash uplift): opt-in env flag routing the W4A16
# GEMM1 (gate+up + clamped silu) through AITER's iter-22 FlyDSL a8w4-FP4W
# kernel: fp8 A (per-token online requant of the hidden states) x the
# production mxfp4 W1 read IN PLACE -- the fp8 B operands are synthesized in
# registers from the fp4 nibbles and e8m0 group scales (no persistent fp8 W
# cache, so the iter-17 KV-capacity wall does not apply). Same load-time
# pattern as iter-14's G2: the W1 repack (fragment-aligned preshuffle + plain
# scale/shift planes) is built in process_weights_after_loading and the
# original packed mxfp4 W1 + its scale tensor are dropped when on (net
# negative VRAM: the repacked scales are smaller than the originals).
_FLYDSL_A8W4FP4_G1_ENV = "VLLM_FLYDSL_A8W4FP4_G1"
_flydsl_a8w4fp4_g1_cache: bool | None = None


def _flydsl_a8w4fp4_g1_enabled() -> bool:
    global _flydsl_a8w4fp4_g1_cache
    if _flydsl_a8w4fp4_g1_cache is None:
        val = os.environ.get(_FLYDSL_A8W4FP4_G1_ENV, "0")
        _flydsl_a8w4fp4_g1_cache = val.strip().lower() in (
            "1", "true", "yes", "on")
    return _flydsl_a8w4fp4_g1_cache


def _logical_kn(t: torch.Tensor, k_dim: int, n_dim: int, what: str) -> torch.Tensor:
    """Normalize a 3D (E, ?, ?) storage to logical (E, k_dim, n_dim).

    The w13 storage may be a transpose view (column-major for the triton
    layout); the semantic N axis is the one with ``n_dim`` elements.
    """
    if t.dim() != 3:
        raise RuntimeError(f"{what} storage must be 3D, got shape {tuple(t.shape)}")
    if t.shape[1] == k_dim and t.shape[2] == n_dim:
        return t
    if t.shape[1] == n_dim and t.shape[2] == k_dim:
        return t.transpose(1, 2)
    raise RuntimeError(
        f"{what} storage shape {tuple(t.shape)} matches neither"
        f" (E,{k_dim},{n_dim}) nor (E,{n_dim},{k_dim})"
    )


def _flydsl_a8w4_g2_enabled() -> bool:
    global _flydsl_a8w4_g2_cache
    if _flydsl_a8w4_g2_cache is None:
        val = os.environ.get(_FLYDSL_A8W4_G2_ENV, "0")
        _flydsl_a8w4_g2_cache = val.strip().lower() in ("1", "true", "yes", "on")
    return _flydsl_a8w4_g2_cache


def _aiter_w4a16_silu_via_a8w4(
    hidden_states: torch.Tensor,
    w1_data,
    w2_data,
    w1_wscale,
    w2_wscale,
    w1_bias,
    w2_bias,
    routing_data,
    gather_idx,
    scatter_idx,
    gammas,
    apply_router_weight_on_input: bool,
    swiglu_limit: float,
    unpadded_N_w1,
    unpadded_K_w1,
    unpadded_N_w2,
    unpadded_K_w2,
) -> torch.Tensor:
    """MXFP4 w4a16 MoE with a SILU (concatenated ``[gate | up]``) activation."""
    from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
    from aiter.ops.triton.quant import dynamic_mxfp8_quant

    try:
        from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
    except ImportError:
        from aiter.ops.triton.moe_op_gemm_a8w4 import moe_gemm_a8w4

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        get_aiter_mx_scale_swizzle,
    )

    swz = get_aiter_mx_scale_swizzle()
    quant_dtype = torch.float8_e4m3fn

    g1_gammas = gammas if apply_router_weight_on_input else None
    g2_gammas = None if apply_router_weight_on_input else gammas

    hidden_q, a1_scale = dynamic_mxfp8_quant(hidden_states, quant_dtype=quant_dtype)
    raw_gate_up = moe_gemm_a8w4(
        hidden_q,
        w1_data,
        a1_scale,
        w1_wscale,
        None,
        None,
        w1_bias,
        routing_data,
        gather_indx=gather_idx,
        gammas=g1_gammas,
        swizzle_mx_scale=swz,
        out_dtype=torch.bfloat16,
        apply_swiglu=False,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1,
    )
    if unpadded_N_w1 is not None:
        raw_gate_up = raw_gate_up[:, :unpadded_N_w1]

    # convert interleaved -> chunked for kernel
    gate, up = raw_gate_up[:, ::2], raw_gate_up[:, 1::2]
    raw_gate_up = torch.cat((gate, up), dim=1).contiguous()

    interim_fp8, a2_scale = fused_clamp_act_mul(
        raw_gate_up,
        swiglu_limit=swiglu_limit,
        activation="silu",
        dtype_quant=quant_dtype,
        scale_dtype_fmt="ue8m0",
        quant_block_size=32,
    )

    out = moe_gemm_a8w4(
        interim_fp8,
        w2_data,
        a2_scale,
        w2_wscale,
        None,
        None,
        w2_bias,
        routing_data,
        scatter_indx=scatter_idx,
        gammas=g2_gammas,
        swizzle_mx_scale=swz,
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
    )
    return out


def aiter_triton_kernel_w4a16_moe_forward(
    hidden_states: torch.Tensor,
    w1,
    w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: MoEActivation = MoEActivation.SWIGLUOAI,
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    score_mode: str | None = None,
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    flydsl_g1_w4: torch.Tensor | None = None,
    flydsl_g1_sc: torch.Tensor | None = None,
    flydsl_g1_sh: torch.Tensor | None = None,
    flydsl_g1_sw: torch.Tensor | None = None,
    flydsl_g1_inter: int | None = None,
    flydsl_g2_w8: torch.Tensor | None = None,
    flydsl_g2_sw: torch.Tensor | None = None,
    flydsl_g2_k_true: int | None = None,
):
    assert quant_config is not None and rocm_aiter_ops.is_enabled()
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        should_use_cdna4_mx_scale_swizzle,
    )
    from vllm.platforms.rocm import on_gfx950, on_gfx1250

    try:
        from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
    except ImportError:
        from aiter.ops.triton.moe_op_gemm_a16w4 import moe_gemm_a16w4
    try:
        from aiter.ops.triton.moe.moe_routing import routing as _routing_mod
    except ImportError:
        from aiter.ops.triton.moe_routing import routing as _routing_mod

    if on_gfx1250():
        _routing_mod.is_tdm_avail = lambda: False
    aiter_routing = _routing_mod.routing

    scale = routed_scaling_factor if routed_scaling_factor is not None else 1.0
    if hash_indices_table is not None:
        assert score_mode == "sqrtsoftplus" and input_ids is not None
        # Same block_m heuristic as aiter's routing().
        n_tokens, n_experts = gating_output.shape
        block_m = max(
            16, min(triton.next_power_of_2(max(1, n_tokens * topk // n_experts)), 128)
        )
        routing_data, gather_idx, scatter_idx = _routing_mod.routing_from_hash(
            gating_output,
            hash_indices_table,
            input_ids,
            topk,
            block_m,
            score_mode=score_mode,
            renorm=renormalize,
            routed_scaling_factor=scale,
        )
    elif score_mode is not None:
        use_grouped_topk = num_expert_group is not None and num_expert_group > 1
        routing_data, gather_idx, scatter_idx = aiter_routing(
            gating_output,
            topk,
            score_mode=score_mode,
            bias=(
                e_score_correction_bias.float()
                if e_score_correction_bias is not None
                else None
            ),
            renorm=renormalize,
            routed_scaling_factor=scale,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )
    else:
        routing_data, gather_idx, scatter_idx = aiter_routing(
            gating_output, topk, sm_first=not renormalize
        )

    if on_gfx1250():
        gather_src = gather_idx.to(torch.long) // topk
        hidden_states = hidden_states[gather_src]
        gather_idx = None

    # Expert parallelism: mark gates whose expert is not on this rank so the
    # scatter combine skips them, instead of zero-filling the skipped experts'
    # output rows.
    gate_valid = None
    if expert_map is not None:
        topk_ids = routing_data.topk_ids
        assert topk_ids is not None, (
            "expert-parallel W4A16 requires aiter routing to expose topk_ids"
        )
        gate_valid = (expert_map[topk_ids.long()] >= 0).to(torch.int32).contiguous()

    assert quant_config.w1_precision is not None
    assert quant_config.w2_precision is not None

    w1_data = w1.storage.data
    w2_data = w2.storage.data
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        weight_mx_scale,
    )

    w1_wscale = weight_mx_scale(quant_config.w1_precision).storage.data
    w2_wscale = weight_mx_scale(quant_config.w2_precision).storage.data

    gammas = routing_data.gate_scal if routing_data else None

    swiglu_alpha = (
        quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
    )
    # 7.0 is the SwigluOAI clamp, so only default to it for that activation;
    # a plain-SILU model that configures no limit must not be clamped.
    if quant_config.gemm1_clamp_limit is not None:
        swiglu_limit = quant_config.gemm1_clamp_limit
    elif activation == MoEActivation.SWIGLUOAI:
        swiglu_limit = 7.0
    else:
        swiglu_limit = None

    # SILU on gfx1250: use the verified a8w4 kernel (dynamic MXFP8); a16w4 faults.
    if activation == MoEActivation.SILU and on_gfx1250():
        if expert_map is not None:
            raise NotImplementedError(
                "Expert parallelism is not supported for SILU on gfx1250 "
                "(the a8w4 fallback kernel has no expert_map support)."
            )
        return _aiter_w4a16_silu_via_a8w4(
            hidden_states,
            w1_data,
            w2_data,
            w1_wscale,
            w2_wscale,
            quant_config.w1_bias,
            quant_config.w2_bias,
            routing_data,
            gather_idx,
            scatter_idx,
            gammas,
            apply_router_weight_on_input,
            # fused_clamp_act_mul encodes "no clamp" as 0, not None.
            swiglu_limit if swiglu_limit is not None else 0.0,
            unpadded_N_w1,
            unpadded_K_w1,
            unpadded_N_w2,
            unpadded_K_w2,
        )

    # SILU: silu(gate) * up — same kernel, just no "+1" residual in swiglu.
    swiglu_add_residual = activation != MoEActivation.SILU

    # Weight loading already swizzled the block scales on gfx950, so the kernel
    # has to be told; the load-time swizzle and this gate must agree or the
    # kernel indexes a swizzled buffer as if it were linear.
    swz = "CDNA4_SCALE" if should_use_cdna4_mx_scale_swizzle() else None

    # AITER's gfx950 Gluon a16w4 kernel (default backend="None"/auto-detect
    # picks it) calls a Triton-Gluon AMD API that isn't available in vLLM's
    # pinned Triton, so force the Triton backend on gfx950. gfx1250 already
    # used Gluon before this and is unaffected, so leave it on auto-detect.
    a16w4_backend = "triton" if on_gfx950() else None

    # iter-22 wire-in (env VLLM_FLYDSL_A8W4FP4_G1): replace the triton GEMM1
    # with AITER's iter-22 FlyDSL a8w4-FP4W stage-1 kernel (fp8 A x native
    # mxfp4 W1 with in-register fp8 synthesis; no persistent fp8 W cache, so
    # no KV-capacity cost). Requires the load-time repack (flydsl_g1_w4/sc/
    # sh/sw, built in process_weights_after_loading). Constraints match
    # iter-14's G2: non-EP path, SILU activation, and no
    # apply_router_weight_on_input (the G1 gathers hidden rows per token, so
    # a per-(token, slot) gamma cannot fold into the per-token A scale).
    if (
        flydsl_g1_w4 is not None
        and flydsl_g1_sc is not None
        and flydsl_g1_sh is not None
        and flydsl_g1_sw is not None
        and flydsl_g1_inter is not None
        and expert_map is None
        and gate_valid is None
        and activation == MoEActivation.SILU
        and not apply_router_weight_on_input
        # Batch-shape gate: the a8w4-FP4W kernel wins at large chunked-prefill
        # shapes (the mixed-step 8k/16k chunks) but loses to the tuned triton
        # path at decode-sized shapes (isolated: 487.7 us/call at 131 tokens
        # vs the 379.8 us in-server triton median; the win is the big-chunk
        # mixed GEMM). Decode stays on the production path.
        and hidden_states.shape[0] >= 2048
    ):
        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.stage1_a8w4fp4_server import (  # noqa: E501
            moe_stage1_a8w4fp4_forward,
        )

        assert routing_data.topk_ids is not None, (
            "VLLM_FLYDSL_A8W4FP4_G1 requires aiter routing to expose topk_ids"
        )
        assert routing_data.gate_scal is not None, (
            "VLLM_FLYDSL_A8W4FP4_G1 requires aiter routing to expose gate_scal"
        )
        num_experts = (
            global_num_experts
            if global_num_experts and global_num_experts > 0
            else routing_data.n_expts_tot
        )
        canon_gammas = routing_data.gate_scal.float().index_select(
            0, scatter_idx.long()
        )
        intermediate = moe_stage1_a8w4fp4_forward(
            hidden_states,
            routing_data.topk_ids,
            canon_gammas,
            scatter_idx,
            routing_data.gate_scal.numel(),
            flydsl_g1_w4,
            flydsl_g1_sc,
            flydsl_g1_sh,
            flydsl_g1_sw,
            num_experts=num_experts,
            inter_dim=flydsl_g1_inter,
            topk=topk,
            swiglu_limit=(
                swiglu_limit if swiglu_limit is not None else float("inf")
            ),
            tile_m=64,
            tile_n=192,
            tile_k=128,
            xcd_swizzle=2,
        )
    else:
        intermediate = moe_gemm_a16w4(
            hidden_states,
            w1_data,
            None,
            w1_wscale,
            None,
            None,
            quant_config.w1_bias,
            routing_data,
            gather_indx=gather_idx,
            gammas=gammas if apply_router_weight_on_input else None,
            swizzle_mx_scale=swz,
            apply_swiglu=True,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            swiglu_add_residual=swiglu_add_residual,
            unpadded_N=unpadded_N_w1,
            unpadded_K=unpadded_K_w1,
            backend=a16w4_backend,
            expert_map=expert_map,
            gate_valid=gate_valid,
        )

    # iter-14 wire-in (env VLLM_FLYDSL_A8W4_G2): replace the triton GEMM2 with
    # AITER's iter-10 FlyDSL a8w4 stage-2 kernel. Requires the load-time W2 ->
    # e4m3fnuz requant cache (flydsl_g2_w8/sw, built in
    # process_weights_after_loading) and only supports the non-EP path.
    if (
        flydsl_g2_w8 is not None
        and flydsl_g2_sw is not None
        and flydsl_g2_k_true is not None
        and expert_map is None
        and gate_valid is None
    ):
        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
            moe_stage2_a8w4_forward,
        )

        assert routing_data.topk_ids is not None, (
            "VLLM_FLYDSL_A8W4_G2 requires aiter routing to expose topk_ids"
        )
        assert routing_data.gate_scal is not None, (
            "VLLM_FLYDSL_A8W4_G2 requires aiter routing to expose gate_scal"
        )
        assert intermediate.stride(1) == 1 and intermediate.dim() == 2
        num_experts = (
            global_num_experts
            if global_num_experts and global_num_experts > 0
            else routing_data.n_expts_tot
        )
        return moe_stage2_a8w4_forward(
            intermediate,
            routing_data.topk_ids,
            routing_data.gate_scal,
            scatter_idx,
            flydsl_g2_w8,
            flydsl_g2_sw,
            num_experts=num_experts,
            model_dim=hidden_states.shape[-1],
            topk=topk,
            k_true=flydsl_g2_k_true,
            use_gamma=not apply_router_weight_on_input,
        )

    out = moe_gemm_a16w4(
        intermediate,
        w2_data,
        None,
        w2_wscale,
        None,
        None,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_idx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=swz,
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
        backend=a16w4_backend,
        expert_map=expert_map,
        gate_valid=gate_valid,
    )

    return out


class AiterW4A16ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token
        self.renormalize = moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.DeepseekV4,
            RoutingMethodType.DeepSeekV3,
        )
        # iter-14: cached W2 -> e4m3fnuz requant for the FlyDSL a8w4 GEMM2
        # (filled in process_weights_after_loading when VLLM_FLYDSL_A8W4_G2 on).
        self._flydsl_g2_w8: torch.Tensor | None = None
        self._flydsl_g2_sw: torch.Tensor | None = None
        self._flydsl_g2_k_true: int | None = None
        # iter-22: layout-only repack of the packed mxfp4 W1 + derived scale
        # planes for the FlyDSL a8w4-FP4W GEMM1 (filled in
        # process_weights_after_loading when VLLM_FLYDSL_A8W4FP4_G1 on).
        self._flydsl_g1_w4: torch.Tensor | None = None
        self._flydsl_g1_sc: torch.Tensor | None = None
        self._flydsl_g1_sh: torch.Tensor | None = None
        self._flydsl_g1_sw: torch.Tensor | None = None
        self._flydsl_g1_inter: int | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """iter-14: build the cached W2 -> fp8 requant for the a8w4 GEMM2.

        Only runs when VLLM_FLYDSL_A8W4_G2 is set: the packed mxfp4 W2 storage
        is dropped after the requant, so a server with the flag on exclusively
        runs the a8w4 GEMM2 path (the flag is a server-start A/B switch).
        """
        self.process_weights_after_loading_g1fp4(layer)
        if not _flydsl_a8w4_g2_enabled():
            return
        from vllm.platforms.rocm import on_gfx942

        if not on_gfx942():
            raise RuntimeError(
                f"{_FLYDSL_A8W4_G2_ENV} is implemented for gfx942 only"
            )

        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
            requant_w2_to_fp8col,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            weight_mx_scale,
        )

        mc = self.moe_config
        k_true = (
            mc.intermediate_size_per_partition_unpadded
            or mc.intermediate_size_per_partition
        )
        assert k_true and k_true % 32 == 0, f"bad true inter size {k_true}"

        w2_data = layer.w2_weight.storage.data
        w2_scale = weight_mx_scale(self.quant_config.w2_precision).storage.data
        w8, s_w = requant_w2_to_fp8col(w2_data, w2_scale, k_true, pair_k0=True)
        self._flydsl_g2_w8 = w8
        self._flydsl_g2_sw = s_w
        self._flydsl_g2_k_true = k_true
        # Drop the packed mxfp4 W2: with the flag on the triton GEMM2 path is
        # never dispatched, so its weight storage is dead memory (0.75 GB/layer
        # back to the KV budget).
        layer.w2_weight.storage.data = torch.empty(
            0, dtype=torch.uint8, device=w8.device
        )
        logger.info(
            "VLLM_FLYDSL_A8W4_G2: cached a8w4 W2 requant (K=%d): %s bytes",
            k_true,
            tuple(w8.shape),
        )

    def process_weights_after_loading_g1fp4(self, layer: torch.nn.Module) -> None:
        """iter-22: layout-only repack of the packed mxfp4 W1 for the
        a8w4-FP4W GEMM1 (fragment-aligned preshuffle + plain scale/shift
        planes + fp32 per-column epilog scale).

        Only runs when VLLM_FLYDSL_A8W4FP4_G1 is set: the packed mxfp4 W1
        and its production scale storage are dropped after the repack (the
        repacked sc plane plus the tiny shift/epilog planes are smaller), so
        a server with the flag on runs the a8w4-FP4W GEMM1 exclusively.
        """
        if not _flydsl_a8w4fp4_g1_enabled():
            return
        from vllm.platforms.rocm import on_gfx942

        if not on_gfx942():
            raise RuntimeError(
                f"{_FLYDSL_A8W4FP4_G1_ENV} is implemented for gfx942 only"
            )

        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
            repack_w13_pairwise_to_a8w4fp4,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            weight_mx_scale,
        )

        mc = self.moe_config
        inter_true = (
            mc.intermediate_size_per_partition_unpadded
            or mc.intermediate_size_per_partition
        )
        assert inter_true and inter_true % 32 == 0, (
            f"bad true inter size {inter_true}"
        )
        hidden = mc.hidden_dim_unpadded or mc.hidden_dim
        n_pad = 2 * mc.intermediate_size_per_partition
        kpack = hidden // 2
        kg = hidden // 32

        w1_raw = layer.w13_weight.storage.data
        w1_scale = weight_mx_scale(self.quant_config.w1_precision).storage.data
        E_ = w1_raw.shape[0]

        w_kn = _logical_kn(w1_raw, kpack, n_pad, "W1 weight")
        s_kn = _logical_kn(w1_scale.view(torch.uint8), kg, n_pad, "W1 scale")

        # Chunked over experts to bound the repack transients (the GGUU
        # index_select + shuffle + fragment-alignment permute stage a few
        # tensors of the packed-W1 size each).
        e_chunk = 32
        w4_parts, sc_parts, sh_parts, sw_parts = [], [], [], []
        for e0 in range(0, E_, e_chunk):
            e1 = min(e0 + e_chunk, E_)
            w4_c, sc_c, sh_c, sw_c = repack_w13_pairwise_to_a8w4fp4(
                w_kn[e0:e1], s_kn[e0:e1], inter_true
            )
            w4_parts.append(w4_c)
            sc_parts.append(sc_c)
            sh_parts.append(sh_c)
            sw_parts.append(sw_c)
        w4 = torch.cat(w4_parts, dim=0)
        sc = torch.cat(sc_parts, dim=0)
        sh = torch.cat(sh_parts, dim=0)
        sw = torch.cat(sw_parts, dim=0)

        self._flydsl_g1fp4_verify(
            w_kn, s_kn, w4, sc, sh, sw, inter_true, hidden
        )

        self._flydsl_g1_w4 = w4
        self._flydsl_g1_sc = sc
        self._flydsl_g1_sh = sh
        self._flydsl_g1_sw = sw
        self._flydsl_g1_inter = inter_true
        # Drop the packed mxfp4 W1 and its production scale storage: with the
        # flag on the triton GEMM1 path is never dispatched.
        dev = w4.device
        layer.w13_weight.storage.data = torch.empty(
            0, dtype=torch.uint8, device=dev
        )
        w1_scale_holder = weight_mx_scale(self.quant_config.w1_precision)
        w1_scale_holder.storage.data = torch.empty(
            0, dtype=torch.uint8, device=dev
        )
        logger.info(
            "VLLM_FLYDSL_A8W4FP4_G1: repacked W1 in place (inter=%d, K=%d):"
            " w4 %s sc %s bytes",
            inter_true,
            hidden,
            tuple(w4.shape),
            tuple(sc.shape),
        )

    @staticmethod
    def _flydsl_g1fp4_verify(w_kn, s_kn, w4, sc, sh, sw, inter_true, hidden,
                             n=8192):
        """Ground-truth the W1 repack against the production W1 storage.

        Every sampled (e, col, k): the v2 fragment-aligned byte at its
        computed address must equal the production packed byte at
        w_kn[e, k//2, row] (the same two e2m1 nibbles, layout-only move);
        the scale planes must satisfy sc[e, col, g] == s_kn[e, g, row],
        sh[e, col] == max_g s_kn[e, g, row], sw == 2^(sh - 127). A wrong
        mapping or orientation fails loudly at load (the iter-17 lesson),
        not as an accuracy regression an hour later.
        """
        dev = w4.device
        E_ = w4.shape[0]
        gen = torch.Generator(device=dev).manual_seed(0)
        e = torch.randint(E_, (n,), device=dev, generator=gen)
        col = torch.randint(2 * inter_true, (n,), device=dev, generator=gen)
        k = torch.randint(hidden, (n,), device=dev, generator=gen)
        gate = col < inter_true
        row = torch.where(gate, 2 * col, 2 * (col - inter_true) + 1)
        # production packed byte + nibble
        byte_p = w_kn[e, k // 2, row].to(torch.int32)
        nib_p = (byte_p >> (4 * (k & 1))) & 0xF
        # v2 fragment-aligned byte address:
        # (E, N0=col/16, K0=k/128, LD=(k%32)/8, NL=col%16, KL=(k/32)%4, b/2)
        n0, nl = col // 16, col % 16
        k128, kl = k // 128, (k // 32) % 4
        ld, bb = (k % 32) // 8, (k % 8) // 2
        addr = (
            ((((n0 * (hidden // 128) + k128) * 4 + ld) * 16 + nl) * 4 + kl)
            * 4 + bb
        )
        byte_v = w4.view(E_, -1)[e, addr].to(torch.int32)
        nib_v = (byte_v >> (4 * (k & 1))) & 0xF
        nib_diff = int((nib_v != nib_p).sum())
        sc_diff = int(
            (sc[e, col, k // 32] != s_kn[e, k // 32, row]).sum()
        )
        sh_ref = s_kn[e, :, row].amax(dim=1)
        sh_diff = int((sh[e, col] != sh_ref).sum())
        sw_ref = torch.exp2(sh[e, col].to(torch.float32) - 127.0)
        sw_diff = float((sw[e, col] - sw_ref).abs().max())
        logger.info(
            "VLLM_FLYDSL_A8W4FP4_G1 verify: nib mismatches %d, sc %d, sh %d,"
            " sw max diff %.3e over %d samples",
            nib_diff, sc_diff, sh_diff, sw_diff, n,
        )
        assert nib_diff == 0 and sc_diff == 0 and sh_diff == 0 and sw_diff == 0.0, (
            "a8w4-FP4W G1 repack does not reproduce the production mxfp4 W1"
            " layout (see counts above)"
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not rocm_aiter_ops.is_enabled():
            return False
        from vllm.platforms.rocm import on_gfx942, on_gfx950, on_gfx1250

        return on_gfx942() or on_gfx950() or on_gfx1250()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SWIGLUOAI, MoEActivation.SILU)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.DeepseekV4,
            RoutingMethodType.DeepSeekV3,
        ]

    @staticmethod
    def _supports_hash_routing() -> bool:
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
        routing_replay_out: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        routing_method = self.moe_config.routing_method
        if routing_method == RoutingMethodType.DeepseekV4:
            score_mode = "sqrtsoftplus"
        elif routing_method == RoutingMethodType.DeepSeekV3:
            # MiMo-V2.6: an ungrouped sigmoid router with a per-expert
            # correction bias. aiter's flat top-k implements this natively
            # (select on sigmoid(logits) + bias, return the unbiased sigmoid
            # scores).
            assert e_score_correction_bias is not None, (
                "DeepSeekV3 routing requires e_score_correction_bias"
            )
            score_mode = "sigmoid"
        else:
            score_mode = None
        return aiter_triton_kernel_w4a16_moe_forward(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            gating_output=router_logits,
            topk=self.topk,
            renormalize=self.renormalize,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=self.quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            unpadded_N_w1=None,
            unpadded_K_w1=None,
            unpadded_N_w2=None,
            unpadded_K_w2=None,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            score_mode=score_mode,
            input_ids=input_ids,
            hash_indices_table=hash_indices_table,
            flydsl_g1_w4=self._flydsl_g1_w4,
            flydsl_g1_sc=self._flydsl_g1_sc,
            flydsl_g1_sh=self._flydsl_g1_sh,
            flydsl_g1_sw=self._flydsl_g1_sw,
            flydsl_g1_inter=self._flydsl_g1_inter,
            flydsl_g2_w8=self._flydsl_g2_w8,
            flydsl_g2_sw=self._flydsl_g2_sw,
            flydsl_g2_k_true=self._flydsl_g2_k_true,
        )
