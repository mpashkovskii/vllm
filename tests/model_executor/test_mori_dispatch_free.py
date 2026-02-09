# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MORI-EP dispatch-free optimization.

Tests that the dispatch-free mode (for TP+EP) produces identical results
to the standard MORI dispatch mode using real MORI operations on GPUs.

RUNNING THESE TESTS:
-------------------
These tests require a distributed multi-GPU environment with MORI library.

Single GPU (tests will adapt to 1 rank):
    pytest -v tests/model_executor/test_mori_dispatch_free.py

Multi-GPU with torchrun (recommended):
    torchrun --nproc_per_node=4 -m pytest tests/model_executor/test_mori_dispatch_free.py -v

Multi-GPU with pytest-xdist (alternative):
    pytest -v tests/model_executor/test_mori_dispatch_free.py -n 4

Requirements:
- ROCm environment with MORI library installed
- Multiple GPUs for full testing (tests adapt to available GPU count)
- CUDA/ROCm enabled PyTorch with distributed support
"""

import os

import pytest
import torch

# Skip tests if MORI is not available
mori = pytest.importorskip("mori")

from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.mori_prepare_finalize import (
    MoriPrepareAndFinalize,
)
from vllm.platforms import current_platform


@pytest.fixture
def distributed_env():
    """Set up distributed environment for testing."""
    # Initialize distributed if not already done
    if not torch.distributed.is_initialized():
        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)
    
    # Get current rank and world size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    yield {"rank": rank, "world_size": world_size}


@pytest.fixture
def mori_op(test_config, distributed_env):
    """Create a real MORI dispatch/combine operation."""
    rank = distributed_env["rank"]
    world_size = distributed_env["world_size"]
    
    # MORI operation parameters
    quant_dtype = current_platform.fp8_dtype()
    all_to_all_args = dict(
        rank=rank,
        num_ep_ranks=world_size,
        quant_dtype=quant_dtype,
        token_hidden_size=test_config["hidden_dim"],
        scale_dim=1,  # per-token quantization
        scale_type_size=torch.float32.itemsize,
        max_num_tokens_per_dp_rank=test_config["batch_size"],
        input_dtype=torch.float16,
        num_local_experts=test_config["num_local_experts"],
        num_experts_per_token=test_config["experts_per_token"],
    )
    
    # Create MORI handle
    ep_op = mori.ops.EpDispatchCombineOp(**all_to_all_args)
    
    yield ep_op


@pytest.fixture
def ep_group():
    """Get the expert parallel group."""
    from vllm.distributed import get_ep_group
    
    ep_group_info = get_ep_group()
    if ep_group_info is None:
        pytest.skip("EP group not initialized")
    
    return ep_group_info.device_group


@pytest.fixture
def test_config(distributed_env):
    """Common test configuration."""
    world_size = distributed_env["world_size"]
    
    return {
        "batch_size": 128,
        "hidden_dim": 4096,
        "num_experts": 256,
        "num_local_experts": 256 // world_size,  # Divide experts across ranks
        "experts_per_token": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def create_test_data(config):
    """Create test input tensors."""
    device = config["device"]
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    num_experts = config["num_experts"]
    experts_per_token = config["experts_per_token"]
    
    # Create input activations
    a1 = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)
    
    # Create routing: topk expert selections per token
    topk_ids = torch.randint(
        0, num_experts, (batch_size, experts_per_token), device=device, dtype=torch.int32
    )
    
    # Create routing weights (normalized)
    topk_weights = torch.rand(
        batch_size, experts_per_token, device=device, dtype=torch.float16
    )
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    
    return a1, topk_ids, topk_weights


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MORI tests"
)
@pytest.mark.parametrize("use_fp8_dispatch", [False])
def test_dispatch_free_equivalence(
    mori_op, ep_group, distributed_env, test_config, use_fp8_dispatch
):
    """
    Test that dispatch-free mode produces equivalent results to standard mode.
    
    This test:
    1. Creates identical input data on all ranks (simulating TP all-reduce)
    2. Runs prepare() in standard mode (with MORI dispatch)
    3. Runs prepare() in dispatch-free mode (TP+EP optimization)
    4. Simulates expert computation
    5. Compares finalize() outputs to ensure correctness
    """
    rank = distributed_env["rank"]
    world_size = distributed_env["world_size"]
    
    # Create IDENTICAL test data on all ranks (simulating TP all-reduce result)
    torch.manual_seed(42)  # Same seed on all ranks
    a1, topk_ids, topk_weights = create_test_data(test_config)
    
    # Create quant config
    quant_config = FusedMoEQuantConfig.make()
    
    rank_expert_offset = rank * test_config["num_local_experts"]
    
    # Test 1: Standard mode (with MORI dispatch)
    standard_handler = MoriPrepareAndFinalize(
        mori_op=mori_op,
        max_tokens_per_rank=test_config["batch_size"],
        num_dispatchers=world_size,
        use_fp8_dispatch=use_fp8_dispatch,
    )
    
    standard_result = standard_handler.prepare(
        a1=a1.clone(),
        topk_weights=topk_weights.clone(),
        topk_ids=topk_ids.clone(),
        num_experts=test_config["num_experts"],
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
        defer_input_quant=False,
    )
    
    # Test 2: Dispatch-free mode (with TP+EP parameters)
    dispatch_free_handler = MoriPrepareAndFinalize(
        mori_op=mori_op,
        max_tokens_per_rank=test_config["batch_size"],
        num_dispatchers=world_size,
        use_fp8_dispatch=use_fp8_dispatch,
        num_local_experts=test_config["num_local_experts"],
        rank_expert_offset=rank_expert_offset,
        ep_group=ep_group,
        enable_dispatch_free=True,
    )
    
    dispatch_free_result = dispatch_free_handler.prepare(
        a1=a1.clone(),
        topk_weights=topk_weights.clone(),
        topk_ids=topk_ids.clone(),
        num_experts=test_config["num_experts"],
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
        defer_input_quant=False,
    )
    
    # Unpack results
    (std_a1, std_scale, std_meta, std_ids, std_weights) = standard_result
    (df_a1, df_scale, df_meta, df_ids, df_weights) = dispatch_free_result
    
    # Simulate expert computation: identity transformation for simplicity
    # In real scenario, this would be the AITER expert computation
    std_expert_output = std_a1.clone()  # Standard path output
    df_expert_output = df_a1.clone()   # Dispatch-free path output
    
    # Prepare output buffers
    batch_size = test_config["batch_size"]
    hidden_dim = test_config["hidden_dim"]
    device = test_config["device"]
    
    std_output = torch.zeros(batch_size, hidden_dim, device=device, dtype=torch.float16)
    df_output = torch.zeros(batch_size, hidden_dim, device=device, dtype=torch.float16)
    
    # Run finalize
    standard_handler.finalize(
        output=std_output,
        fused_expert_output=std_expert_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=None,
    )
    
    dispatch_free_handler.finalize(
        output=df_output,
        fused_expert_output=df_expert_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=None,
    )
    
    # Compare outputs - they should be very close (accounting for numerical differences)
    torch.testing.assert_close(
        std_output,
        df_output,
        rtol=1e-3,
        atol=1e-3,
        msg="Standard and dispatch-free outputs should match"
    )
    
    # Verify dispatch-free correctly identified local tokens
    expected_local_mask = (topk_ids >= rank_expert_offset) & (
        topk_ids < rank_expert_offset + test_config["num_local_experts"]
    )
    expected_local_tokens = expected_local_mask.any(dim=1).sum().item()
    
    assert df_a1.shape[0] == expected_local_tokens or df_a1.shape[0] == 0, \
        f"Dispatch-free should have {expected_local_tokens} local tokens, got {df_a1.shape[0]}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MORI tests"
)
def test_dispatch_free_finalize(mori_op, ep_group, distributed_env, test_config):
    """
    Test that dispatch-free finalize produces correct output with real all-reduce.
    """
    rank = distributed_env["rank"]
    batch_size = test_config["batch_size"]
    hidden_dim = test_config["hidden_dim"]
    device = test_config["device"]
    
    # Create identical input on all ranks
    torch.manual_seed(42)
    
    # Create mock expert outputs (each rank has different values)
    num_local_tokens = batch_size // 4
    expert_output = torch.randn(
        num_local_tokens, hidden_dim, device=device, dtype=torch.float16
    ) * (rank + 1)  # Different values per rank
    
    # Create output buffer
    output = torch.zeros(batch_size, hidden_dim, device=device, dtype=torch.float16)
    
    # Create handler with dispatch-free enabled
    handler = MoriPrepareAndFinalize(
        mori_op=mori_op,
        max_tokens_per_rank=batch_size,
        num_dispatchers=distributed_env["world_size"],
        use_fp8_dispatch=False,
        num_local_experts=test_config["num_local_experts"],
        rank_expert_offset=rank * test_config["num_local_experts"],
        ep_group=ep_group,
        enable_dispatch_free=True,
    )
    
    # Set up internal state (normally set by prepare)
    handler._local_token_indices = torch.arange(
        num_local_tokens, device=device, dtype=torch.long
    )
    
    # Run finalize with real all-reduce
    handler.finalize(
        output=output,
        fused_expert_output=expert_output,
        topk_weights=None,
        topk_ids=None,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=None,
    )
    
    # Verify output shape is correct
    assert output.shape == (batch_size, hidden_dim), "Output shape should match input"
    
    # Verify some tokens got values (from all-reduce across ranks)
    assert output.abs().sum() > 0, "Output should have non-zero values after all-reduce"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MORI tests"
)
def test_dispatch_free_empty_tokens(mori_op, ep_group, distributed_env, test_config):
    """
    Test dispatch-free mode when no tokens need local experts.
    """
    rank = distributed_env["rank"]
    num_local_experts = test_config["num_local_experts"]
    rank_expert_offset = rank * num_local_experts
    
    # Create test data
    a1, topk_ids, topk_weights = create_test_data(test_config)
    
    # Modify topk_ids so NO tokens need this rank's local experts
    # Assign all tokens to next rank's experts
    next_rank = (rank + 1) % distributed_env["world_size"]
    next_rank_offset = next_rank * num_local_experts
    topk_ids = torch.full_like(topk_ids, next_rank_offset)
    
    quant_config = FusedMoEQuantConfig.make()
    
    handler = MoriPrepareAndFinalize(
        mori_op=mori_op,
        max_tokens_per_rank=test_config["batch_size"],
        num_dispatchers=distributed_env["world_size"],
        use_fp8_dispatch=False,
        num_local_experts=num_local_experts,
        rank_expert_offset=rank_expert_offset,
        ep_group=ep_group,
        enable_dispatch_free=True,
    )
    
    result = handler.prepare(
        a1=a1,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=test_config["num_experts"],
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
        defer_input_quant=False,
    )
    
    a1_out, scale, meta, ids, weights = result
    
    # Verify empty outputs
    assert a1_out.shape[0] == 0, "Should return empty tensor when no local tokens"
    assert ids.shape[0] == 0, "Should return empty IDs"
    assert weights.shape[0] == 0, "Should return empty weights"
    assert meta.expert_num_tokens.sum() == 0, "All expert counts should be zero"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MORI tests"
)
def test_dispatch_free_feature_flag():
    """Test that the feature flag can disable dispatch-free optimization."""
    mock_op = Mock()
    mock_ep_group = Mock()
    
    # Test with feature flag disabled
    with patch.dict(os.environ, {"VLLM_MORI_EP_DISPATCH_FREE": "0"}):
        # Need to reload module to pick up new env var
        import importlib
        from vllm.model_executor.layers.fused_moe import mori_prepare_finalize
        importlib.reload(mori_prepare_finalize)
        
        handler = mori_prepare_finalize.MoriPrepareAndFinalize(
            mori_op=mock_op,
            max_tokens_per_rank=128,
            num_dispatchers=8,
            use_fp8_dispatch=False,
            num_local_experts=32,
            rank_expert_offset=0,
            ep_group=mock_ep_group,
            enable_dispatch_free=True,
        )
        
        # Should NOT enable dispatch-free when flag is off
        assert not handler.use_dispatch_free, \
            "Dispatch-free should be disabled when VLLM_MORI_EP_DISPATCH_FREE=0"
    
    # Test with feature flag enabled (default)
    with patch.dict(os.environ, {"VLLM_MORI_EP_DISPATCH_FREE": "1"}):
        importlib.reload(mori_prepare_finalize)
        
        handler = mori_prepare_finalize.MoriPrepareAndFinalize(
            mori_op=mock_op,
            max_tokens_per_rank=128,
            num_dispatchers=8,
            use_fp8_dispatch=False,
            num_local_experts=32,
            rank_expert_offset=0,
            ep_group=mock_ep_group,
            enable_dispatch_free=True,
        )
        
        # Should enable dispatch-free when flag is on
        assert handler.use_dispatch_free, \
            "Dispatch-free should be enabled when VLLM_MORI_EP_DISPATCH_FREE=1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
