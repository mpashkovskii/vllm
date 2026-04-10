#!/usr/bin/env bash
# run_pd_demo_2node.sh — Two-node MoRIIO PD-disaggregation demo
#
# Run the SAME script on BOTH nodes, setting IS_PREFILL to distinguish roles:
#
#   Node 1 — prefill instance + vllm-router:
#     IS_PREFILL=1 PREFILL_IP=<node1-ip> DECODE_IP=<node2-ip> \
#       ./examples/online_serving/disaggregated_serving/moriio_pd_demo/run_pd_demo_2node.sh
#
#   Node 2 — decode instance:
#     IS_PREFILL=0 PREFILL_IP=<node1-ip> DECODE_IP=<node2-ip> \
#       ./examples/online_serving/disaggregated_serving/moriio_pd_demo/run_pd_demo_2node.sh
#
# Prerequisites
#   • 8 ROCm GPUs on each node
#   • Docker image pulled: ghcr.io/simondanielsson/vllm-rocm-moriio:dev-0410-1542
#   • Router image pulled: ghcr.io/simondanielsson/vllm-router:dev (or :dev-streaming)
#   • RDMA / InfiniBand devices visible at /dev/infiniband on each node
#   • Ports PREFILL_PORT, DECODE_PORT, ROUTER_PORT, PROXY_PING_PORT,
#     HANDSHAKE_PORT, NOTIFY_PORT open between the two nodes

set -euo pipefail

# ── Required env vars ─────────────────────────────────────────────────────────
IS_PREFILL="${IS_PREFILL:-}"
PREFILL_IP="${PREFILL_IP:-}"
DECODE_IP="${DECODE_IP:-}"

if [[ -z "${IS_PREFILL}" || -z "${PREFILL_IP}" || -z "${DECODE_IP}" ]]; then
    echo "ERROR: IS_PREFILL, PREFILL_IP, and DECODE_IP must all be set." >&2
    echo "" >&2
    echo "  Node 1 (prefill + router):" >&2
    echo "    IS_PREFILL=1 PREFILL_IP=<node1-ip> DECODE_IP=<node2-ip> $0" >&2
    echo "" >&2
    echo "  Node 2 (decode):" >&2
    echo "    IS_PREFILL=0 PREFILL_IP=<node1-ip> DECODE_IP=<node2-ip> $0" >&2
    exit 1
fi

# ── Configuration ─────────────────────────────────────────────────────────────
# DeepSeek-R1-0528 is a 671B MoE model; --load-format dummy skips weight
# loading so the script can be used for integration testing without the
# full model checkpoint.  Set MODEL to the real HF id when running with
# actual weights.
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"

PREFILL_PORT="${PREFILL_PORT:-8100}"      # HTTP port for the prefill vLLM instance
DECODE_PORT="${DECODE_PORT:-8200}"        # HTTP port for the decode vLLM instance
ROUTER_PORT="${ROUTER_PORT:-8080}"        # HTTP port for vllm-router (prefill node only)
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}" # ZMQ service-discovery port (router ↔ vLLM)

# MoRIIO internal ports — nodes are separate machines so no port conflicts;
# both instances can use the same numbers.
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"  # MoRIIO engine handshake
NOTIFY_PORT="${NOTIFY_PORT:-61005}"       # Prefill↔decode stage synchronisation

VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/simondanielsson/vllm-rocm-moriio:dev-0410-1542}"
# Basic router (smoke-test only — no streaming support)
ROUTER_IMAGE="${ROUTER_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev}"
# Streaming-capable router (required for USE_BENCH=1 and USE_GSM8K=1)
ROUTER_STREAMING_IMAGE="${ROUTER_STREAMING_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev-streaming}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-${HOME}/moriio-logs}"
SHM_SIZE="${SHM_SIZE:-256G}"

USE_BENCH="${USE_BENCH:-0}"    # Set to 1 to run the perf benchmark (prefill node only)
USE_GSM8K="${USE_GSM8K:-0}"   # Set to 1 to run GSM8K accuracy eval (prefill node only)
KEEP_ALIVE="${KEEP_ALIVE:-0}" # Set to 1 to leave containers running after the script exits

BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-16}"
BENCH_NUM_WARMUPS=$((BENCH_MAX_CONCURRENCY * 2))
BENCH_NUM_PROMPTS=$((BENCH_MAX_CONCURRENCY * 10))

mkdir -p "${LOG_DIR}"

# ── vLLM serve flags shared between prefill and decode ───────────────────────
VLLM_SERVE_ARGS=(
    --tensor-parallel-size 8
    --kv-cache-dtype fp8
    --load-format dummy
    --gpu-memory-utilization 0.7
    --max-num-batched-tokens 32768
    --max-model-len 16384
    --enable-expert-parallel
    --trust-remote-code
)

# ── KV-transfer configs ───────────────────────────────────────────────────────
# proxy_ip      : IP of the node running vllm-router (always the prefill node)
# proxy_ping_port : ZMQ port the router listens on for instance registration
# http_port     : this instance's own HTTP port (embedded in zmq_address by router)
# handshake_port / notify_port : MoRIIO RDMA coordination ports on this node
PREFILL_KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "proxy_ip": "${PREFILL_IP}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${PREFILL_PORT}",
    "handshake_port": "${HANDSHAKE_PORT}",
    "notify_port": "${NOTIFY_PORT}"
  }
}
EOF
)

DECODE_KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "proxy_ip": "${PREFILL_IP}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${DECODE_PORT}",
    "handshake_port": "${HANDSHAKE_PORT}",
    "notify_port": "${NOTIFY_PORT}"
  }
}
EOF
)

# ── Common docker run flags ───────────────────────────────────────────────────
VLLM_COMMON_ARGS=(
    --init
    --network host
    --ipc host
    --privileged
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --ulimit memlock=-1
    --ulimit stack=67108864
    --shm-size "${SHM_SIZE}"
    --group-add video
    --group-add render
    --device /dev/kfd
    --device /dev/dri
    --device /dev/infiniband
    -v /sys:/sys
    -v "${HF_HOME}:/root/.cache/huggingface"
    -e HF_HOME=/root/.cache/huggingface
    -e HF_HUB_ENABLE_HF_TRANSFER=0
    -e VLLM_MORIIO_CONNECTOR_READ_MODE=1
    -e NCCL_MIN_NCHANNELS=112
    -e VLLM_USE_V1=1
    -e VLLM_ROCM_USE_AITER=1
    -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    -e VLLM_ROCM_USE_AITER_RMSNORM=1
    -e VLLM_USE_AITER_TRITON_SILU_MUL=0
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600
    -e VLLM_SERVER_DEV_MODE=1
)

# ── Helpers ───────────────────────────────────────────────────────────────────
wait_for_health() {
    local name="$1"
    local url="$2"
    local max_wait="${3:-600}"
    local interval=10
    local elapsed=0

    echo "Waiting for ${name} (${url}) to become healthy..."
    while true; do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo "  ${name} is healthy."
            return 0
        fi
        sleep "${interval}"
        elapsed=$((elapsed + interval))
        if [[ "${elapsed}" -ge "${max_wait}" ]]; then
            echo "ERROR: ${name} did not become healthy after ${max_wait}s" >&2
            exit 1
        fi
        echo "  Still waiting for ${name} (${elapsed}s / ${max_wait}s)..."
    done
}

# ── Cleanup trap ──────────────────────────────────────────────────────────────
_cleanup() {
    if [[ "${KEEP_ALIVE}" == "1" ]]; then
        echo ""
        echo "KEEP_ALIVE=1 — containers left running."
        if [[ "${IS_PREFILL}" == "1" ]]; then
            echo "  To tear down: docker rm -f moriio-prefill moriio-router"
        else
            echo "  To tear down: docker rm -f moriio-decode"
        fi
        return
    fi
    echo ""
    echo ">>> Shutting down containers..."
    if [[ "${IS_PREFILL}" == "1" ]]; then
        docker rm -f moriio-prefill moriio-router 2>/dev/null || true
    else
        docker rm -f moriio-decode 2>/dev/null || true
    fi
    echo "Done."
}
trap _cleanup EXIT

# ── Remove stale containers ───────────────────────────────────────────────────
if [[ "${IS_PREFILL}" == "1" ]]; then
    _stale_containers=(moriio-prefill moriio-router)
else
    _stale_containers=(moriio-decode)
fi
for cname in "${_stale_containers[@]}"; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${cname}$"; then
        echo "Removing existing container: ${cname}"
        docker rm -f "${cname}"
    fi
done

# ── Print summary ─────────────────────────────────────────────────────────────
_role="$([ "${IS_PREFILL}" == "1" ] && echo "prefill + router" || echo "decode")"
echo "=== MoRIIO PD disaggregation demo (2-node) ==="
echo "  Model       : ${MODEL}"
echo "  This node   : ${_role}"
echo "  Prefill     : http://${PREFILL_IP}:${PREFILL_PORT}"
echo "  Decode      : http://${DECODE_IP}:${DECODE_PORT}"
echo "  Router      : http://${PREFILL_IP}:${ROUTER_PORT}  (prefill node)"
echo "  Discovery   : ${PREFILL_IP}:${PROXY_PING_PORT}"
echo "  MoRIIO ports: handshake=${HANDSHAKE_PORT}  notify=${NOTIFY_PORT}"
echo "  Log dir     : ${LOG_DIR}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# PREFILL NODE
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "${IS_PREFILL}" == "1" ]]; then

echo ">>> Starting prefill instance (port ${PREFILL_PORT})..."
docker run -d \
    --name moriio-prefill \
    "${VLLM_COMMON_ARGS[@]}" \
    "${VLLM_IMAGE}" \
    vllm serve "${MODEL}" \
        --port "${PREFILL_PORT}" \
        "${VLLM_SERVE_ARGS[@]}" \
        --kv-transfer-config "${PREFILL_KV_CONFIG}"

docker logs -f moriio-prefill 2>&1 | tee "${LOG_DIR}/prefill.log" &

wait_for_health "moriio-prefill" "http://localhost:${PREFILL_PORT}/health" 600

# ── Launch vllm-router ────────────────────────────────────────────────────────
if [[ "${USE_BENCH}" == "1" || "${USE_GSM8K}" == "1" ]]; then
    _ACTIVE_ROUTER_IMAGE="${ROUTER_STREAMING_IMAGE}"
else
    _ACTIVE_ROUTER_IMAGE="${ROUTER_IMAGE}"
fi

echo ""
echo ">>> Starting vllm-router (port ${ROUTER_PORT}, discovery port ${PROXY_PING_PORT})..."
echo "    Image: ${_ACTIVE_ROUTER_IMAGE}"
docker run -d \
    --name moriio-router \
    --network host \
    "${_ACTIVE_ROUTER_IMAGE}" \
    vllm-router \
        --vllm-pd-disaggregation \
        --vllm-discovery-address "0.0.0.0:${PROXY_PING_PORT}" \
        --port "${ROUTER_PORT}" \
        --host 0.0.0.0 \
        --policy consistent_hash \
        --prefill-policy consistent_hash \
        --decode-policy consistent_hash \
        --log-level info

docker logs -f moriio-router 2>&1 | tee "${LOG_DIR}/router.log" &

# ── Wait for the decode node to be healthy (cross-node health check) ──────────
echo ""
echo ">>> Waiting for decode node (${DECODE_IP}:${DECODE_PORT}) to become healthy..."
echo "    Start the decode node now if you haven't already."
wait_for_health "moriio-decode (remote)" "http://${DECODE_IP}:${DECODE_PORT}/health" 1200

# Brief pause for the decode instance to register with the router via ZMQ.
echo "Waiting 10s for decode instance to register with the router..."
sleep 10

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== All services ready ==="
echo "  Prefill  : http://localhost:${PREFILL_PORT}"
echo "  Decode   : http://${DECODE_IP}:${DECODE_PORT}"
echo "  Router   : http://localhost:${ROUTER_PORT}  ← send requests here"
echo ""
echo "To follow logs (written to ${LOG_DIR}):"
echo "  tail -f ${LOG_DIR}/prefill.log"
echo "  tail -f ${LOG_DIR}/router.log"
echo ""
echo "Containers will be shut down automatically when this script exits."
echo "(Set KEEP_ALIVE=1 to leave them running.)"

# ── Smoke test / benchmark ────────────────────────────────────────────────────
if [[ "${USE_GSM8K}" == "1" ]]; then

    # Note: only a single-phase eval against vllm-router is run here.
    # The toy proxy comparison requires restarting the decode instance on the
    # remote node, which is not supported in 2-node mode.
    GSM8K_LOG="${LOG_DIR}/gsm8k_results.log"
    GSM8K_JSON="${LOG_DIR}/gsm8k_results.json"
    _out_dir="/tmp/lm_eval_out"

    echo ""
    echo ">>> Running GSM8K accuracy evaluation (lm_eval) through vllm-router..."
    {
        echo "======================================================"
        echo "  GSM8K evaluation (lm_eval) via MoRIIO PD-disaggregation (2-node)"
        echo "  Model : ${MODEL}"
        echo "  Date  : $(date)"
        echo "======================================================"
    } | tee "${GSM8K_LOG}"

    docker exec moriio-prefill bash -c \
        "pip install --quiet 'lm_eval[api]' && \
         rm -rf ${_out_dir} && \
         lm_eval \
             --model local-completions \
             --model_args model=${MODEL},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/completions,tokenized_requests=False,trust_remote_code=True \
             --tasks gsm8k \
             --output_path ${_out_dir}" \
        2>&1 | tee -a "${GSM8K_LOG}"

    _remote_json=$(docker exec moriio-prefill \
        find "${_out_dir}" -name "results.json" 2>/dev/null | head -1)
    if [[ -n "${_remote_json}" ]]; then
        docker cp "moriio-prefill:${_remote_json}" "${GSM8K_JSON}" 2>/dev/null || true
    else
        echo "WARNING: lm_eval results.json not found in ${_out_dir}" >&2
    fi

    echo ""
    echo "=== GSM8K evaluation complete ==="
    echo "  Log  : ${GSM8K_LOG}"
    echo "  JSON : ${GSM8K_JSON}"

elif [[ "${USE_BENCH}" == "1" ]]; then

    # Note: Phase 2 (toy proxy comparison) is not supported in 2-node mode
    # because it requires restarting the decode instance on the remote node.
    BENCH_LOG="${LOG_DIR}/benchmark_results.log"

    echo ""
    echo ">>> Benchmarking through vllm-router..."
    {
        echo "======================================================"
        echo "  Router: vllm-router (2-node)"
        echo "  Model : ${MODEL}"
        echo "  Date  : $(date)"
        echo "======================================================"
    } | tee "${BENCH_LOG}"

    docker exec moriio-prefill \
        vllm bench serve \
            --base-url "http://localhost:${ROUTER_PORT}" \
            --backend vllm \
            --model "${MODEL}" \
            --dataset-name random \
            --random-input-len 1000 \
            --random-output-len 1000 \
            --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
            --num-warmups "${BENCH_NUM_WARMUPS}" \
            --num-prompts "${BENCH_NUM_PROMPTS}" \
            --ready_check_timeout_sec 3000 \
            --seed 1234 \
        2>&1 | tee -a "${BENCH_LOG}"

    echo ""
    echo "=== Benchmark complete ==="
    echo "  Results: ${BENCH_LOG}"

else

    echo ""
    echo ">>> Smoke test: sending a completion request through vllm-router..."
    curl -s "http://localhost:${ROUTER_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"${MODEL}\",
          \"prompt\": \"San Francisco is a\",
          \"max_tokens\": 64,
          \"temperature\": 0
        }" | python3 -m json.tool
    echo ""
    echo "(Set USE_BENCH=1 for a perf benchmark, or USE_GSM8K=1 for accuracy eval.)"

fi

# ═══════════════════════════════════════════════════════════════════════════════
# DECODE NODE
# ═══════════════════════════════════════════════════════════════════════════════
else

echo ">>> Starting decode instance (port ${DECODE_PORT})..."
docker run -d \
    --name moriio-decode \
    "${VLLM_COMMON_ARGS[@]}" \
    "${VLLM_IMAGE}" \
    vllm serve "${MODEL}" \
        --port "${DECODE_PORT}" \
        "${VLLM_SERVE_ARGS[@]}" \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-transfer-config "${DECODE_KV_CONFIG}"

docker logs -f moriio-decode 2>&1 | tee "${LOG_DIR}/decode.log" &

wait_for_health "moriio-decode" "http://localhost:${DECODE_PORT}/health" 600

echo ""
echo "=== Decode instance is healthy and registered with router on ${PREFILL_IP} ==="
echo "  Decode : http://localhost:${DECODE_PORT}"
echo "  Router : http://${PREFILL_IP}:${ROUTER_PORT}"
echo ""
echo "To follow logs:"
echo "  tail -f ${LOG_DIR}/decode.log"
echo ""
echo "Containers will be shut down when this script exits (Ctrl+C)."
echo "(Set KEEP_ALIVE=1 to leave the container running after exit.)"
echo ""

# Block until the user stops the script; the cleanup trap handles teardown.
wait

fi
