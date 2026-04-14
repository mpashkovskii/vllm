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
#   • Docker image pulled: ghcr.io/simondanielsson/vllm-rocm-moriio:dev-0414-0859
#   • Router image pulled: ghcr.io/simondanielsson/vllm-router:dev (or :dev-streaming-cn-cjy)
#   • RDMA / InfiniBand devices visible at /dev/infiniband on each node
#   • Ports PREFILL_PORT, DECODE_PORT, ROUTER_PORT, PROXY_PING_PORT,
#     HANDSHAKE_PORT, NOTIFY_PORT, PHASE2_SIGNAL_PORT open between the two nodes
#
# Phase 2 coordination (USE_BENCH=1 or USE_GSM8K=1)
#   The prefill node starts a tiny HTTP signal server on PHASE2_SIGNAL_PORT
#   after Phase 1 completes.  The decode node polls that port; once it gets a
#   response it tears down its container and restarts so the new instance
#   registers with the toy proxy instead of the router.  The prefill node waits
#   for the decode to become healthy again before running Phase 2.

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

# Port used by the prefill node to signal the decode node to restart for Phase 2.
# The prefill node starts a simple HTTP server on this port; the decode node polls it.
PHASE2_SIGNAL_PORT="${PHASE2_SIGNAL_PORT:-19876}"

VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/simondanielsson/vllm-rocm-moriio:dev-0414-0859}"
# Basic router (smoke-test only — no streaming support)
ROUTER_IMAGE="${ROUTER_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev}"
# Streaming-capable router (required for USE_BENCH=1 and USE_GSM8K=1)
ROUTER_STREAMING_IMAGE="${ROUTER_STREAMING_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev-streaming-cn-cjy}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-${HOME}/moriio-logs}"
SHM_SIZE="${SHM_SIZE:-256G}"

USE_BENCH="${USE_BENCH:-0}"    # Set to 1 to run the perf benchmark (prefill node only)
USE_GSM8K="${USE_GSM8K:-0}"   # Set to 1 to run GSM8K accuracy eval (prefill node only)
KEEP_ALIVE="${KEEP_ALIVE:-0}" # Set to 1 to leave containers running after the script exits

BENCH_NUM_PROMPTS_FACTOR=${BENCH_NUM_PROMPTS_FACTOR:-10}
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-16}"
BENCH_NUM_WARMUPS=$((BENCH_MAX_CONCURRENCY * 2))
BENCH_NUM_PROMPTS=$((BENCH_MAX_CONCURRENCY * $BENCH_NUM_PROMPTS_FACTOR))

# HTTP port the toy proxy listens on (hardcoded in the image).
TOY_PROXY_HTTP_PORT=10001
# Path to the toy proxy script inside the vLLM image.
TOY_PROXY_CONTAINER_PATH="/app/vllm/examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py"

mkdir -p "${LOG_DIR}"

# ── Toy-proxy patch: fix non-streaming responses ──────────────────────────────
# Makes handle_request() return a proper JSON body when the client did not
# request streaming (needed by lm_eval and any non-streaming client).
TOY_PROXY_PATCH_SCRIPT="$(mktemp /tmp/patch_toy_proxy.XXXXXX.py)"
cat > "${TOY_PROXY_PATCH_SCRIPT}" << 'PYEOF'
import pathlib

TARGET = pathlib.Path(
    "/app/vllm/examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py"
)
src = TARGET.read_text()

OLD = (
    "        session, decode_response = await decode_request_task\n"
    "        stream_generator = stream_decode_response(session, decode_response, request_id)\n"
    "        response = await make_response(stream_generator)\n"
    "        return response"
)
NEW = (
    "        session, decode_response = await decode_request_task\n"
    "        if req_data.get(\"stream\", False):\n"
    "            stream_generator = stream_decode_response(\n"
    "                session, decode_response, request_id\n"
    "            )\n"
    "            response = await make_response(stream_generator)\n"
    "            return response\n"
    "        else:\n"
    "            try:\n"
    "                body = await decode_response.read()\n"
    "                content_type = decode_response.headers.get(\n"
    "                    \"Content-Type\", \"application/json\"\n"
    "                )\n"
    "            finally:\n"
    "                await session.close()\n"
    "            response = await make_response(body, decode_response.status)\n"
    "            response.headers[\"Content-Type\"] = content_type\n"
    "            return response"
)

if OLD in src:
    TARGET.write_text(src.replace(OLD, NEW, 1))
    print("toy proxy patch: non-streaming fix applied.")
else:
    print("toy proxy patch: target not found — skipping (already patched?).")
PYEOF

# ── vLLM serve flags shared between prefill and decode ───────────────────────
VLLM_SERVE_ARGS=(
    --tensor-parallel-size 8
    --kv-cache-dtype fp8
    --gpu-memory-utilization 0.7
    --max-num-batched-tokens 32768
    --max-model-len 16384
    --trust-remote-code
    --no-enable-prefix-caching
    --block-size 1
)

# ── Role-specific vLLM serve flags ───────────────────────────────────────────
PREFILL_EXTRA_ARGS=(
    --enforce-eager
)

DECODE_EXTRA_ARGS=(
    --enable-expert-parallel
    --all2all-backend mori
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
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
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600
    -e VLLM_SERVER_DEV_MODE=1
    -e VLLM_ROCM_USE_AITER=1
    -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    -e VLLM_ROCM_USE_AITER_RMSNORM=1
    -e VLLM_USE_AITER_TRITON_SILU_MUL=0
    
)

# ── Helpers ───────────────────────────────────────────────────────────────────
wait_for_health() {
    local name="$1"
    local url="$2"
    local max_wait="${3:-1800}"
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

# Poll docker logs until both P and D have registered with the toy proxy.
wait_for_toy_proxy_registrations() {
    local max_wait=120 interval=3 elapsed=0
    echo "Waiting for prefill and decode to register with toy proxy..."
    while true; do
        local p d
        p=$(docker logs moriio-toy-proxy 2>&1 | grep -c "Registered Prefill" || true)
        d=$(docker logs moriio-toy-proxy 2>&1 | grep -c "Registered Decode"  || true)
        if [[ "${p:-0}" -ge 1 && "${d:-0}" -ge 1 ]]; then
            echo "  Both instances registered (prefill=${p} decode=${d})."
            return 0
        fi
        sleep "${interval}"
        elapsed=$((elapsed + interval))
        if [[ "${elapsed}" -ge "${max_wait}" ]]; then
            echo "WARNING: timed out waiting for toy proxy registrations after ${max_wait}s" >&2
            return 0
        fi
        echo "  Still waiting (${elapsed}s / ${max_wait}s) — prefill=${p:-0} decode=${d:-0}..."
    done
}

# Stop router+prefill, launch toy proxy, restart prefill, wait for decode Phase 2.
# Called by both USE_BENCH and USE_GSM8K after Phase 1 completes.
run_phase2_switchover() {
    # Signal the decode node to restart by serving a one-line HTTP response on
    # PHASE2_SIGNAL_PORT.  We use a Python one-liner so the response is proper
    # HTTP (nc -l alone sends no headers and some curl versions reject it).
    echo ""
    echo ">>> Phase 2: signalling decode node to restart (port ${PHASE2_SIGNAL_PORT})..."
    echo "    The decode node is polling http://${PREFILL_IP}:${PHASE2_SIGNAL_PORT}/phase2"
    # Serve exactly one request then exit (Python http.server handles one GET then we kill it).
    python3 -c "
import http.server, socketserver, threading, time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'phase2\n')
        # Schedule shutdown after we've sent the response
        threading.Thread(target=self.server.shutdown, daemon=True).start()
    def log_message(self, *a): pass

with socketserver.TCPServer(('0.0.0.0', ${PHASE2_SIGNAL_PORT}), Handler) as srv:
    srv.serve_forever()
" &
    _SIGNAL_PID=$!

    # Now tear down the router and this node's prefill container.
    echo ">>> Stopping vllm-router and prefill (Phase 1)..."
    docker rm -f moriio-router moriio-prefill
    # Wait for GPU memory and host-network TCP sockets (TIME_WAIT) to be released
    # before starting the Phase 2 prefill container.
    echo "Waiting 30s for GPU/network resources to be released..."
    sleep 30

    # Start toy proxy (HTTP :${TOY_PROXY_HTTP_PORT}, ZMQ :${PROXY_PING_PORT})
    echo ">>> Starting toy proxy container (HTTP :${TOY_PROXY_HTTP_PORT}, ZMQ :${PROXY_PING_PORT})..."
    docker run -d \
        --name moriio-toy-proxy \
        --network host \
        -v "${TOY_PROXY_PATCH_SCRIPT}:/tmp/patch_toy_proxy.py:ro" \
        "${VLLM_IMAGE}" \
        bash -c "pip install --quiet --ignore-installed quart aiohttp msgpack && \
                 python3 /tmp/patch_toy_proxy.py && \
                 python3 -u ${TOY_PROXY_CONTAINER_PATH}"

    docker logs -f moriio-toy-proxy 2>&1 | tee "${LOG_DIR}/toy_proxy.log" &

    # Wait for toy proxy HTTP port to open before starting vLLM.
    echo "Waiting for toy proxy HTTP port ${TOY_PROXY_HTTP_PORT} to open..."
    _tp_wait=0
    until curl -sf "http://localhost:${TOY_PROXY_HTTP_PORT}/" >/dev/null 2>&1 \
       || curl -sf "http://localhost:${TOY_PROXY_HTTP_PORT}/v1/completions" >/dev/null 2>&1 \
       || nc -z 127.0.0.1 "${TOY_PROXY_HTTP_PORT}" 2>/dev/null; do
        sleep 2
        _tp_wait=$((_tp_wait + 2))
        if [[ "${_tp_wait}" -ge 60 ]]; then
            echo "WARNING: toy proxy did not open port ${TOY_PROXY_HTTP_PORT} after 60s" >&2
            docker logs moriio-toy-proxy 2>&1 | tail -20 >&2
            break
        fi
    done
    echo "  Toy proxy is up."

    # Restart the prefill container so it registers with the toy proxy.
    echo ">>> Restarting prefill instance (port ${PREFILL_PORT})..."
    docker run -d \
        --name moriio-prefill \
        "${VLLM_COMMON_ARGS[@]}" \
        "${VLLM_IMAGE}" \
        vllm serve "${MODEL}" \
            --port "${PREFILL_PORT}" \
            "${VLLM_SERVE_ARGS[@]}" \
            "${PREFILL_EXTRA_ARGS[@]}" \
            --kv-transfer-config "${PREFILL_KV_CONFIG}"

    docker logs -f moriio-prefill 2>&1 | tee "${LOG_DIR}/prefill_phase2.log" &

    wait_for_health "moriio-prefill" "http://localhost:${PREFILL_PORT}/health" 1800

    # Kill the signal server now that decode has been notified (it may already be gone).
    kill "${_SIGNAL_PID}" 2>/dev/null || true

    # Wait for the remote decode to come back up (it restarts itself upon seeing the signal).
    echo ""
    echo ">>> Waiting for decode node (${DECODE_IP}:${DECODE_PORT}) to come back up..."
    wait_for_health "moriio-decode (remote, Phase 2)" "http://${DECODE_IP}:${DECODE_PORT}/health" 1200

    echo "Waiting 10s for decode to register with toy proxy..."
    sleep 10

    wait_for_toy_proxy_registrations
}

# ── Cleanup trap ──────────────────────────────────────────────────────────────
_cleanup() {
    if [[ "${KEEP_ALIVE}" == "1" ]]; then
        echo ""
        echo "KEEP_ALIVE=1 — containers left running."
        if [[ "${IS_PREFILL}" == "1" ]]; then
            echo "  To tear down: docker rm -f moriio-prefill moriio-router moriio-toy-proxy"
        else
            echo "  To tear down: docker rm -f moriio-decode"
        fi
        return
    fi
    echo ""
    echo ">>> Shutting down containers..."
    if [[ "${IS_PREFILL}" == "1" ]]; then
        docker rm -f moriio-prefill moriio-router moriio-toy-proxy 2>/dev/null || true
    else
        docker rm -f moriio-decode 2>/dev/null || true
    fi
    echo "Done."
}
trap _cleanup EXIT

# ── Remove stale containers ───────────────────────────────────────────────────
if [[ "${IS_PREFILL}" == "1" ]]; then
    _stale_containers=(moriio-prefill moriio-router moriio-toy-proxy)
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
echo "  Phase2 sig  : ${PREFILL_IP}:${PHASE2_SIGNAL_PORT}  (USE_BENCH=1 or USE_GSM8K=1)"
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
        "${PREFILL_EXTRA_ARGS[@]}" \
        --kv-transfer-config "${PREFILL_KV_CONFIG}"

docker logs -f moriio-prefill 2>&1 | tee "${LOG_DIR}/prefill.log" &

wait_for_health "moriio-prefill" "http://localhost:${PREFILL_PORT}/health" 1800

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

    GSM8K_LOG="${LOG_DIR}/gsm8k_results.log"
    GSM8K_JSON_ROUTER="${LOG_DIR}/gsm8k_results_router.json"
    GSM8K_JSON_PROXY="${LOG_DIR}/gsm8k_results_proxy.json"
    _out_dir="/tmp/lm_eval_out"

    _run_lm_eval() {
        local base_url="$1"
        docker exec moriio-prefill bash -c \
            "pip install --quiet 'lm_eval[api]' && \
             rm -rf ${_out_dir} && \
             lm_eval \
                 --model local-completions \
                 --model_args model=${MODEL},base_url=${base_url}/v1/completions,tokenized_requests=False,trust_remote_code=True \
                 --tasks gsm8k \
                 --output_path ${_out_dir}" \
            2>&1 | tee -a "${GSM8K_LOG}"
    }

    _save_lm_eval_json() {
        local dest="$1"
        local _remote_json
        _remote_json=$(docker exec moriio-prefill \
            find "${_out_dir}" -name "results.json" 2>/dev/null | head -1)
        if [[ -n "${_remote_json}" ]]; then
            docker cp "moriio-prefill:${_remote_json}" "${dest}" 2>/dev/null || true
        else
            echo "WARNING: lm_eval results.json not found in ${_out_dir}" >&2
        fi
    }

    if [[ "${GSM8K_PHASE2_ONLY:-0}" != "1" ]]; then
        # ── Phase 1: GSM8K through vllm-router ───────────────────────────────────
        echo ""
        echo ">>> Phase 1: running GSM8K accuracy evaluation (lm_eval) through vllm-router..."
        {
            echo "======================================================"
            echo "  GSM8K evaluation (lm_eval) via MoRIIO PD-disaggregation (2-node)"
            echo "  Router: vllm-router"
            echo "  Model : ${MODEL}"
            echo "  Date  : $(date)"
            echo "======================================================"
        } | tee "${GSM8K_LOG}"

        _run_lm_eval "http://127.0.0.1:${ROUTER_PORT}"
        _save_lm_eval_json "${GSM8K_JSON_ROUTER}"

        # ── Phase 2: switch to toy proxy ──────────────────────────────────────────
        run_phase2_switchover
    else
        echo ""
        echo ">>> GSM8K_PHASE2_ONLY=1: skipping Phase 1 (vllm-router) and running switchover."
        {
            echo "======================================================"
            echo "  GSM8K evaluation (lm_eval) via MoRIIO PD-disaggregation (2-node)"
            echo "  (Phase 2 only run)"
            echo "  Date  : $(date)"
            echo "======================================================"
        } | tee "${GSM8K_LOG}"

        # The switchover must still happen: it stops the router + prefill (freeing
        # PROXY_PING_PORT so the toy proxy's ZMQ listener can bind), starts the toy
        # proxy, restarts prefill, and waits for decode to re-register.
        run_phase2_switchover
    fi

    # ── Phase 2: GSM8K through toy proxy ─────────────────────────────────────
    echo ""
    echo ">>> Phase 2: running GSM8K accuracy evaluation (lm_eval) through toy proxy..."
    {
        echo ""
        echo "======================================================"
        echo "  GSM8K evaluation (lm_eval) via MoRIIO PD-disaggregation (2-node)"
        echo "  Router: moriio_toy_proxy_server.py"
        echo "  Model : ${MODEL}"
        echo "  Date  : $(date)"
        echo "======================================================"
    } | tee -a "${GSM8K_LOG}"

    _run_lm_eval "http://127.0.0.1:${TOY_PROXY_HTTP_PORT}"
    _save_lm_eval_json "${GSM8K_JSON_PROXY}"

    echo ""
    echo "=== GSM8K evaluation complete ==="
    echo "  Log        : ${GSM8K_LOG}"
    if [[ "${GSM8K_PHASE2_ONLY:-0}" != "1" ]]; then
        echo "  Router JSON: ${GSM8K_JSON_ROUTER}"
    fi
    echo "  Proxy JSON : ${GSM8K_JSON_PROXY}"

elif [[ "${USE_BENCH}" == "1" ]]; then

    BENCH_LOG="${LOG_DIR}/benchmark_results.log"

    BENCH_ARGS=(
        --backend vllm
        --model "${MODEL}"
        --dataset-name random
        --random-input-len 1000
        --random-output-len 1000
        --max-concurrency "${BENCH_MAX_CONCURRENCY}"
        --num-warmups "${BENCH_NUM_WARMUPS}"
        --num-prompts "${BENCH_NUM_PROMPTS}"
        --ready_check_timeout_sec 3000
        --seed 1234
    )

    # ── Phase 1: benchmark through vllm-router ────────────────────────────────
    echo ""
    echo ">>> Phase 1: benchmarking through vllm-router..."
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
            "${BENCH_ARGS[@]}" 2>&1 | tee -a "${BENCH_LOG}"

    # ── Phase 2: switch to toy proxy and benchmark again ──────────────────────
    run_phase2_switchover

    echo ""
    echo ">>> Phase 2: benchmarking through toy proxy (HTTP port ${TOY_PROXY_HTTP_PORT})..."
    {
        echo ""
        echo "======================================================"
        echo "  Router: moriio_toy_proxy_server.py (2-node)"
        echo "  Model : ${MODEL}"
        echo "  Date  : $(date)"
        echo "======================================================"
    } | tee -a "${BENCH_LOG}"

    docker exec moriio-prefill \
        vllm bench serve \
            --base-url "http://localhost:${TOY_PROXY_HTTP_PORT}" \
            "${BENCH_ARGS[@]}" 2>&1 | tee -a "${BENCH_LOG}"

    echo ""
    echo "=== Both benchmark runs complete ==="
    echo "  Results      : ${BENCH_LOG}"
    echo "  Toy proxy log: ${LOG_DIR}/toy_proxy.log"

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

# ── Helper: start (or restart) the decode container ──────────────────────────
start_decode() {
    local log_suffix="${1:-}"
    docker run -d \
        --name moriio-decode \
        "${VLLM_COMMON_ARGS[@]}" \
        "${VLLM_IMAGE}" \
        vllm serve "${MODEL}" \
            --port "${DECODE_PORT}" \
            "${VLLM_SERVE_ARGS[@]}" \
            "${DECODE_EXTRA_ARGS[@]}" \
            --kv-transfer-config "${DECODE_KV_CONFIG}"

    docker logs -f moriio-decode 2>&1 | tee "${LOG_DIR}/decode${log_suffix}.log" &
}

echo ">>> Starting decode instance (port ${DECODE_PORT})..."
start_decode ""

wait_for_health "moriio-decode" "http://localhost:${DECODE_PORT}/health" 1800

echo ""
echo "=== Decode instance is healthy and registered with router on ${PREFILL_IP} ==="
echo "  Decode : http://localhost:${DECODE_PORT}"
echo "  Router : http://${PREFILL_IP}:${ROUTER_PORT}"
echo ""
echo "To follow logs:"
echo "  tail -f ${LOG_DIR}/decode.log"
echo ""

if [[ "${USE_BENCH}" == "1" || "${USE_GSM8K}" == "1" ]]; then
    # ── Phase 2: wait for the prefill node to signal a restart ───────────────
    echo ">>> USE_BENCH/USE_GSM8K: waiting for Phase 2 restart signal from prefill node"
    echo "    (polling http://${PREFILL_IP}:${PHASE2_SIGNAL_PORT}/phase2 ...)"
    _sig_wait=0
    until curl -sf --max-time 5 "http://${PREFILL_IP}:${PHASE2_SIGNAL_PORT}/phase2" >/dev/null 2>&1; do
        sleep 5
        _sig_wait=$((_sig_wait + 5))
        if [[ "${_sig_wait}" -ge 7200 ]]; then
            echo "WARNING: Phase 2 signal never arrived after ${_sig_wait}s — exiting." >&2
            exit 0
        fi
        if (( _sig_wait % 60 == 0 )); then
            echo "  Still waiting for Phase 2 signal (${_sig_wait}s)..."
        fi
    done

    echo ""
    echo ">>> Phase 2 signal received — restarting decode container to register with toy proxy..."
    docker rm -f moriio-decode 2>/dev/null || true

    # The kv_transfer_config proxy_ip still points at PREFILL_IP where the toy proxy runs.
    start_decode "_phase2"

    wait_for_health "moriio-decode (Phase 2)" "http://localhost:${DECODE_PORT}/health" 1800

    echo ""
    echo "=== Decode Phase 2 instance is healthy and registered with toy proxy ==="
    echo "  Decode : http://localhost:${DECODE_PORT}"
    echo "  Proxy  : http://${PREFILL_IP}:10001"
    echo ""
    echo "Containers will be shut down when the prefill node finishes (Ctrl+C to abort)."
    echo "(Set KEEP_ALIVE=1 to leave the container running after exit.)"
    echo ""

    # Block until the user stops the script (prefill node runs the benchmark).
    wait
else
    echo "Containers will be shut down when this script exits (Ctrl+C)."
    echo "(Set KEEP_ALIVE=1 to leave the container running after exit.)"
    echo ""

    # Block until the user stops the script; the cleanup trap handles teardown.
    wait
fi

fi
