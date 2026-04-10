#!/usr/bin/env bash
# run_pd_demo.sh — Launch a single-node MoRIIO PD-disaggregation demo
#
# Prerequisites
#   • Docker images built (see README in this directory):
#       vllm-rocm-moriio:dev   (from Dockerfile.vllm-rocm)
#       vllm-router:dev        (from Dockerfile.router)
#   • At least 2 ROCm GPUs visible to the host
#   • A HuggingFace model cache at HF_HOME (default: ~/.cache/huggingface)
#
# Usage:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct ./run_pd_demo.sh
#   MODEL=meta-llama/Llama-3.1-8B-Instruct PREFILL_GPU=0 DECODE_GPU=1 ./run_pd_demo.sh
#
# To tear everything down:
#   docker rm -f moriio-prefill moriio-decode moriio-router

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-8B}"

PREFILL_GPU="${PREFILL_GPU:-0}"            # GPU index for the prefill instance
DECODE_GPU="${DECODE_GPU:-1}"             # GPU index for the decode instance

PREFILL_PORT="${PREFILL_PORT:-8100}"      # HTTP port exposed by the prefill vLLM
DECODE_PORT="${DECODE_PORT:-8200}"        # HTTP port exposed by the decode vLLM
ROUTER_PORT="${ROUTER_PORT:-8080}"        # HTTP port exposed by vllm-router
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}" # ZMQ service-discovery port (router ↔ vLLM)

# MoRIIO internal ports — must be distinct between prefill and decode on the same host.
# handshake_port : initial MoRIIO engine handshake (default in code: 6301)
# notify_port    : prefill↔decode stage synchronisation (default in code: 61005)
PREFILL_HANDSHAKE_PORT="${PREFILL_HANDSHAKE_PORT:-6301}"
DECODE_HANDSHAKE_PORT="${DECODE_HANDSHAKE_PORT:-6302}"
PREFILL_NOTIFY_PORT="${PREFILL_NOTIFY_PORT:-61005}"
DECODE_NOTIFY_PORT="${DECODE_NOTIFY_PORT:-61006}"

VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/simondanielsson/vllm-rocm-moriio:dev-0410-1542}"
# Basic router (smoke-test only — no streaming support)
ROUTER_IMAGE="${ROUTER_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev}"
# Streaming-capable router (required for USE_BENCH=1 and USE_GSM8K=1)
ROUTER_STREAMING_IMAGE="${ROUTER_STREAMING_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev-streaming}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

LOG_DIR="${LOG_DIR:-${HOME}/moriio-logs}"
SHM_SIZE="${SHM_SIZE:-128G}"
USE_BENCH="${USE_BENCH:-0}"      # Set to 1 to run full benchmark on both routers
USE_GSM8K="${USE_GSM8K:-0}"     # Set to 1 to run GSM8K accuracy eval instead of the perf benchmark
KEEP_ALIVE="${KEEP_ALIVE:-0}"   # Set to 1 to leave containers running after the script exits

# Max tokens the prefill instance is allowed to generate (1 = prefill only)
# The decode instance does the actual generation.
PREFILL_MAX_MODEL_LEN="${PREFILL_MAX_MODEL_LEN:-8192}"
DECODE_MAX_MODEL_LEN="${DECODE_MAX_MODEL_LEN:-8192}"

# ── Derive the host IP that containers will reach each other on ───────────────
# Use the docker bridge gateway as "this host" so containers can talk to each
# other through the host network.  On Linux, host.docker.internal may not
# resolve; the bridge gateway (172.17.0.1) is usually reliable.
HOST_IP="${HOST_IP:-$(docker network inspect bridge \
    --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null \
    || echo "172.17.0.1")}"

mkdir -p "${LOG_DIR}"

echo "=== MoRIIO PD disaggregation demo ==="
echo "  Model         : ${MODEL}"
echo "  Host IP       : ${HOST_IP}"
echo "  Prefill GPU   : GPU ${PREFILL_GPU}  → port ${PREFILL_PORT}"
echo "  Decode  GPU   : GPU ${DECODE_GPU}  → port ${DECODE_PORT}"
echo "  Router  port  : ${ROUTER_PORT}"
echo "  Discovery port: ${PROXY_PING_PORT}"
echo "  Prefill MoRIIO: handshake=${PREFILL_HANDSHAKE_PORT} notify=${PREFILL_NOTIFY_PORT}"
echo "  Decode  MoRIIO: handshake=${DECODE_HANDSHAKE_PORT} notify=${DECODE_NOTIFY_PORT}"
echo "  Log dir       : ${LOG_DIR}"
echo ""

# ── Helper: wait for a vLLM /health endpoint ─────────────────────────────────
wait_for_health() {
    local name="$1"
    local port="$2"
    local max_wait=300   # seconds
    local interval=5
    local elapsed=0

    echo "Waiting for ${name} (port ${port}) to become healthy..."
    while true; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "  ${name} is healthy."
            return 0
        fi
        sleep "${interval}"
        elapsed=$((elapsed + interval))
        if [[ "${elapsed}" -ge "${max_wait}" ]]; then
            echo "ERROR: ${name} did not become healthy after ${max_wait}s" >&2
            docker logs "${name}" | tail -30 >&2
            exit 1
        fi
        echo "  Still waiting for ${name} (${elapsed}s / ${max_wait}s)..."
    done
}

# ── Remove stale containers (if any) ─────────────────────────────────────────
for cname in moriio-prefill moriio-decode moriio-router moriio-toy-proxy; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${cname}$"; then
        echo "Removing existing container: ${cname}"
        docker rm -f "${cname}"
    fi
done

# ── KV-transfer config shared between prefill and decode ─────────────────────
# proxy_ip   : address that vLLM uses to register with the router's ZMQ socket
# proxy_ping_port : the port the router listens on for registration messages
# http_port  : the vLLM instance's own HTTP port (used in the zmq_address that
#              gets embedded in request IDs by the router)

PREFILL_KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "proxy_ip": "${HOST_IP}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${PREFILL_PORT}",
    "handshake_port": "${PREFILL_HANDSHAKE_PORT}",
    "notify_port": "${PREFILL_NOTIFY_PORT}"
  }
}
EOF
)

DECODE_KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "MoRIIOConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "proxy_ip": "${HOST_IP}",
    "proxy_ping_port": "${PROXY_PING_PORT}",
    "http_port": "${DECODE_PORT}",
    "handshake_port": "${DECODE_HANDSHAKE_PORT}",
    "notify_port": "${DECODE_NOTIFY_PORT}"
  }
}
EOF
)

# ── Common docker run flags for both vLLM containers ─────────────────────────
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
)

# ── Launch prefill instance ───────────────────────────────────────────────────
echo ""
echo ">>> Starting prefill instance (GPU ${PREFILL_GPU}, port ${PREFILL_PORT})..."

docker run -d \
    --name moriio-prefill \
    "${VLLM_COMMON_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES="${PREFILL_GPU}" \
    "${VLLM_IMAGE}" \
    vllm serve "${MODEL}" \
        --port "${PREFILL_PORT}" \
        --max-model-len "${PREFILL_MAX_MODEL_LEN}" \
        --trust-remote-code \
        --kv-transfer-config "${PREFILL_KV_CONFIG}"

docker logs -f moriio-prefill 2>&1 | tee "${LOG_DIR}/prefill.log" &

# ── Launch decode instance ────────────────────────────────────────────────────
echo ">>> Starting decode instance (GPU ${DECODE_GPU}, port ${DECODE_PORT})..."

docker run -d \
    --name moriio-decode \
    "${VLLM_COMMON_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES="${DECODE_GPU}" \
    "${VLLM_IMAGE}" \
    vllm serve "${MODEL}" \
        --port "${DECODE_PORT}" \
        --max-model-len "${DECODE_MAX_MODEL_LEN}" \
        --trust-remote-code \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-transfer-config "${DECODE_KV_CONFIG}"

docker logs -f moriio-decode 2>&1 | tee "${LOG_DIR}/decode.log" &

# ── Wait for both vLLM instances to be healthy before starting the router ─────
wait_for_health "moriio-prefill" "${PREFILL_PORT}"
wait_for_health "moriio-decode"  "${DECODE_PORT}"

# ── Launch vllm-router ────────────────────────────────────────────────────────
# USE_BENCH and USE_GSM8K require the streaming-capable router image.
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

# ── Wait briefly for the router to be ready (it starts fast) ─────────────────
echo "Waiting for router to start..."
sleep 5

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "=== All services running ==="
echo "  Prefill  : http://localhost:${PREFILL_PORT}"
echo "  Decode   : http://localhost:${DECODE_PORT}"
echo "  Router   : http://localhost:${ROUTER_PORT}  ← send requests here"
echo ""
echo "Example request:"
echo "  curl -s http://localhost:${ROUTER_PORT}/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"${MODEL}\","
echo "      \"prompt\": \"San Francisco is a\","
echo "      \"max_tokens\": 64,"
echo "      \"temperature\": 0"
echo "    }' | python3 -m json.tool"
echo ""
echo "To follow logs (log files are written to ${LOG_DIR}):"
echo "  tail -f ${LOG_DIR}/prefill.log"
echo "  tail -f ${LOG_DIR}/decode.log"
echo "  tail -f ${LOG_DIR}/router.log"
echo ""
echo "Containers will be shut down automatically when the script exits."
echo "  (Set KEEP_ALIVE=1 to leave them running.)"

if [[ "${USE_GSM8K}" == "1" ]]; then

# ── GSM8K accuracy evaluation ─────────────────────────────────────────────────
# Uses vLLM's built-in gsm8k_eval.py which talks directly to the vLLM HTTP
# server.  We point it at the prefill/decode pair via the router so the
# requests flow through PD-disaggregation.
#
# The script is already present in the image at:
#   /app/vllm/tests/evals/gsm8k/gsm8k_eval.py
GSM8K_LOG="${LOG_DIR}/gsm8k_results.log"
GSM8K_JSON="${LOG_DIR}/gsm8k_results.json"
GSM8K_SCRIPT="/app/vllm/tests/evals/gsm8k/gsm8k_eval.py"
# Host-side path (so we can copy the script into the container if the
# pre-built image predates the Dockerfile change that added tests/evals/).
GSM8K_HOST_SCRIPT="$(cd "$(dirname "$0")/../../../.."; pwd)/tests/evals/gsm8k/gsm8k_eval.py"

echo ""
echo ">>> Running GSM8K accuracy evaluation through vllm-router..."
echo "    (1319 questions, 5-shot, temperature=0, results → ${GSM8K_LOG})"

# Ensure the eval script is present in the container (pre-built images may
# not have tests/ copied in; fall back to the host repo copy).
if ! docker exec moriio-prefill test -f "${GSM8K_SCRIPT}" 2>/dev/null; then
    if [[ -f "${GSM8K_HOST_SCRIPT}" ]]; then
        echo "    gsm8k_eval.py not found in image — copying from host repo..."
        docker exec moriio-prefill mkdir -p "$(dirname "${GSM8K_SCRIPT}")"
        docker cp "${GSM8K_HOST_SCRIPT}" "moriio-prefill:${GSM8K_SCRIPT}"
    else
        echo "ERROR: gsm8k_eval.py not found in image or host repo (${GSM8K_HOST_SCRIPT})" >&2
        exit 1
    fi
fi
{
    echo "======================================================"
    echo "  GSM8K evaluation via MoRIIO PD-disaggregation"
    echo "  Model : ${MODEL}"
    echo "  Date  : $(date)"
    echo "======================================================"
} | tee "${GSM8K_LOG}"

# Install lightweight deps (requests/tqdm/regex/numpy) then run the eval.
# The eval script calls the router's /v1/completions endpoint.
docker exec moriio-prefill bash -c \
    "pip install --quiet requests tqdm regex numpy && \
     python3 ${GSM8K_SCRIPT} \
         --host http://127.0.0.1 \
         --port ${ROUTER_PORT} \
         --num-questions 1319 \
         --num-shots 5 \
         --max-tokens 256 \
         --temperature 0.0 \
         --seed 42 \
         --save-results /tmp/gsm8k_results.json" \
    2>&1 | tee -a "${GSM8K_LOG}"

# Copy the JSON results out of the container
docker cp moriio-prefill:/tmp/gsm8k_results.json "${GSM8K_JSON}" 2>/dev/null || true

echo ""
echo "=== GSM8K evaluation complete ==="
echo "  Log  : ${GSM8K_LOG}"
echo "  JSON : ${GSM8K_JSON}"

elif [[ "${USE_BENCH}" == "1" ]]; then

# ── Benchmark helpers ─────────────────────────────────────────────────────────
# Both the benchmark tool and the toy proxy run inside moriio-prefill because:
#   • vllm (bench serve) is installed there
#   • examples/ are copied to /app/vllm/examples/ in the image (see Dockerfile)
#   • --network host means localhost:ROUTER_PORT is the same inside and outside
BENCH_LOG="${LOG_DIR}/benchmark_results.log"
TOY_PROXY_CONTAINER_PATH="/app/vllm/examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py"

# Common args shared by both benchmark runs
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-16}"
BENCH_NUM_WARMUPS=$((BENCH_MAX_CONCURRENCY * 2))
BENCH_NUM_PROMPTS=$((BENCH_MAX_CONCURRENCY * 10))

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

# Poll docker logs of the toy proxy container until both instances have registered.
# Using docker logs avoids Python stdout-buffering issues (print() to a file is
# fully buffered; docker logs reads directly from the container's log driver).
wait_for_toy_proxy_registrations() {
    local max_wait=120
    local interval=3
    local elapsed=0
    echo "Waiting for prefill and decode to register with toy proxy..."
    while true; do
        local p d
        p=$(docker logs moriio-toy-proxy 2>&1 | grep -c "Registered Prefill" || true)
        d=$(docker logs moriio-toy-proxy 2>&1 | grep -c "Registered Decode"  || true)
        if [[ "${p:-0}" -ge 1 && "${d:-0}" -ge 1 ]]; then
            echo "  Both instances registered."
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

# ── Phase 1: benchmark through vllm-router ────────────────────────────────────
echo ""
echo ">>> Phase 1: benchmarking through vllm-router..."
{
    echo "======================================================"
    echo "  Router: vllm-router"
    echo "  Date  : $(date)"
    echo "======================================================"
} | tee -a "${BENCH_LOG}"

docker exec moriio-prefill \
    vllm bench serve \
        --base-url "http://localhost:${ROUTER_PORT}" \
        "${BENCH_ARGS[@]}" 2>&1 | tee -a "${BENCH_LOG}"

# ── Phase 2: switch to toy proxy and benchmark again ──────────────────────────
# The toy proxy's HTTP port is hardcoded in the image as 10001; its ZMQ
# discovery port defaults to PROXY_PING_PORT (36367), same as vllm-router.
# We run it as a dedicated container so docker logs captures stdout without
# any Python output-buffering issues (no grep-on-file race).
# Prefill and decode are restarted so they register fresh with the new proxy.
TOY_PROXY_HTTP_PORT=10001

echo ""
echo ">>> Stopping vllm-router, prefill, and decode..."
docker rm -f moriio-router moriio-prefill moriio-decode

echo ">>> Starting toy proxy container (HTTP :${TOY_PROXY_HTTP_PORT}, ZMQ :${PROXY_PING_PORT})..."
docker run -d \
    --name moriio-toy-proxy \
    --network host \
    "${VLLM_IMAGE}" \
    bash -c "pip install --quiet --ignore-installed quart aiohttp msgpack && \
             python3 -u ${TOY_PROXY_CONTAINER_PATH}"

docker logs -f moriio-toy-proxy 2>&1 | tee "${LOG_DIR}/toy_proxy.log" &

# Wait for the toy proxy HTTP port before starting vLLM (avoids a race where
# instances start sending heartbeats before the ZMQ socket is bound).
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

echo ">>> Restarting prefill instance (GPU ${PREFILL_GPU}, port ${PREFILL_PORT})..."
docker run -d \
    --name moriio-prefill \
    "${VLLM_COMMON_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES="${PREFILL_GPU}" \
    "${VLLM_IMAGE}" \
    vllm serve "${MODEL}" \
        --port "${PREFILL_PORT}" \
        --max-model-len "${PREFILL_MAX_MODEL_LEN}" \
        --trust-remote-code \
        --kv-transfer-config "${PREFILL_KV_CONFIG}"

docker logs -f moriio-prefill 2>&1 | tee "${LOG_DIR}/prefill_phase2.log" &

echo ">>> Restarting decode instance (GPU ${DECODE_GPU}, port ${DECODE_PORT})..."
docker run -d \
    --name moriio-decode \
    "${VLLM_COMMON_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES="${DECODE_GPU}" \
    "${VLLM_IMAGE}" \
    vllm serve "${MODEL}" \
        --port "${DECODE_PORT}" \
        --max-model-len "${DECODE_MAX_MODEL_LEN}" \
        --trust-remote-code \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-transfer-config "${DECODE_KV_CONFIG}"

docker logs -f moriio-decode 2>&1 | tee "${LOG_DIR}/decode_phase2.log" &

wait_for_health "moriio-prefill" "${PREFILL_PORT}"
wait_for_health "moriio-decode"  "${DECODE_PORT}"

wait_for_toy_proxy_registrations

echo ""
echo ">>> Phase 2: benchmarking through toy proxy..."
{
    echo ""
    echo "======================================================"
    echo "  Router: moriio_toy_proxy_server.py"
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

# ── Smoke test: single completion request through the router ──────────────────
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
echo "(Set USE_BENCH=1 to run the full perf benchmark, or USE_GSM8K=1 for accuracy evaluation.)"

fi

# ── Teardown ──────────────────────────────────────────────────────────────────
if [[ "${KEEP_ALIVE}" == "1" ]]; then
    echo ""
    echo "KEEP_ALIVE=1 — containers left running."
    echo "  To tear down: docker rm -f moriio-prefill moriio-decode moriio-router"
else
    echo ""
    echo ">>> Shutting down containers..."
    docker rm -f moriio-prefill moriio-decode moriio-router moriio-toy-proxy 2>/dev/null || true
    echo "Done. All containers removed."
fi
