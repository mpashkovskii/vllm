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
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

PREFILL_GPU="${PREFILL_GPU:-0}"            # GPU index for the prefill instance
DECODE_GPU="${DECODE_GPU:-1}"             # GPU index for the decode instance

PREFILL_PORT="${PREFILL_PORT:-8100}"      # HTTP port exposed by the prefill vLLM
DECODE_PORT="${DECODE_PORT:-8200}"        # HTTP port exposed by the decode vLLM
ROUTER_PORT="${ROUTER_PORT:-8080}"        # HTTP port exposed by vllm-router
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}" # ZMQ service-discovery port (router ↔ vLLM)

VLLM_IMAGE="${VLLM_IMAGE:-vllm-rocm-moriio:dev}"
ROUTER_IMAGE="${ROUTER_IMAGE:-vllm-router:dev}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

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

echo "=== MoRIIO PD disaggregation demo ==="
echo "  Model         : ${MODEL}"
echo "  Host IP       : ${HOST_IP}"
echo "  Prefill GPU   : GPU ${PREFILL_GPU}  → port ${PREFILL_PORT}"
echo "  Decode  GPU   : GPU ${DECODE_GPU}  → port ${DECODE_PORT}"
echo "  Router  port  : ${ROUTER_PORT}"
echo "  Discovery port: ${PROXY_PING_PORT}"
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
for cname in moriio-prefill moriio-decode moriio-router; do
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
    "http_port": "${PREFILL_PORT}"
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
    "http_port": "${DECODE_PORT}"
  }
}
EOF
)

# ── Common docker run flags for both vLLM containers ─────────────────────────
VLLM_COMMON_ARGS=(
    --network host
    --ipc host
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --group-add video
    -v /dev/kfd:/dev/kfd
    -v /dev/dri:/dev/dri
    -v "${HF_HOME}:/root/.cache/huggingface"
    -e HF_HOME=/root/.cache/huggingface
    -e HF_HUB_ENABLE_HF_TRANSFER=0
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

# ── Wait for both vLLM instances to be healthy before starting the router ─────
wait_for_health "moriio-prefill" "${PREFILL_PORT}"
wait_for_health "moriio-decode"  "${DECODE_PORT}"

# ── Launch vllm-router ────────────────────────────────────────────────────────
echo ""
echo ">>> Starting vllm-router (port ${ROUTER_PORT}, discovery port ${PROXY_PING_PORT})..."

docker run -d \
    --name moriio-router \
    --network host \
    "${ROUTER_IMAGE}" \
    vllm-router \
        --vllm-pd-disaggregation \
        --vllm-discovery-address "0.0.0.0:${PROXY_PING_PORT}" \
        --port "${ROUTER_PORT}" \
        --host 0.0.0.0 \
        --policy consistent_hash \
        --prefill-policy consistent_hash \
        --decode-policy consistent_hash \
        --log-level info

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
echo "To follow logs:"
echo "  docker logs -f moriio-prefill"
echo "  docker logs -f moriio-decode"
echo "  docker logs -f moriio-router"
echo ""
echo "To tear down:"
echo "  docker rm -f moriio-prefill moriio-decode moriio-router"
