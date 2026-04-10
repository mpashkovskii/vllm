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

VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/simondanielsson/vllm-rocm-moriio:dev}"
ROUTER_IMAGE="${ROUTER_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

LOG_DIR="${LOG_DIR:-${HOME}/moriio-logs}"
SHM_SIZE="${SHM_SIZE:-128G}"

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
)

# ── Temporary: Broadcom bnxt_re RDMA driver install (runs inside each container)
# Remove once the image is rebuilt with this baked in (see Dockerfile.vllm-rocm).
BNXT_DRIVER_INSTALL=$(cat <<'BNXT_EOF'
echo "--- Installing Broadcom bnxt_re RDMA driver ---"
apt-get update -q -y && apt-get install -y -q autoconf libibverbs-dev ibverbs-utils libtool unzip wget
cd /tmp
wget -q https://docs.broadcom.com/docs-and-downloads/ethernet-network-adapters/NXE/Thor2/GCA1/bcm5760x_230.2.52.0a.zip
unzip -q bcm5760x_230.2.52.0a.zip
cd bcm5760x_230.2.52.0a/drivers_linux/bnxt_rocelib/
results=$(find -name "libbnxt*.tar.gz")
tar -xf $results
untar_dir=$(find . -maxdepth 1 -type d -name "libbnxt*" ! -name "*.tar.gz" | head -n 1)
cd $untar_dir
sh autogen.sh && ./configure && make
find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \;
make install all
sh -c "echo /usr/local/lib >> /etc/ld.so.conf"
ldconfig
cp -f bnxt_re.driver /etc/libibverbs.d/
cd /tmp && rm -rf bcm5760x_230.2.52.0a bcm5760x_230.2.52.0a.zip
echo "--- Broadcom driver installed successfully ---"
BNXT_EOF
)

# ── Temporary: patch MoRIIOConstants.PING_INTERVAL in the installed vLLM wheel
# Aligns the heartbeat with the router's DEFAULT_PING_SECONDS (5s TTL).
# Remove once the vLLM image is rebuilt with PING_INTERVAL = 3.
VLLM_PATCH=$(cat <<'PATCH_EOF'
python3 -c "
import inspect
import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common as m
f = inspect.getfile(m)
txt = open(f).read()
patched = txt.replace('PING_INTERVAL = 5', 'PING_INTERVAL = 3')
open(f, 'w').write(patched)
print('Patched PING_INTERVAL → 3 in', f)
"
PATCH_EOF
)

# ── Launch prefill instance ───────────────────────────────────────────────────
echo ""
echo ">>> Starting prefill instance (GPU ${PREFILL_GPU}, port ${PREFILL_PORT})..."

docker run -d \
    --name moriio-prefill \
    "${VLLM_COMMON_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES="${PREFILL_GPU}" \
    "${VLLM_IMAGE}" \
    bash -c "${BNXT_DRIVER_INSTALL} && pip install --quiet msgpack && ${VLLM_PATCH} && vllm serve '${MODEL}' \
        --port '${PREFILL_PORT}' \
        --max-model-len '${PREFILL_MAX_MODEL_LEN}' \
        --trust-remote-code \
        --kv-transfer-config '${PREFILL_KV_CONFIG}'"

docker logs -f moriio-prefill 2>&1 | tee "${LOG_DIR}/prefill.log" &

# ── Launch decode instance ────────────────────────────────────────────────────
echo ">>> Starting decode instance (GPU ${DECODE_GPU}, port ${DECODE_PORT})..."

docker run -d \
    --name moriio-decode \
    "${VLLM_COMMON_ARGS[@]}" \
    -e HIP_VISIBLE_DEVICES="${DECODE_GPU}" \
    "${VLLM_IMAGE}" \
    bash -c "${BNXT_DRIVER_INSTALL} && pip install --quiet msgpack && ${VLLM_PATCH} && vllm serve '${MODEL}' \
        --port '${DECODE_PORT}' \
        --max-model-len '${DECODE_MAX_MODEL_LEN}' \
        --trust-remote-code \
        --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' \
        --kv-transfer-config '${DECODE_KV_CONFIG}'"

docker logs -f moriio-decode 2>&1 | tee "${LOG_DIR}/decode.log" &

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
echo "To tear down:"
echo "  docker rm -f moriio-prefill moriio-decode moriio-router"
