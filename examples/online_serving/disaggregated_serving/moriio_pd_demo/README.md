# MoRIIO PD-disaggregation demo

Minimal reproduction script for running vLLM PD-disaggregation with the
MoRIIOConnector KV connector and the vllm-router.

Requires two ROCm GPUs on a single host.

---

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.router` | Builds the `vllm-router` Rust binary |
| `Dockerfile.vllm-rocm` | Builds vLLM from source on the ROCm base image |
| `run_pd_demo.sh` | Launches prefill, decode, and router containers |

---

## 1. Build images

### Router image

Build from the **root of the vllm-router repo** (`~/repos/router`):

```bash
cd ~/repos/router
docker build \
    -f <path-to-this-dir>/Dockerfile.router \
    -t vllm-router:dev \
    .
```

### vLLM image

Build from the **root of this vllm repo** (branch `fix/moriio-format-compatibility`):

```bash
cd ~/repos/mpashkov/vllm
docker build \
    -f examples/online_serving/disaggregated_serving/moriio_pd_demo/Dockerfile.vllm-rocm \
    -t vllm-rocm-moriio:dev \
    .
```

The vLLM build compiles the ROCm wheel from source, which takes a while.

---

## 2. Run the demo

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct \
PREFILL_GPU=0 \
DECODE_GPU=1 \
./examples/online_serving/disaggregated_serving/moriio_pd_demo/run_pd_demo.sh
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model id |
| `PREFILL_GPU` | `0` | GPU index for the prefill instance |
| `DECODE_GPU` | `1` | GPU index for the decode instance |
| `PREFILL_PORT` | `8100` | HTTP port for the prefill vLLM server |
| `DECODE_PORT` | `8200` | HTTP port for the decode vLLM server |
| `ROUTER_PORT` | `8080` | HTTP port for vllm-router |
| `PROXY_PING_PORT` | `36367` | ZMQ service-discovery port (router ↔ vLLM) |
| `HF_HOME` | `~/.cache/huggingface` | Host path to HuggingFace model cache |
| `VLLM_IMAGE` | `vllm-rocm-moriio:dev` | vLLM Docker image name |
| `ROUTER_IMAGE` | `vllm-router:dev` | Router Docker image name |

---

## 3. Send a test request

Once all three containers are healthy, send requests through the **router**:

```bash
curl -s http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 64,
        "temperature": 0
    }' | python3 -m json.tool
```

---

## 4. Teardown

```bash
docker rm -f moriio-prefill moriio-decode moriio-router
```

---

## Architecture

```
Client
  │
  ▼
vllm-router (port 8080)
  │  ZMQ service-discovery on PROXY_PING_PORT
  │  (vLLM instances register themselves at startup)
  ├──► Prefill instance (GPU 0, port 8100)
  │       kv_role = kv_producer
  │       MoRIIOConnector writes KV cache → Decode via RDMA
  │
  └──► Decode instance (GPU 1, port 8200)
          kv_role = kv_consumer
          MoRIIOConnector reads KV cache from Prefill via RDMA
```

The router uses `--vllm-pd-disaggregation` + `--vllm-discovery-address` so
that vLLM instances register dynamically at startup rather than being passed
as static `--prefill`/`--decode` URLs.
