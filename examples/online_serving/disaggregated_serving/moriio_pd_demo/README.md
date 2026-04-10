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

## 1. Get the Docker images

### Option A — Pull pre-built images (recommended)

```bash
docker pull ghcr.io/simondanielsson/vllm-rocm-moriio:dev
# Basic router support, i.e. PR https://github.com/vllm-project/router/pull/138
docker pull ghcr.io/simondanielsson/vllm-router:dev
# Basic router support + streaming, i.e. both PRs https://github.com/vllm-project/router/pull/138 and https://github.com/vllm-project/router/pull/139
docker pull ghcr.io/simondanielsson/vllm-router:dev-streaming
```

### Option B — Build from source

#### Router image

```bash
# clone the fork with the PR
git clone git@github.com:simondanielsson/router.git
cd router
# To build the image with basic mori support
git switch feature/moriio-support
docker build \
    -f <path-to-this-dir>/Dockerfile.router \
    -t ghcr.io/simondanielsson/vllm-router:dev \
    .

# To build the image with mori support + streaming: this branch contains the two PR's on top of each other
git switch reprod/moriio-support-and-streaming
docker build \
    -f <path-to-this-dir>/Dockerfile.router \
    -t ghcr.io/simondanielsson/vllm-router:dev-streaming \
    .
```

#### vLLM image

Build from the **root of this vllm repo**:

```bash
# clone the fork
git clone git@github.com:mpashkovskii/vllm.git
cd vllm
git switch fix/moriio-sane-defaults
docker build \
    -f examples/online_serving/disaggregated_serving/moriio_pd_demo/Dockerfile.vllm-rocm \
    -t ghcr.io/simondanielsson/vllm-rocm-moriio:dev \
    .
```

The vLLM build compiles the ROCm wheel from source, which takes a while.

---

## 2. Run the demo

### PR 1 — basic PD-disaggregation (smoke test only to confirm everything works)

```bash
MODEL=Qwen/Qwen3-8B \
PREFILL_GPU=0 \
DECODE_GPU=1 \
./examples/online_serving/disaggregated_serving/moriio_pd_demo/run_pd_demo.sh
```

This sends a single smoke-test request through the router after all services are healthy.

### PR 2 — with streaming support (full benchmark)

Once the streaming PR is merged into the router image, enable the full two-phase benchmark
(vllm-router + toy proxy) by setting `USE_BENCH=1`:

```bash
MODEL=Qwen/Qwen3-8B \
PREFILL_GPU=0 \
DECODE_GPU=1 \
USE_BENCH=1 \
./examples/online_serving/disaggregated_serving/moriio_pd_demo/run_pd_demo.sh
```

Benchmark results are written to `~/moriio-logs/benchmark_results.log`.

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-8B` | HuggingFace model id |
| `PREFILL_GPU` | `0` | GPU index for the prefill instance |
| `DECODE_GPU` | `1` | GPU index for the decode instance |
| `PREFILL_PORT` | `8100` | HTTP port for the prefill vLLM server |
| `DECODE_PORT` | `8200` | HTTP port for the decode vLLM server |
| `ROUTER_PORT` | `8080` | HTTP port for vllm-router |
| `PROXY_PING_PORT` | `36367` | ZMQ service-discovery port (router ↔ vLLM) |
| `HF_HOME` | `~/.cache/huggingface` | Host path to HuggingFace model cache |
| `LOG_DIR` | `~/moriio-logs` | Directory for container and benchmark logs |
| `USE_BENCH` | `0` | Set to `1` to run the full perf benchmark (requires streaming support) |
| `USE_GSM8K` | `0` | Set to `1` to run a GSM8K accuracy evaluation instead of the perf benchmark |
| `KEEP_ALIVE` | `0` | Set to `1` to leave containers running after the script exits |
| `VLLM_IMAGE` | `ghcr.io/simondanielsson/vllm-rocm-moriio:dev` | vLLM Docker image name |
| `ROUTER_IMAGE` | `ghcr.io/simondanielsson/vllm-router:dev` | Router image used for smoke-test (no streaming) |
| `ROUTER_STREAMING_IMAGE` | `ghcr.io/simondanielsson/vllm-router:dev-streaming` | Router image used for `USE_BENCH=1` / `USE_GSM8K=1` (streaming support required) |

---

## 3. Send a test request

Once all three containers are healthy, send requests through the **router**:

```bash
curl -s http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-8B",
        "prompt": "San Francisco is a",
        "max_tokens": 64,
        "temperature": 0
    }' | python3 -m json.tool
```

---

## 4. Load test with `vllm bench serve`

Run a synthetic load test against the router with 1 000-token inputs, 1 000-token
outputs, and a max concurrency of 16:

```bash
vllm bench serve \
    --base-url http://localhost:8080 \
    --backend openai-completions \
    --model Qwen/Qwen3-8B \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 1000 \
    --max-concurrency 16 \
    --num-prompts 200
```

| Flag | Value | Notes |
|------|-------|-------|
| `--base-url` | `http://localhost:8080` | Points at the router, not the vLLM instances directly |
| `--backend` | `openai-completions` | Uses the `/v1/completions` endpoint |
| `--model` | `Qwen/Qwen3-8B` | Must match the model served |
| `--dataset-name` | `random` | Fully synthetic, no external dataset file needed |
| `--random-input-len` | `1000` | Input sequence length (ISL) in tokens |
| `--random-output-len` | `1000` | Output sequence length (OSL) in tokens |
| `--max-concurrency` | `16` | Maximum number of in-flight requests |
| `--num-prompts` | `200` | Total requests to send; increase for a longer run |

---

## 5. GSM8K accuracy evaluation

To validate that PD-disaggregation produces correct outputs, run the GSM8K accuracy
evaluation against the router instead of a perf benchmark:

```bash
MODEL=Qwen/Qwen3-8B \
PREFILL_GPU=0 \
DECODE_GPU=1 \
USE_GSM8K=1 \
./examples/online_serving/disaggregated_serving/moriio_pd_demo/run_pd_demo.sh
```

This uses vLLM's built-in `tests/evals/gsm8k/gsm8k_eval.py` script, which sends the
full GSM8K test set (1 319 questions, 5-shot, temperature=0) through the router so all
requests flow through the MoRIIO PD-disaggregation pipeline.

Results are written to:

| File | Contents |
|------|----------|
| `~/moriio-logs/gsm8k_results.log` | Human-readable accuracy summary |
| `~/moriio-logs/gsm8k_results.json` | Machine-readable JSON with all metrics |

> **Note:** `USE_GSM8K=1` and `USE_BENCH=1` are mutually exclusive — GSM8K takes
> priority if both are set.

---

## 6. Teardown

The script shuts down all containers automatically when it exits.
To leave them running (e.g. to inspect logs or send additional requests), set `KEEP_ALIVE=1`.

If you need to tear down manually:

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
