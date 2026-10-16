#!/usr/bin/env bash
# Generate the LiveCodeBench probe inputs consumed by the PCR reward (paper setting).
#
# Usage (from any directory):
#   bash recipe/code_bench/scripts/generate_probes.sh
#
# Setting: Qwen3-4B in thinking mode, probe prompt v1, K=10 probes per problem,
# temperature 0.8, top_p 0.8 (top_k/min_p are not sent, so vLLM uses the
# model's generation_config defaults), 32 problems in flight.
#   Pass 1: every problem, max_tokens 10000 per probe.
#   Pass 2: problems with fewer than K unique non-empty probes after pass 1 get
#           (K - unique) additional probes with max_tokens 32000; all pass-1
#           probes are kept.
#
# Environment variables:
#   MODEL_PATH  model served by vLLM            (default: Qwen/Qwen3-4B)
#   N_GPUS      vLLM data-parallel size          (default: 8)
#   PORT        vLLM server port                 (default: 8000)
#   OUTPUT_DIR  output directory                 (default: eval/livecodebench_v6/probes)
#
# Output: ${OUTPUT_DIR}/lcb_probes_K10.parquet (pass it to train.sh as PROBE_PARQUET).
# Requires eval/livecodebench_v6/shared/test6_benchmark_json (scripts/prepare_data.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
N_GPUS="${N_GPUS:-8}"
PORT="${PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/livecodebench_v6/probes}"

BENCHMARK_DIR="eval/livecodebench_v6/shared/test6_benchmark_json"
PASS1_PARQUET="${OUTPUT_DIR}/lcb_probes_K10_pass1.parquet"
FINAL_PARQUET="${OUTPUT_DIR}/lcb_probes_K10.parquet"
SERVER_READY_TIMEOUT_S=1800

if [[ ! -d "${BENCHMARK_DIR}" ]]; then
    echo "ERROR: expected benchmark dir ${BENCHMARK_DIR}; not found. Run recipe/code_bench/scripts/prepare_data.sh first." >&2
    exit 1
fi
for path in "${PASS1_PARQUET}" "${FINAL_PARQUET}"; do
    if [[ -e "${path}" ]]; then
        echo "ERROR: expected ${path} to not exist (outputs are never overwritten); found an existing file." >&2
        exit 1
    fi
done
mkdir -p "${OUTPUT_DIR}"

VLLM_PID=""

start_server() {  # start_server <max_model_len>
    vllm serve "${MODEL_PATH}" \
        --tensor-parallel-size 1 \
        --data-parallel-size "${N_GPUS}" \
        --port "${PORT}" \
        --max-model-len "$1" \
        --gpu-memory-utilization 0.85 &
    VLLM_PID=$!
    local waited=0
    until curl --silent --fail "http://localhost:${PORT}/v1/models" > /dev/null; do
        if ! kill -0 "${VLLM_PID}"; then
            echo "ERROR: vLLM server (pid ${VLLM_PID}) exited before becoming ready." >&2
            exit 1
        fi
        if (( waited >= SERVER_READY_TIMEOUT_S )); then
            echo "ERROR: vLLM server not ready after ${SERVER_READY_TIMEOUT_S}s (expected HTTP 200 from /v1/models)." >&2
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
}

stop_server() {
    kill "${VLLM_PID}"
    # A SIGTERM'd server exits with 0 (graceful) or 143 (128 + SIGTERM); anything else is an error.
    wait "${VLLM_PID}" || [[ $? -eq 143 ]]
    VLLM_PID=""
}

on_exit() {
    if [[ -n "${VLLM_PID}" ]]; then
        kill "${VLLM_PID}"
    fi
}
trap on_exit EXIT

# ---- Pass 1: K=10 probes per problem, max_tokens 10000 ----
start_server 16000
python3 -m recipe.code_bench.generate_aug_inputs_iterative \
    --benchmark-json-dir "${BENCHMARK_DIR}" \
    --output-parquet "${PASS1_PARQUET}" \
    --base-url "http://localhost:${PORT}" \
    --model-name "${MODEL_PATH}" \
    --k 10 --concurrency 32 --max-tokens 10000 --temperature 0.8 \
    --enable-thinking --prompt-version v1
stop_server

# ---- Pass 2: top up problems with < 10 unique probes, max_tokens 32000 ----
start_server 40000
python3 -m recipe.code_bench.generate_aug_inputs_iterative \
    --benchmark-json-dir "${BENCHMARK_DIR}" \
    --incremental-from "${PASS1_PARQUET}" \
    --output-parquet "${FINAL_PARQUET}" \
    --base-url "http://localhost:${PORT}" \
    --model-name "${MODEL_PATH}" \
    --k 10 --concurrency 32 --max-tokens 32000 --temperature 0.8 \
    --enable-thinking --prompt-version v1
stop_server

echo "Probes written to ${FINAL_PARQUET}"
