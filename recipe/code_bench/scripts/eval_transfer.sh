#!/usr/bin/env bash
# Evaluation on CodeContests / CodeForces / TACO (transfer) and optionally LiveCodeBench.
#
# Usage (from any directory):
#   bash recipe/code_bench/scripts/eval_transfer.sh <model_path_or_fsdp_actor_dir> <out_dir>
#
#   <model_path_or_fsdp_actor_dir>
#       - an FSDP actor checkpoint dir written by train.sh
#         (e.g. checkpoints/qwen3_4b_lcb_erpo/global_step_150/actor): it is first
#         merged into a Hugging Face model under <out_dir>/merged_hf;
#       - or a Hugging Face model dir / hub id (e.g. Qwen/Qwen3-4B for the base model).
#   <out_dir>  output directory; per benchmark it gets <bench>/responses.parquet
#              and <bench>/eval_pass_at_k.csv.
#
# N_SAMPLES samples per problem are drawn with the training sampling parameters
# (temperature 0.8, top_p 0.8, top_k 20, min_p 0, max response 8192 tokens,
# enable_thinking=False) and scored with strict pass@k (public + private tests)
# for k = 1, 2, 4, ..., N_SAMPLES.
#
# Environment variables:
#   BENCHMARKS    space-separated subset of {codecontests codeforces taco lcb}
#                                                          (default: "codecontests codeforces taco")
#   N_SAMPLES     samples per problem, a power of two       (default: 16; e.g. 256 for inference-time scaling)
#   N_GPUS        GPUs for generation                      (default: 8)
#   EVAL_WORKERS  main_eval --max-workers                  (default: main_eval's cgroup-aware default)
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <model_path_or_fsdp_actor_dir> <out_dir>" >&2
    exit 1
fi
MODEL_ARG="$1"
OUT_DIR="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-8}"
N_SAMPLES="${N_SAMPLES:-16}"
read -r -a BENCHMARKS <<< "${BENCHMARKS:-codecontests codeforces taco}"

if [[ ! "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || (( (N_SAMPLES & (N_SAMPLES - 1)) != 0 )); then
    echo "ERROR: expected N_SAMPLES to be a power of two (1, 2, 4, ...), got '${N_SAMPLES}'." >&2
    exit 1
fi
PASS_K_LIST="1"
for (( k = 2; k <= N_SAMPLES; k *= 2 )); do
    PASS_K_LIST="${PASS_K_LIST},${k}"
done

# bench_paths <bench>: sets BENCH_PARQUET and BENCH_JSON_DIR for one benchmark.
bench_paths() {
    case "$1" in
        codecontests|codeforces|taco)
            BENCH_PARQUET="eval/$1/shared/test_prepared.parquet"
            BENCH_JSON_DIR="eval/$1/shared/benchmark_json" ;;
        lcb)
            BENCH_PARQUET="eval/livecodebench_v6/shared/test6.parquet"
            BENCH_JSON_DIR="eval/livecodebench_v6/shared/test6_benchmark_json" ;;
        *)
            echo "ERROR: expected each BENCHMARKS entry in {codecontests codeforces taco lcb}, got '$1'." >&2
            exit 1 ;;
    esac
}

for bench in "${BENCHMARKS[@]}"; do
    bench_paths "${bench}"
    for path in "${BENCH_PARQUET}" "${BENCH_JSON_DIR}"; do
        if [[ ! -e "${path}" ]]; then
            echo "ERROR: expected ${path}; not found. Run recipe/code_bench/scripts/prepare_data.sh first." >&2
            exit 1
        fi
    done
    if [[ -e "${OUT_DIR}/${bench}" ]]; then
        echo "ERROR: expected ${OUT_DIR}/${bench} to not exist (outputs are never overwritten); found an existing path." >&2
        exit 1
    fi
done
mkdir -p "${OUT_DIR}"

# ---- Resolve the model: merge an FSDP actor checkpoint if one is given ----
if compgen -G "${MODEL_ARG}/model_world_size_*_rank_0.pt" > /dev/null; then
    MODEL_PATH="${OUT_DIR}/merged_hf"
    if [[ -e "${MODEL_PATH}" ]]; then
        echo "ERROR: expected ${MODEL_PATH} to not exist (outputs are never overwritten); found an existing path." >&2
        exit 1
    fi
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${MODEL_ARG}" \
        --target_dir "${MODEL_PATH}" \
        --trust-remote-code
else
    MODEL_PATH="${MODEL_ARG}"
fi

# ---- Generate N_SAMPLES samples per problem and score them ----
for bench in "${BENCHMARKS[@]}"; do
    bench_paths "${bench}"
    bench_out="${OUT_DIR}/${bench}"
    mkdir -p "${bench_out}"
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="${N_GPUS}" \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=0.8 \
        actor_rollout_ref.rollout.top_p=0.8 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.min_p=0.0 \
        actor_rollout_ref.rollout.response_length=8192 \
        actor_rollout_ref.rollout.max_model_len=12288 \
        actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.n="${N_SAMPLES}" \
        "data.train_files=['${BENCH_PARQUET}']" \
        data.prompt_key=prompt \
        +data.output_path="${bench_out}/responses.parquet" \
        +data.apply_chat_template_kwargs.enable_thinking=False

    EVAL_ARGS=(
        --data-path "${bench_out}/responses.parquet"
        --benchmark-dir "${BENCH_JSON_DIR}"
        --timeout 10
        --pass-k-list "${PASS_K_LIST}"
        --output-csv "${bench_out}/eval_pass_at_k.csv"
    )
    if [[ -n "${EVAL_WORKERS:-}" ]]; then
        EVAL_ARGS+=(--max-workers "${EVAL_WORKERS}")
    fi
    python3 -m recipe.code_bench.main_eval "${EVAL_ARGS[@]}"
done

# ---- Summary: pass@k per benchmark and the mean over the evaluated benchmarks ----
python3 - "${OUT_DIR}" "${PASS_K_LIST}" "${BENCHMARKS[@]}" <<'PY'
import sys

import pandas as pd

out_dir, pass_ks, benches = sys.argv[1], [int(k) for k in sys.argv[2].split(",")], sys.argv[3:]
rows = []
for bench in benches:
    row = pd.read_csv(f"{out_dir}/{bench}/eval_pass_at_k.csv").iloc[0]
    rows.append((bench, [100.0 * row[f"pass@{k}"] for k in pass_ks]))
print(f"{'benchmark':<14}" + "".join(f"{'P@' + str(k):>8}" for k in pass_ks))
for bench, values in rows:
    print(f"{bench:<14}" + "".join(f"{v:>8.2f}" for v in values))
means = [sum(values[i] for _, values in rows) / len(rows) for i in range(len(pass_ks))]
print(f"{'mean':<14}" + "".join(f"{v:>8.2f}" for v in means))
PY
