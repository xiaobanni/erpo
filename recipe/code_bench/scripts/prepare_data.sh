#!/usr/bin/env bash
# Download and prepare the four code benchmarks used in the paper.
#
# Usage (from any directory):
#   bash recipe/code_bench/scripts/prepare_data.sh
#
# Outputs (relative to the repository root):
#   eval/livecodebench_v6/shared/test6.parquet               LCB v6 prompts (175 problems; RL val set)
#   eval/livecodebench_v6/shared/test6_hidden_output.parquet  same prompts with sample outputs 2+ hidden (RL train set)
#   eval/livecodebench_v6/shared/test6_benchmark_json/       per-problem public/private tests
#   eval/codecontests/shared/{test_prepared.parquet,benchmark_json/}   CodeContests (239 problems)
#   eval/codeforces/shared/{test_prepared.parquet,benchmark_json/}     CodeForces (467 problems)
#   eval/taco/shared/{test_prepared.parquet,benchmark_json/}           TACO test (881 of 945; problems
#                                                                      without a parsable public example are dropped)
#
# Raw files are downloaded from the Hugging Face Hub at pinned revisions and
# checked against their SHA-256 before use. Existing outputs are never
# overwritten: delete them first to re-run a step.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

LCB_REPO="livecodebench/code_generation_lite"
LCB_REVISION="0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILE="test6.jsonl"
LCB_SHA256="bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"

CC_REPO="Gen-Verse/CodeContests"
CC_REVISION="e06e6e140899dddc8ef841255db0b050b6da27d6"
CC_FILE="test/CodeContests.json"
CC_SHA256="f1a689886de41478573ebc16a6aea9489369ea6fe2fd8095f9ebe438d72ae283"

CF_REPO="Gen-Verse/CodeForces"
CF_REVISION="b3b3df092edb8f5412a6c4f83ec8753bcd9de943"
CF_FILE="test/CodeForces.json"
CF_SHA256="4690512fe260754eb82ac26e36436067cc0c2ab8e9398e7976f6e95ea03c0671"

TACO_REPO="NeurIPS20403/TACO_Eval"
TACO_REVISION="b58a2ddd6f6e4058384ae87ef5e52e58a97158df"
TACO_FILE="taco_for_test_evaluation.pkl"
TACO_SHA256="b77eabaff987d95922b8ef912c1fecfb3c7a82edbcf68b99eac8f94f57a83f76"

# download <repo_id> <filename> <revision> <local_dir> <sha256>
# Downloads <filename> into <local_dir>/<filename> and verifies its SHA-256.
download() {
    local repo_id="$1" filename="$2" revision="$3" local_dir="$4" sha256="$5"
    python3 - "${repo_id}" "${filename}" "${revision}" "${local_dir}" <<'PY'
import sys

from huggingface_hub import hf_hub_download

repo_id, filename, revision, local_dir = sys.argv[1:]
hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, repo_type="dataset", local_dir=local_dir)
PY
    echo "${sha256}  ${local_dir}/${filename}" | sha256sum --check --strict -
}

require_absent() {
    if [[ -e "$1" ]]; then
        echo "ERROR: expected $1 to not exist (outputs are never overwritten); found an existing path. Delete it to re-run." >&2
        exit 1
    fi
}

# ---- LiveCodeBench v6 (release_v6 increment: 175 problems, 2025-01 to 2025-04) ----
LCB_SHARED="eval/livecodebench_v6/shared"
require_absent "${LCB_SHARED}/test6.parquet"
require_absent "${LCB_SHARED}/test6_hidden_output.parquet"
require_absent "${LCB_SHARED}/test6_benchmark_json"
download "${LCB_REPO}" "${LCB_FILE}" "${LCB_REVISION}" eval/livecodebench_v6/raw "${LCB_SHA256}"
python3 -m recipe.code_bench.prepare_livecodebench_v6 \
    --raw-jsonl "eval/livecodebench_v6/raw/${LCB_FILE}" \
    --output-parquet "${LCB_SHARED}/test6.parquet" \
    --benchmark-dir "${LCB_SHARED}/test6_benchmark_json"
python3 -m recipe.code_bench.create_hidden_output_parquet \
    --input-parquet "${LCB_SHARED}/test6.parquet" \
    --output-parquet "${LCB_SHARED}/test6_hidden_output.parquet"

# ---- CodeContests / CodeForces (CURE releases; sample outputs 2+ hidden) ----
download "${CC_REPO}" "${CC_FILE}" "${CC_REVISION}" eval/codecontests/raw "${CC_SHA256}"
python3 -m recipe.code_bench.prepare_cure_code_benchmark \
    --dataset codecontests \
    --input-json "eval/codecontests/raw/${CC_FILE}" \
    --output-dir eval/codecontests/shared

download "${CF_REPO}" "${CF_FILE}" "${CF_REVISION}" eval/codeforces/raw "${CF_SHA256}"
python3 -m recipe.code_bench.prepare_cure_code_benchmark \
    --dataset codeforces \
    --input-json "eval/codeforces/raw/${CF_FILE}" \
    --output-dir eval/codeforces/shared

# ---- TACO test split ----
download "${TACO_REPO}" "${TACO_FILE}" "${TACO_REVISION}" eval/taco/raw "${TACO_SHA256}"
python3 -m recipe.code_bench.prepare_taco \
    --output-dir eval/taco/shared \
    --test-pkl "eval/taco/raw/${TACO_FILE}"

echo "Done. Benchmarks are under eval/{livecodebench_v6,codecontests,codeforces,taco}/shared/."
