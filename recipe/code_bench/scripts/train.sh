#!/usr/bin/env bash
# Test-time RL on LiveCodeBench v6 with Qwen3-4B (non-thinking), paper setting.
#
# Usage (from any directory):
#   bash recipe/code_bench/scripts/train.sh <erpo|nsr_pcr|grpo_pcr|grpo_pub> [extra hydra overrides...]
#
# Methods:
#   erpo      advantage grpo_fmv_penalty_only + PCR reward + entropy ceiling (lambda 0.5, target 0.5)
#   nsr_pcr   advantage nsr_pcr + PCR reward
#   grpo_pcr  advantage grpo + PCR reward
#   grpo_pub  advantage grpo + public-test binary reward (no probes)
#
# Environment variables:
#   PROBE_PARQUET     probe parquet from generate_probes.sh (required for erpo / nsr_pcr / grpo_pcr)
#   MODEL_PATH        policy initialization              (default: Qwen/Qwen3-4B)
#   N_GPUS            GPUs on this node                  (default: 8)
#   RUN_NAME          experiment name                    (default: qwen3_4b_lcb_<method>)
#   OUTPUT_DIR        checkpoint directory               (default: checkpoints/<RUN_NAME>)
#   LOGGER            trainer.logger                     (default: '["console"]')
#   VAL_BEFORE_TRAIN  validate the initial policy first  (default: false)
#
# Train set = LCB v6 prompts with sample outputs 2+ hidden; validation set = the
# same 175 problems scored on public + private tests (16 samples per problem).
# Checkpoints and validation every 10 steps; 150 steps in total.
set -euo pipefail

METHODS="erpo | nsr_pcr | grpo_pcr | grpo_pub"
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <${METHODS}> [extra hydra overrides...]" >&2
    exit 1
fi
METHOD="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
N_GPUS="${N_GPUS:-8}"
RUN_NAME="${RUN_NAME:-qwen3_4b_lcb_${METHOD}}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
LOGGER="${LOGGER:-[\"console\"]}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"

LCB_SHARED="eval/livecodebench_v6/shared"
TRAIN_PARQUET="${LCB_SHARED}/test6_hidden_output.parquet"
VAL_PARQUET="${LCB_SHARED}/test6.parquet"
BENCHMARK_DIR="${LCB_SHARED}/test6_benchmark_json"
for path in "${TRAIN_PARQUET}" "${VAL_PARQUET}" "${BENCHMARK_DIR}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: expected ${path}; not found. Run recipe/code_bench/scripts/prepare_data.sh first." >&2
        exit 1
    fi
done

case "${METHOD}" in
    erpo)     ADV_ESTIMATOR="grpo_fmv_penalty_only"; REWARD_MODE="fmv_pointwise" ;;
    nsr_pcr)  ADV_ESTIMATOR="nsr_pcr";               REWARD_MODE="fmv_pointwise" ;;
    grpo_pcr) ADV_ESTIMATOR="grpo";                  REWARD_MODE="fmv_pointwise" ;;
    grpo_pub) ADV_ESTIMATOR="grpo";                  REWARD_MODE="public_binary" ;;
    *) echo "ERROR: expected method in {${METHODS}}, got '${METHOD}'" >&2; exit 1 ;;
esac

METHOD_ARGS=()
if [[ "${REWARD_MODE}" == "fmv_pointwise" ]]; then
    if [[ -z "${PROBE_PARQUET:-}" ]]; then
        echo "ERROR: method ${METHOD} uses the PCR reward; expected PROBE_PARQUET=<probe parquet from generate_probes.sh>, got it unset." >&2
        exit 1
    fi
    if [[ ! -f "${PROBE_PARQUET}" ]]; then
        echo "ERROR: expected PROBE_PARQUET to be a file; ${PROBE_PARQUET} not found." >&2
        exit 1
    fi
    METHOD_ARGS+=(
        "+reward.reward_kwargs.fmv_aug_inputs_parquet=${PROBE_PARQUET}"
        +reward.reward_kwargs.fmv_timeout=10
        +reward.reward_kwargs.fmv_max_workers=16
        +reward.reward_kwargs.fmv_exec_granularity=pair
    )
fi
if [[ "${METHOD}" == "erpo" ]]; then
    METHOD_ARGS+=(
        actor_rollout_ref.actor.entropy_ceiling_lambda=0.5
        actor_rollout_ref.actor.entropy_ceiling_target=0.5
        actor_rollout_ref.actor.calculate_entropy=true
    )
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward=False \
    "data.train_files=['${TRAIN_PARQUET}']" \
    "data.val_files=['${VAL_PARQUET}']" \
    data.prompt_key=prompt \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.nccl_timeout=1200 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.min_p=0.0 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.num_workers=8 \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=CodeBenchTestRLRewardManager \
    reward.reward_manager.module.path=recipe/code_bench/test_rl_reward_manager.py \
    +reward.reward_kwargs.reward_mode="${REWARD_MODE}" \
    +reward.reward_kwargs.benchmark_dir="${BENCHMARK_DIR}" \
    +reward.reward_kwargs.timeout=10 \
    +reward.reward_kwargs.evaluation_concurrency=auto \
    "${METHOD_ARGS[@]}" \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=code_ttrl \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=80 \
    trainer.total_training_steps=150 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    "$@"
