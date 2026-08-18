#!/bin/bash
# ============================================================================
#
#
#
# ============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_MODE="${ROLLOUT_MODE:-state_group}"
SELECTION_MODE="${SELECTION_MODE:-best}"  # state_group candidate commit: best/random/mixed
RANDOM_SELECT_PROB="${RANDOM_SELECT_PROB:-0}"
RANDOM_SELECT_PROB_SCHEDULE="${RANDOM_SELECT_PROB_SCHEDULE:-}"
VAL_BATCH="${VAL_BATCH:-128}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_RESP="${MAX_RESP:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SAVE_FREQ="${SAVE_FREQ:-25}"
RESUME_MODE="${RESUME_MODE:-disable}"     # disable/auto/resume_path
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
TEST_FREQ="${TEST_FREQ:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
USE_KL="${USE_KL:-True}"
KL_COEF="${KL_COEF:-0.001}"
PPO_MICRO="${PPO_MICRO:-2}"
LOGPROB_MICRO="${LOGPROB_MICRO:-4}"    # rollout/ref log-prob micro-batch
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-16384}"
LOGPROB_MAX_TOKENS_PER_GPU="${LOGPROB_MAX_TOKENS_PER_GPU:-16384}"
SP_SIZE="${SP_SIZE:-1}"                # actor Ulysses sequence parallel size
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
RAY_CPUS="${RAY_CPUS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
GAME_ACTION_FORMAT="${GAME_ACTION_FORMAT:-action_tag}"
ORACLE_POLICY="${ORACLE_POLICY:-all_oracle_actions}"  # turn-level oracle action policy
ORACLE_REWARD="${ORACLE_REWARD:-2}"       # safe reveal reward
ORACLE_FLAG_REWARD="${ORACLE_FLAG_REWARD:-1}"  # certain flag reward
ORACLE_GUESS_REWARD="${ORACLE_GUESS_REWARD:-1}"  # min-posterior guess reveal reward
NON_ORACLE_PENALTY="${NON_ORACLE_PENALTY:--1}"  # legal non-oracle reveal reward
NON_ORACLE_FLAG_PENALTY="${NON_ORACLE_FLAG_PENALTY:--1.5}"  # legal non-oracle flag penalty
INVALID_PENALTY="${INVALID_PENALTY:--2}"  # parse/illegal/truncate penalty
OUTCOME_REWARD_SCALE="${OUTCOME_REWARD_SCALE:-0}"  # terminal outcome bonus disabled for imitation
STATE_GROUP_ADV_MODE="${STATE_GROUP_ADV_MODE:-mean_then_batch_whiten}"
VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD="${VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD:-null}"  # null disables update skipping
LOSS_MODE="${LOSS_MODE:-vanilla}"

if (( MAX_PROMPT + MAX_RESP > MAX_MODEL_LEN )); then
    echo "ERROR: MAX_PROMPT + MAX_RESP must not exceed MAX_MODEL_LEN" >&2
    exit 1
fi
if (( N_GPUS % SP_SIZE != 0 )); then
    echo "ERROR: N_GPUS must be divisible by SP_SIZE" >&2
    exit 1
fi
case "$USE_REMOVE_PADDING" in
    true | True | TRUE | 1 | yes | Yes | YES) use_remove_padding=true ;;
    *) use_remove_padding=false ;;
esac
if (( SP_SIZE > 1 )) && [[ "$use_remove_padding" != "true" ]]; then
    echo "ERROR: SP_SIZE > 1 requires USE_REMOVE_PADDING=True" >&2
    exit 1
fi
case "$USE_DYNAMIC_BSZ" in
    true | True | TRUE | 1 | yes | Yes | YES) use_dynamic_bsz=true ;;
    *) use_dynamic_bsz=false ;;
esac
if [[ "$use_dynamic_bsz" == "true" ]]; then
    if (( MAX_PROMPT + MAX_RESP > PPO_MAX_TOKENS_PER_GPU * SP_SIZE )); then
        echo "ERROR: training sequence exceeds PPO dynamic token budget across SP ranks" >&2
        exit 1
    fi
    if (( MAX_PROMPT + MAX_RESP > LOGPROB_MAX_TOKENS_PER_GPU * SP_SIZE )); then
        echo "ERROR: training sequence exceeds log-prob dynamic token budget across SP ranks" >&2
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPR_GAMES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$VPR_GAMES_DIR/../.." && pwd)"
source "$VPR_GAMES_DIR/launcher_utils.sh"
DATA_DIR="$VPR_GAMES_DIR/data/vpr_minesweeper"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$TS}"
mkdir -p "$RUN_DIR" "$RUN_DIR/ckpt" "$RUN_DIR/tensorboard"
LOG_FILE="$RUN_DIR/train.log"

echo "=== Minesweeper | VPR (per-turn oracle reward + VPR advantage) ==="
echo "Model:        $MODEL_PATH"
echo "Steps: $TRAIN_STEPS | rollout_mode: $ROLLOUT_MODE | selection: $SELECTION_MODE p_random=$RANDOM_SELECT_PROB schedule=${RANDOM_SELECT_PROB_SCHEDULE:-none} | rollout/step: ${TRAIN_BATCH}x${ROLLOUT_N} | val: $VAL_BATCH | max_resp: $MAX_RESP | thinking: $ENABLE_THINKING"
echo "Reward:       policy: $ORACLE_POLICY | safe_reveal: $ORACLE_REWARD | certain_flag: $ORACLE_FLAG_REWARD | guess: $ORACLE_GUESS_REWARD | non_oracle_reveal: $NON_ORACLE_PENALTY | non_oracle_flag: $NON_ORACLE_FLAG_PENALTY | invalid/truncate: $INVALID_PENALTY | outcome_scale: $OUTCOME_REWARD_SCALE | loss_mode: $LOSS_MODE"
echo "Run dir:      $RUN_DIR"
echo "Resume:       mode=$RESUME_MODE path=${RESUME_FROM_PATH:-auto/latest-or-none}"

vpr_validate_launcher
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN: configuration validated; training was not started."
    exit 0
fi
if [ ! -d "$MODEL_PATH" ]; then echo "ERROR: Model not found at $MODEL_PATH" >&2; exit 1; fi
if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$PYTHON" ]; then echo "ERROR: Python not found: $PYTHON" >&2; exit 1; fi
if [ "$RESUME_MODE" = "resume_path" ] && [ -z "$RESUME_FROM_PATH" ]; then
    echo "ERROR: RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 1
fi
if [ -n "$RESUME_FROM_PATH" ] && [ ! -d "$RESUME_FROM_PATH" ]; then
    echo "ERROR: RESUME_FROM_PATH not found: $RESUME_FROM_PATH" >&2
    exit 1
fi
if ! "$PYTHON" -c "import gem" 2>/dev/null; then
    echo "ERROR: 'gem' not found in $PYTHON (pip install 'git+https://github.com/axon-rl/gem.git')" >&2
    exit 1
fi

"$PYTHON" "$VPR_GAMES_DIR/prepare_data.py" \
    --env-name vpr_minesweeper --train-size "$TRAIN_BATCH" --val-size "$VAL_BATCH" \
    --output-dir "$DATA_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
TENSORBOARD_DIR="$RUN_DIR/tensorboard" \
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name vpr_minesweeper \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESP" \
    data.filter_overlong_prompts=False \
    data.return_raw_chat=True \
    +data.dataloader_num_workers=0 \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO" \
    actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.actor.use_kl_loss="$USE_KL" \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    env.minesweeper.oracle_policy="$ORACLE_POLICY" \
    env.minesweeper.oracle_reward="$ORACLE_REWARD" \
    env.minesweeper.oracle_flag_reward="$ORACLE_FLAG_REWARD" \
    env.minesweeper.oracle_guess_reward="$ORACLE_GUESS_REWARD" \
    env.minesweeper.non_oracle_penalty="$NON_ORACLE_PENALTY" \
    env.minesweeper.non_oracle_flag_penalty="$NON_ORACLE_FLAG_PENALTY" \
    env.invalid_penalty="$INVALID_PENALTY" \
    env.minesweeper.reward_mode=oracle \
    env.minesweeper.mines=5 \
    env.game_action_format="$GAME_ACTION_FORMAT" \
    algorithm.vpr.outcome_reward_scale="$OUTCOME_REWARD_SCALE" \
    algorithm.vpr.state_group_advantage_mode="$STATE_GROUP_ADV_MODE" \
    algorithm.vpr.skip_update_equal_reward_threshold="$VPR_SKIP_UPDATE_EQUAL_REWARD_THRESHOLD" \
    env.seed=0 \
    env.rollout.n="$ROLLOUT_N" \
    env.rollout.mode="$ROLLOUT_MODE" \
    env.rollout.selection_mode="$SELECTION_MODE" \
    env.rollout.random_select_prob="$RANDOM_SELECT_PROB" \
    env.rollout.random_select_prob_schedule="'${RANDOM_SELECT_PROB_SCHEDULE}'" \
    algorithm.use_kl_in_reward=False \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=vpr_minesweeper \
    trainer.experiment_name="vpr_${TS}" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.logger=["console","tensorboard"] \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    hydra.run.dir="$RUN_DIR/hydra" \
    +ray_init.num_cpus="$RAY_CPUS" 2>&1 | tee "$LOG_FILE"

echo ""
echo "All outputs under: $RUN_DIR  (train.log / ckpt/ / tensorboard/ / hydra/)"
echo "  tensorboard --logdir $RUN_DIR/tensorboard"
echo "=== done ==="
