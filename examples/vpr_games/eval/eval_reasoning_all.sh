#!/usr/bin/env bash
# Evaluate an arbitrary model manifest on the standard reasoning suite.
set -euo pipefail

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$@"
    else
        shasum -a 256 "$@"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%S)"

MODEL_MANIFEST="${MODEL_MANIFEST:-}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/eval_reasoning_$RUN_ID}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$REPO_ROOT/runs/eval_model_cache}"
PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"
EVALSCOPE_ROOT="${EVALSCOPE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/eval-scope}"
EVAL_PYTHON="${EVAL_PYTHON:-$EVALSCOPE_ROOT/.venv/bin/python}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/runs/evalscope_cache/datasets}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
GPU_BUSY_MEMORY_MIB="${GPU_BUSY_MEMORY_MIB:-2048}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
PORT="${PORT:-8801}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
EVAL_SEED="${EVAL_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

SERVER_PID=""
TEMP_MODEL_DIR=""
PREPARED_MODEL_PATH=""
PREPARED_SOURCE_IDENTITY=""

usage() {
    cat <<'EOF'
Usage:
  MODEL_MANIFEST=/path/to/models.tsv \
    bash examples/vpr_games/eval/eval_reasoning_all.sh

Manifest columns (tab-separated):
  model_id    source_type    source_path    benchmarks

source_type:
  hf          Hugging Face model directory.
  verl_fsdp   VERL global_step_* directory containing actor FSDP shards.

benchmarks:
  Comma-separated subset of:
  gsm8k,math_500,aime24,aime25,gpqa_diamond,bbh,mmlu_pro

The default protocol matches the existing VPR reasoning results:
  - 16K response / 20K model context
  - temperature=0.6, top_p=0.95, top_k=20, min_p=0
  - enable_thinking=true and seed=42
  - repeats: GSM8K 3, MATH-500/GPQA 5, AIME24/AIME25 32,
    BBH/MMLU-Pro 1

FSDP checkpoints are merged once under runs/eval_model_cache. Reusing RUN_DIR
skips benchmark jobs that already have a .done marker.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

abspath() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$REPO_ROOT/$path"
    fi
}

model_file_metadata() {
    local directory="$1" file name size mtime
    find "$directory" -maxdepth 1 -type f -print | LC_ALL=C sort | while IFS= read -r file; do
        name="$(basename "$file")"
        case "$name" in
            model* | *.bin | config.json | generation_config.json | tokenizer* | special_tokens_map.json | chat_template* | merges.txt | vocab.json | added_tokens.json) ;;
            *) continue ;;
        esac
        size="$(wc -c < "$file" | tr -d ' ')"
        mtime="$(stat -f '%m' "$file" 2>/dev/null || stat -c '%Y' "$file")"
        printf '%s|%s|%s\n' "$name" "$size" "$mtime"
    done
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

cleanup() {
    stop_server
    if [[ -n "$TEMP_MODEL_DIR" && -d "$TEMP_MODEL_DIR" ]]; then
        rm -rf "$TEMP_MODEL_DIR"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

[[ -n "$MODEL_MANIFEST" ]] || {
    usage >&2
    die "MODEL_MANIFEST is required"
}

RUN_DIR="$(abspath "$RUN_DIR")"
MODEL_CACHE_ROOT="$(abspath "$MODEL_CACHE_ROOT")"
MODEL_MANIFEST="$(abspath "$MODEL_MANIFEST")"
DATASET_DIR="$(abspath "$DATASET_DIR")"

[[ -f "$MODEL_MANIFEST" ]] || die "model manifest not found: $MODEL_MANIFEST"
if [[ "$DRY_RUN" != 1 ]]; then
    command -v "$PYTHON" >/dev/null 2>&1 || [[ -x "$PYTHON" ]] || die "training Python not found: $PYTHON"
    command -v "$EVAL_PYTHON" >/dev/null 2>&1 || [[ -x "$EVAL_PYTHON" ]] || die "EvalScope Python not found: $EVAL_PYTHON"
fi
[[ "$N_GPUS" =~ ^[1-9][0-9]*$ ]] || die "N_GPUS must be positive"
[[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || die "TP_SIZE must be positive"
[[ "$DP_SIZE" =~ ^[1-9][0-9]*$ ]] || die "DP_SIZE must be positive"
(( TP_SIZE * DP_SIZE == N_GPUS )) || die "TP_SIZE * DP_SIZE must equal N_GPUS"
(( MAX_TOKENS < MAX_MODEL_LEN )) || die "MAX_TOKENS must be less than MAX_MODEL_LEN"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
(( ${#GPU_IDS[@]} == N_GPUS )) ||
    die "CUDA_VISIBLE_DEVICES has ${#GPU_IDS[@]} devices, expected $N_GPUS"

selected_gpus_are_free() {
    local gpu_id used pids
    for gpu_id in "${GPU_IDS[@]}"; do
        [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "GPU IDs must be numeric: $gpu_id"
        pids="$(nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
        [[ ! "$pids" =~ [0-9] ]] || return 1
        used="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        [[ "$used" =~ ^[0-9]+$ ]] || die "cannot query memory for GPU $gpu_id"
        (( used < GPU_BUSY_MEMORY_MIB )) || return 1
    done
}

wait_for_selected_gpus() {
    [[ "$DRY_RUN" == 1 || "$WAIT_FOR_FREE_GPUS" == 0 ]] && return
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required for GPU waiting"
    while ! selected_gpus_are_free; do
        log "Selected GPUs are busy; waiting $GPU_POLL_SECONDS seconds"
        sleep "$GPU_POLL_SECONDS"
    done
    log "Selected GPUs are free: $CUDA_VISIBLE_DEVICES"
}

validate_source() {
    local model_id="$1"
    local source_type="$2"
    local source_path="$3"
    [[ -d "$source_path" ]] || die "source for '$model_id' not found: $source_path"

    if [[ "$source_type" == hf ]]; then
        [[ -f "$source_path/config.json" ]] || die "missing config.json for '$model_id'"
        compgen -G "$source_path/*.safetensors" >/dev/null ||
            compgen -G "$source_path/*.bin" >/dev/null ||
            die "missing HF weights for '$model_id'"
        return
    fi

    [[ "$source_type" == verl_fsdp ]] ||
        die "unsupported source_type '$source_type' for '$model_id'"
    local actor_dir="$source_path/actor"
    [[ -f "$actor_dir/config.json" ]] || die "missing actor config for '$model_id'"
    local rank_zero world_size shard_count
    rank_zero="$(find "$actor_dir" -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' -print -quit)"
    [[ -n "$rank_zero" ]] || die "missing rank-zero FSDP shard for '$model_id'"
    world_size="$(basename "$rank_zero" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
    [[ "$world_size" =~ ^[0-9]+$ ]] || die "cannot parse FSDP world size for '$model_id'"
    shard_count="$(find "$actor_dir" -maxdepth 1 -type f -name "model_world_size_${world_size}_rank_*.pt" | wc -l)"
    [[ "$shard_count" -eq "$world_size" ]] ||
        die "checkpoint '$model_id' has $shard_count/$world_size model shards"
}

source_identity() {
    local source_type="$1"
    local source_path="$2"
    local scan_path="$source_path"
    [[ "$source_type" != verl_fsdp ]] || scan_path="$source_path/actor"
    {
        printf 'source_type=%s\n' "$source_type"
        printf 'source_path=%s\n' "$(realpath "$source_path")"
        [[ "$source_type" != verl_fsdp ]] ||
            printf 'merger=%s\n' "$(sha256 "$REPO_ROOT/scripts/model_merger.py")"
        model_file_metadata "$scan_path"
    } | sha256 | awk '{print $1}'
}

prepare_model() {
    local model_id="$1"
    local source_type="$2"
    local source_path="$3"
    PREPARED_SOURCE_IDENTITY="$(source_identity "$source_type" "$source_path")"

    if [[ "$DRY_RUN" == 1 ]]; then
        PREPARED_MODEL_PATH="$source_path"
        return
    fi

    if [[ "$source_type" == hf ]]; then
        PREPARED_MODEL_PATH="$source_path"
        return
    fi

    local cache_dir="$MODEL_CACHE_ROOT/$model_id/$PREPARED_SOURCE_IDENTITY"
    if [[ -f "$cache_dir/.merge_complete" && -f "$cache_dir/config.json" ]] &&
        compgen -G "$cache_dir/*.safetensors" >/dev/null; then
        PREPARED_MODEL_PATH="$cache_dir"
        return
    fi
    [[ ! -e "$cache_dir" ]] || die "incomplete model cache exists: $cache_dir"

    TEMP_MODEL_DIR="$cache_dir.tmp.$$"
    mkdir -p "$(dirname "$cache_dir")" "$TEMP_MODEL_DIR"
    log "Merging $model_id into $cache_dir"
    "$PYTHON" "$REPO_ROOT/scripts/model_merger.py" merge \
        --backend fsdp \
        --local_dir "$source_path/actor" \
        --target_dir "$TEMP_MODEL_DIR"
    [[ -f "$TEMP_MODEL_DIR/config.json" ]] || die "merge produced no config.json"
    compgen -G "$TEMP_MODEL_DIR/*.safetensors" >/dev/null ||
        die "merge produced no safetensors"
    {
        printf 'model_id=%s\n' "$model_id"
        printf 'source=%s\n' "$(realpath "$source_path")"
        printf 'source_identity=%s\n' "$PREPARED_SOURCE_IDENTITY"
        printf 'merged_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$TEMP_MODEL_DIR/.merge_complete"
    mv "$TEMP_MODEL_DIR" "$cache_dir"
    TEMP_MODEL_DIR=""
    PREPARED_MODEL_PATH="$cache_dir"
}

write_protocol() {
    mkdir -p "$RUN_DIR/results" "$RUN_DIR/server_logs"
    local candidate="$RUN_DIR/protocol.env.new"
    {
        echo "PROTOCOL_VERSION=1"
        printf 'MODEL_MANIFEST=%s\n' "$MODEL_MANIFEST"
        printf 'GPU=cuda:%s,n_gpus:%s,tp:%s,dp:%s,memory_util:%s\n' \
            "$CUDA_VISIBLE_DEVICES" "$N_GPUS" "$TP_SIZE" "$DP_SIZE" "$GPU_MEM_UTIL"
        printf 'MODEL_LIMITS=max_tokens:%s,max_model_len:%s,max_batched_tokens:%s\n' \
            "$MAX_TOKENS" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS"
        printf 'SAMPLING=temperature:%s,top_p:%s,top_k:%s,min_p:%s,enable_thinking:true\n' \
            "$TEMPERATURE" "$TOP_P" "$TOP_K" "$MIN_P"
        printf 'EVAL=seed:%s,batch_size:%s,repeats:gsm8k-3_math500-5_aime24-32_aime25-32_gpqa-5_bbh-1_mmlupro-1\n' \
            "$EVAL_SEED" "$EVAL_BATCH_SIZE"
        sha256 "$MODEL_MANIFEST"
    } > "$candidate"
    if [[ -f "$RUN_DIR/protocol.env" ]]; then
        cmp -s "$RUN_DIR/protocol.env" "$candidate" || {
            diff -u "$RUN_DIR/protocol.env" "$candidate" >&2 || true
            rm -f "$candidate"
            die "protocol differs from existing RUN_DIR; use a new RUN_DIR"
        }
        rm -f "$candidate"
    else
        mv "$candidate" "$RUN_DIR/protocol.env"
    fi
}

start_server() {
    local model_id="$1"
    local model_path="$2"
    local log_path="$RUN_DIR/server_logs/$model_id.log"

    if [[ "$DRY_RUN" == 1 ]]; then
        log "DRY RUN: would start vLLM for $model_id from $model_path"
        return
    fi
    log "Starting vLLM for $model_id"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --host 127.0.0.1 \
        --port "$PORT" \
        --model "$model_path" \
        --served-model-name "$model_id" \
        --trust-remote-code \
        --dtype bfloat16 \
        --seed 0 \
        --max-model-len "$MAX_MODEL_LEN" \
        --generation-config vllm \
        --reasoning-parser qwen3 \
        --tensor-parallel-size "$TP_SIZE" \
        --data-parallel-size "$DP_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        > "$log_path" 2>&1 &
    SERVER_PID=$!

    for _ in $(seq 1 180); do
        kill -0 "$SERVER_PID" 2>/dev/null ||
            die "vLLM for '$model_id' exited before becoming ready; see $log_path"
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 &&
            curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -Fq "$model_id"; then
            return
        fi
        sleep 5
    done
    die "vLLM for '$model_id' did not become ready; see $log_path"
}

benchmark_settings() {
    case "$1" in
        gsm8k) echo "3 mean_and_pass_at_k" ;;
        math_500) echo "5 mean_and_pass_at_k" ;;
        aime24 | aime25) echo "32 mean_and_pass_at_k" ;;
        gpqa_diamond) echo "5 mean_and_pass_at_k" ;;
        bbh | mmlu_pro) echo "1 mean" ;;
        *) die "unsupported benchmark: $1" ;;
    esac
}

run_benchmark() {
    local model_id="$1"
    local benchmark="$2"
    local repeats aggregation
    read -r repeats aggregation <<< "$(benchmark_settings "$benchmark")"

    local output_dir="$RUN_DIR/results/$model_id/$benchmark"
    if [[ -f "$output_dir/.done" ]]; then
        log "Skipping completed $model_id/$benchmark"
        return
    fi
    mkdir -p "$output_dir"

    local dataset_args generation_config
    dataset_args="$(printf '{"%s":{"few_shot_num":0,"shuffle":false,"filters":{"remove_until":"</think>"},"aggregation":"%s"}}' "$benchmark" "$aggregation")"
    generation_config="$(printf '{"max_tokens":%s,"temperature":%s,"top_p":%s,"top_k":%s,"n":1,"stream":true,"extra_body":{"min_p":%s,"chat_template_kwargs":{"enable_thinking":true}}}' \
        "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P" "$TOP_K" "$MIN_P")"

    local -a cmd=(
        "$EVAL_PYTHON" -m evalscope.cli.cli eval
        --model "$model_id"
        --model-id "$model_id"
        --api-url "http://127.0.0.1:$PORT/v1"
        --api-key EMPTY
        --eval-type openai_api
        --datasets "$benchmark"
        --dataset-dir "$DATASET_DIR"
        --dataset-hub modelscope
        --dataset-args "$dataset_args"
        --generation-config "$generation_config"
        --eval-batch-size "$EVAL_BATCH_SIZE"
        --repeats "$repeats"
        --seed "$EVAL_SEED"
        --timeout 60000
        --stream
        --use-cache "$output_dir"
        --work-dir "$output_dir"
        --no-timestamp
    )
    printf '%q ' "${cmd[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    if [[ "$DRY_RUN" == 1 ]]; then
        log "DRY RUN: wrote command for $model_id/$benchmark"
        return
    fi
    log "Evaluating $model_id/$benchmark"
    "${cmd[@]}" 2>&1 | tee "$output_dir/evalscope.log"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$output_dir/.done"
}

write_protocol
if [[ "$DRY_RUN" != 1 ]]; then
    exec 9>"$RUN_DIR/.eval.lock"
    flock -n 9 || die "another evaluator is using $RUN_DIR"
fi
wait_for_selected_gpus

while IFS=$'\t' read -r model_id source_type source_path benchmarks extra; do
    [[ -n "$model_id" && "${model_id:0:1}" != "#" ]] || continue
    [[ -z "${extra:-}" ]] || die "manifest row has more than four columns: $model_id"
    [[ "$model_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid model_id: $model_id"
    source_path="$(abspath "$source_path")"
    validate_source "$model_id" "$source_type" "$source_path"
    prepare_model "$model_id" "$source_type" "$source_path"

    mkdir -p "$RUN_DIR/results/$model_id"
    {
        printf 'model_id=%s\n' "$model_id"
        printf 'source_type=%s\n' "$source_type"
        printf 'source_path=%s\n' "$(realpath "$source_path")"
        printf 'prepared_path=%s\n' "$(realpath "$PREPARED_MODEL_PATH")"
        printf 'source_identity=%s\n' "$PREPARED_SOURCE_IDENTITY"
        printf 'benchmarks=%s\n' "$benchmarks"
    } > "$RUN_DIR/results/$model_id/model.env"

    start_server "$model_id" "$PREPARED_MODEL_PATH"
    IFS=',' read -r -a benchmark_list <<< "$benchmarks"
    for benchmark in "${benchmark_list[@]}"; do
        run_benchmark "$model_id" "$benchmark"
    done
    [[ "$DRY_RUN" == 1 ]] || stop_server
done < "$MODEL_MANIFEST"

log "Reasoning evaluation complete: $RUN_DIR"
