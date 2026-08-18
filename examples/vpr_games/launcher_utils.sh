#!/usr/bin/env bash

# Shared validation for paper-facing VPR game launchers. The caller defines the
# variables below; this helper intentionally performs no filesystem writes.
vpr_validate_launcher() {
    local name value gpu_count
    for name in TRAIN_STEPS TRAIN_BATCH ROLLOUT_N PPO_MINI_BATCH N_GPUS TP_SIZE; do
        value="${!name:-}"
        if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: $name must be a positive integer (got '$value')" >&2
            return 1
        fi
    done
    if (( N_GPUS % TP_SIZE != 0 )); then
        echo "ERROR: N_GPUS must be divisible by TP_SIZE" >&2
        return 1
    fi
    if (( (TRAIN_BATCH * ROLLOUT_N) % PPO_MINI_BATCH != 0 )); then
        echo "ERROR: TRAIN_BATCH*ROLLOUT_N must be divisible by PPO_MINI_BATCH" >&2
        return 1
    fi
    IFS=',' read -r -a _vpr_gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-}"
    gpu_count="${#_vpr_gpu_ids[@]}"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && (( gpu_count != N_GPUS )); then
        echo "ERROR: CUDA_VISIBLE_DEVICES exposes $gpu_count devices but N_GPUS=$N_GPUS" >&2
        return 1
    fi
    case "${DRY_RUN:-0}" in
        0 | 1) ;;
        *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; return 1 ;;
    esac
    if [[ -z "${MODEL_PATH:-}" ]]; then
        echo "ERROR: MODEL_PATH is required" >&2
        return 1
    fi
    if [[ -n "${MAX_PROMPT:-}" && -n "${MAX_MODEL_LEN:-}" ]] &&
        (( MAX_PROMPT + MAX_RESP > MAX_MODEL_LEN )); then
        echo "ERROR: MAX_PROMPT + MAX_RESP must not exceed MAX_MODEL_LEN" >&2
        return 1
    fi
}
