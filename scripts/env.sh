#!/bin/bash
# ============================================================================
# Shared environment configuration for steering_fast SLURM job scripts.
# Source this at the top of every job script:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# To use on a different cluster:
#   1. Copy this file
#   2. Edit ENV_PATH, CACHE_DIR, HF_HOME, ACCOUNT, PARTITION
#   3. All job scripts will automatically use the new paths
# ============================================================================

# --- Auto-detect paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PARENT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

# --- Python environment (EDIT FOR YOUR CLUSTER) ---
# Option 1: Set ENV_PATH before sourcing this file
# Option 2: Set PYTHON directly
# Option 3: Let it auto-detect from common locations
if [ -z "${PYTHON:-}" ]; then
    if [ -n "${ENV_PATH:-}" ] && [ -f "${ENV_PATH}/bin/python" ]; then
        export PYTHON="${ENV_PATH}/bin/python"
    else
        # Auto-detect: look for conda env in common locations
        # Check multiple parent levels since conda env may be above the project tree
        USER_HOME="$(dirname "${PARENT_DIR}")"
        for candidate in \
            "${PARENT_DIR}/conda_envs/llm_steering/bin/python" \
            "${USER_HOME}/conda_envs/llm_steering/bin/python" \
            "${HOME}/conda_envs/llm_steering/bin/python" \
            "${HOME}/.conda/envs/llm_steering/bin/python" \
            "$(which python3 2>/dev/null)" \
            "$(which python 2>/dev/null)"; do
            if [ -f "${candidate}" ]; then
                export PYTHON="${candidate}"
                break
            fi
        done
    fi
fi

# --- Model and data paths ---
export CACHE_DIR="${CACHE_DIR:-}"
export HF_HOME="${HF_HOME:-}"
export STEERING_DATA_DIR="${STEERING_DATA_DIR:-${PROJECT_DIR}/data}"
export PYTHONUNBUFFERED=1

# --- SLURM account (EDIT FOR YOUR CLUSTER) ---
# These are used by submit_full_pipeline.sh, not by individual job scripts
# (individual scripts have #SBATCH directives which can't use variables)
export SLURM_ACCOUNT_GPU="${SLURM_ACCOUNT_GPU:-bdau-delta-gpu}"
export SLURM_ACCOUNT_CPU="${SLURM_ACCOUNT_CPU:-bdau-delta-cpu}"
export SLURM_PARTITION_GPU="${SLURM_PARTITION_GPU:-gpuA100x4}"
export SLURM_PARTITION_CPU="${SLURM_PARTITION_CPU:-cpu}"

# --- Validation ---
if [ -z "${PYTHON}" ] || [ ! -f "${PYTHON}" ]; then
    echo "WARNING: Python not found. Set ENV_PATH or PYTHON environment variable."
    echo "  Tried: ${PYTHON:-'(none)'}"
fi
