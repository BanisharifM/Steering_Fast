#!/bin/bash
# ============================================================================
# SLURM Job: GCG Prefix Optimization (single model load, all experiments)
#
# Loads model ONCE, runs GCG with multiple configs sequentially.
# ~7 min model load + ~5 min per experiment = ~25 min total (vs ~50 min before)
#
# Usage:
#   sbatch scripts/run_gcg.sh
#   CONCEPT=spiders sbatch scripts/run_gcg.sh
# ============================================================================
#SBATCH --job-name=gcg_prefix
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.out

# --- Path resolution ---
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/work/hdd/bdau/mbanisharifdehkordi/LLM_Steering/steering_fast}"
ENV_SCRIPT="${PROJECT_DIR}/scripts/env.sh"
if [ -f "${ENV_SCRIPT}" ]; then source "${ENV_SCRIPT}"; fi

if [ -z "${PYTHON:-}" ] || [ ! -f "${PYTHON:-}" ]; then
    PARENT_DIR="$(dirname "${PROJECT_DIR}")"
    USER_HOME="$(dirname "${PARENT_DIR}")"
    for candidate in \
        "${USER_HOME}/conda_envs/llm_steering/bin/python" \
        "${PARENT_DIR}/conda_envs/llm_steering/bin/python" \
        "$(which python3 2>/dev/null)"; do
        if [ -f "${candidate}" ]; then PYTHON="${candidate}"; break; fi
    done
fi

export PYTHONUNBUFFERED=1
DATA_DIR="${DATA_DIR:-$(readlink -f "${PROJECT_DIR}/data" 2>/dev/null || echo "${PROJECT_DIR}/data")}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/prefix_optimization"

CONCEPT="${CONCEPT:-}"
if [ -z "${CONCEPT}" ]; then
    CONCEPT=$(ls "${DATA_DIR}/directions/" 2>/dev/null | grep "^rfm_" | head -1 | sed 's/rfm_//; s/_tokenidx_.*//')
fi

echo "============================================"
echo "  GCG Prefix Optimization (single load)"
echo "  Concept: ${CONCEPT}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: ${PYTHON}"
echo "  Data: ${DATA_DIR}"
echo "  Start: $(date)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

# Single process: load model once, run all experiments
${PYTHON} -u -m steering_fast.prefix_optimization.run_all \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --model_name llama_3.1_8b \
    --output_dir "${OUTPUT_DIR}" \
    --n_steps 200 \
    --gcg_topk 256 \
    --gcg_batch_size 64 \
    --log_every 10 \
    --seed 42 \
    2>&1

echo ""
echo "============================================"
echo "  Complete: $(date)"
echo "============================================"
