#!/bin/bash
# ============================================================================
# SLURM Job: GCG (Greedy Coordinate Gradient) prefix optimization
#
# v2: batched candidate evaluation, random position selection.
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
#SBATCH --time=02:00:00
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
echo "  GCG Prefix Optimization (v2: batched)"
echo "  Concept: ${CONCEPT}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: ${PYTHON}"
echo "  Data: ${DATA_DIR}"
echo "  Start: $(date)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

# ============================================================================
# Test 1: GCG single layer 16 -- 200 steps with batched eval
# ============================================================================
echo ""
echo ">>> GCG: Layer 16, top_k=256, batch=64, 200 steps"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method gcg \
    --layers 16 \
    --n_steps 200 \
    --loss_type cosine \
    --output_dir "${OUTPUT_DIR}/gcg_v2_layer16" \
    --log_every 10 \
    --seed 42 \
    2>&1

echo ""
echo ">>> GCG layer 16 complete: $(date)"

# ============================================================================
# Test 2: GCG all layers -- 200 steps
# ============================================================================
echo ""
echo ">>> GCG: All layers, top_k=256, batch=64, 200 steps"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method gcg \
    --layers all \
    --n_steps 200 \
    --loss_type cosine \
    --output_dir "${OUTPUT_DIR}/gcg_v2_all_layers" \
    --log_every 10 \
    --seed 42 \
    2>&1

echo ""
echo ">>> GCG all layers complete: $(date)"

echo ""
echo "============================================"
echo "  GCG tests complete: $(date)"
echo "  Results in: ${OUTPUT_DIR}"
echo "============================================"
