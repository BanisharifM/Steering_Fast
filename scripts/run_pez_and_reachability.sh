#!/bin/bash
# ============================================================================
# SLURM Job: PEZ + Reachability Analysis
#
# Runs PEZ (projected gradient descent) for discrete prefix optimization
# and Jacobian reachability analysis across all layers. Loads model once.
#
# Usage:
#   sbatch scripts/run_pez_and_reachability.sh
#   CONCEPT=spiders sbatch scripts/run_pez_and_reachability.sh
# ============================================================================
#SBATCH --job-name=pez_reach
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
if [ -f "${ENV_SCRIPT}" ]; then
    source "${ENV_SCRIPT}"
fi

# Fallback Python detection
if [ -z "${PYTHON:-}" ] || [ ! -f "${PYTHON:-}" ]; then
    PARENT_DIR="$(dirname "${PROJECT_DIR}")"
    USER_HOME="$(dirname "${PARENT_DIR}")"
    for candidate in \
        "${USER_HOME}/conda_envs/llm_steering/bin/python" \
        "${PARENT_DIR}/conda_envs/llm_steering/bin/python" \
        "$(which python3 2>/dev/null)"; do
        if [ -f "${candidate}" ]; then
            PYTHON="${candidate}"
            break
        fi
    done
fi

export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-$(readlink -f "${PROJECT_DIR}/data" 2>/dev/null || echo "${PROJECT_DIR}/data")}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/prefix_optimization"

# Get concept from env or use first available
CONCEPT="${CONCEPT:-}"
if [ -z "${CONCEPT}" ]; then
    CONCEPT=$(ls "${DATA_DIR}/directions/" 2>/dev/null | grep "^rfm_" | head -1 | sed 's/rfm_//; s/_tokenidx_.*//')
fi

echo "============================================"
echo "  PEZ + Reachability Analysis"
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
# Test 1: PEZ on single layer (layer 16) -- verify discrete recovery works
# ============================================================================
echo ""
echo ">>> PEZ: Single layer 16, cosine loss, 500 steps"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method pez \
    --layers 16 \
    --prefix_length 10 \
    --n_steps 500 \
    --lr 0.1 \
    --loss_type cosine \
    --init_strategy concept_name \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/pez_layer16" \
    --log_every 50 \
    --seed 42 \
    2>&1

echo ""
echo ">>> PEZ layer 16 complete: $(date)"

# ============================================================================
# Test 2: PEZ on all layers
# ============================================================================
echo ""
echo ">>> PEZ: All layers, cosine loss, 500 steps"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method pez \
    --layers all \
    --prefix_length 10 \
    --n_steps 500 \
    --lr 0.1 \
    --loss_type cosine \
    --init_strategy concept_name \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/pez_all_layers" \
    --log_every 50 \
    --seed 42 \
    2>&1

echo ""
echo ">>> PEZ all layers complete: $(date)"

# ============================================================================
# Test 3: Jacobian reachability across ALL layers
# ============================================================================
echo ""
echo ">>> Jacobian reachability: All layers"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method jacobian \
    --layers all \
    --prefix_length 10 \
    --jacobian_rank 32 \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/reachability_all_layers" \
    --seed 42 \
    2>&1

echo ""
echo ">>> Reachability complete: $(date)"

# ============================================================================
# Test 4: PEZ with angular loss (avoids quadratic slowdown)
# ============================================================================
echo ""
echo ">>> PEZ: Layer 16, angular loss"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method pez \
    --layers 16 \
    --prefix_length 10 \
    --n_steps 500 \
    --lr 0.1 \
    --loss_type angular \
    --init_strategy concept_name \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/pez_angular_layer16" \
    --log_every 50 \
    --seed 42 \
    2>&1

echo ""
echo ">>> PEZ angular complete: $(date)"

echo ""
echo "============================================"
echo "  All tests complete: $(date)"
echo "  Results in: ${OUTPUT_DIR}"
echo "============================================"
