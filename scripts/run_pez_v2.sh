#!/bin/bash
# ============================================================================
# SLURM Job: PEZ v2 -- Corrected prefix optimization
#
# Fixes: chat template wrapping, per-layer token position (max_attn_per_layer),
# proper prefix boundary detection, comparison with hand-crafted baseline.
#
# Usage:
#   sbatch scripts/run_pez_v2.sh
#   CONCEPT=spiders sbatch scripts/run_pez_v2.sh
# ============================================================================
#SBATCH --job-name=pez_v2
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

# Get concept
CONCEPT="${CONCEPT:-}"
if [ -z "${CONCEPT}" ]; then
    CONCEPT=$(ls "${DATA_DIR}/directions/" 2>/dev/null | grep "^rfm_" | head -1 | sed 's/rfm_//; s/_tokenidx_.*//')
fi

# Verify attention files exist (needed for layer_to_token)
ATTN_FILE="${DATA_DIR}/attention_to_prompt/attentions_meanhead_llama_3.1_8b_${CONCEPT}_paired_statements.npy"
if [ ! -f "${ATTN_FILE}" ]; then
    echo "WARNING: Attention file not found: ${ATTN_FILE}"
    echo "PEZ v2 needs pre-computed attention files (stage 0 output)."
fi

echo "============================================"
echo "  PEZ v2: Corrected Prefix Optimization"
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
# Test 1: PEZ v2 single layer 16 (cosine loss)
# ============================================================================
echo ""
echo ">>> PEZ v2: Layer 16, cosine, 500 steps, LR=0.1"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method pez_v2 \
    --layers 16 \
    --n_steps 500 \
    --lr 0.1 \
    --loss_type cosine \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/pez_v2_layer16" \
    --log_every 50 \
    --seed 42 \
    2>&1

echo ""
echo ">>> PEZ v2 layer 16 complete: $(date)"

# ============================================================================
# Test 2: PEZ v2 all layers
# ============================================================================
echo ""
echo ">>> PEZ v2: All layers, cosine, 500 steps"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR:-}" \
    --method pez_v2 \
    --layers all \
    --n_steps 500 \
    --lr 0.1 \
    --loss_type cosine \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/pez_v2_all_layers" \
    --log_every 50 \
    --seed 42 \
    2>&1

echo ""
echo ">>> PEZ v2 all layers complete: $(date)"

# ============================================================================
# Test 3: PEZ v2 with different LRs
# ============================================================================
echo ""
echo ">>> PEZ v2: LR sweep (layer 16)"
echo "-----------------------------------------------------------"
for LR in 0.01 0.05 0.1 0.3; do
    echo "  ... LR=${LR}"
    ${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
        --concept "${CONCEPT}" \
        --concept_class fears \
        --data_dir "${DATA_DIR}" \
        --cache_dir "${CACHE_DIR:-}" \
        --method pez_v2 \
        --layers 16 \
        --n_steps 300 \
        --lr ${LR} \
        --loss_type cosine \
        --output_dir "${OUTPUT_DIR}/pez_v2_lr_${LR}" \
        --log_every 100 \
        --seed 42 \
        2>&1
done

echo ""
echo ">>> LR sweep complete: $(date)"

echo ""
echo "============================================"
echo "  All PEZ v2 tests complete: $(date)"
echo "  Results in: ${OUTPUT_DIR}"
echo "============================================"
