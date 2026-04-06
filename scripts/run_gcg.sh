#!/bin/bash
# ============================================================================
# SLURM Job: GCG Prefix Optimization
#
# Single concept mode:
#   sbatch scripts/run_gcg.sh
#   CONCEPT=spiders sbatch scripts/run_gcg.sh
#
# All concepts (single GPU, sequential):
#   ALL_CONCEPTS=1 sbatch scripts/run_gcg.sh
#
# All concepts (SLURM array, parallel across GPUs):
#   ALL_CONCEPTS=1 sbatch --array=0-4 scripts/run_gcg.sh
#
# Smoke test (3 concepts):
#   SMOKE_TEST=3 sbatch scripts/run_gcg.sh
# ============================================================================
#SBATCH --job-name=gcg_prefix
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.out

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_DIR="${DATA_DIR:-$(readlink -f "${PROJECT_DIR}/data" 2>/dev/null || echo "${PROJECT_DIR}/data")}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/prefix_optimization"
CONCEPT_CLASS="${CONCEPT_CLASS:-fears}"
SMOKE_TEST="${SMOKE_TEST:-0}"
ALL_CONCEPTS="${ALL_CONCEPTS:-0}"
CONCEPT="${CONCEPT:-}"

# --- Build Python command ---
ARGS=(
    --concept_class "${CONCEPT_CLASS}"
    --data_dir "${DATA_DIR}"
    --model_name llama_3.1_8b
    --output_dir "${OUTPUT_DIR}"
    --n_steps 200
    --gcg_topk 256
    --gcg_batch_size 64
    --log_every 10
    --seed 42
)

if [ -n "${CACHE_DIR:-}" ]; then
    ARGS+=(--cache_dir "${CACHE_DIR}")
fi

if [ "${ALL_CONCEPTS}" = "1" ]; then
    ARGS+=(--all_concepts)

    # SLURM array slicing
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        CONCEPT_FILE="${DATA_DIR}/concepts/${CONCEPT_CLASS}.txt"
        TOTAL_CONCEPTS=$(wc -l < "${CONCEPT_FILE}")
        N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
        TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
        PER_TASK=$(( (TOTAL_CONCEPTS + N_TASKS - 1) / N_TASKS ))
        START=$(( TASK_ID * PER_TASK ))
        END=$(( START + PER_TASK ))
        if [ ${END} -gt ${TOTAL_CONCEPTS} ]; then END=${TOTAL_CONCEPTS}; fi
        ARGS+=(--slice_start ${START} --slice_end ${END})
        echo "Array task ${TASK_ID}/${N_TASKS}: concepts [${START}, ${END})"
    fi
elif [ -n "${CONCEPT}" ]; then
    ARGS+=(--concept "${CONCEPT}")
else
    # Auto-detect first available concept from directions
    CONCEPT=$(ls "${DATA_DIR}/directions/" 2>/dev/null | grep "^rfm_" | head -1 | sed 's/rfm_//; s/_tokenidx_.*//')
    ARGS+=(--concept "${CONCEPT}")
fi

if [ "${SMOKE_TEST}" -gt 0 ]; then
    ARGS+=(--smoke_test "${SMOKE_TEST}")
fi

echo "============================================"
echo "  GCG Prefix Optimization"
echo "  Concept class: ${CONCEPT_CLASS}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: ${PYTHON}"
echo "  Data: ${DATA_DIR}"
echo "  Args: ${ARGS[*]}"
echo "  Start: $(date)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

${PYTHON} -u -m steering_fast.prefix_optimization.run_all "${ARGS[@]}" 2>&1

echo ""
echo "============================================"
echo "  Complete: $(date)"
echo "============================================"
