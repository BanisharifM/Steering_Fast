#!/bin/bash
# ============================================================================
# SLURM Array Job: Split concepts across N GPUs for parallel processing
#
# Usage:
#   # 5 tasks (one per class), each on its own GPU
#   sbatch --array=0-4 scripts/slurm_array.sh
#
#   # Single class with 10 GPUs splitting concepts
#   CONCEPT_CLASS=fears STAGE=0 sbatch --array=0-9 scripts/slurm_array.sh
#
# Environment variables (set before sbatch or in env.sh):
#   CONCEPT_CLASS: fears, moods, personas, personalities, places
#   STAGE: 0, 1, 2 (which pipeline stage to run)
#   DATA_DIR: path to data directory
#   CACHE_DIR: path to model cache
# ============================================================================
#SBATCH --job-name=steer_array
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.out

# --- Dynamic path resolution ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

# Use environment variables with sensible defaults
# Python: search known conda env locations (conda may be above PARENT_DIR)
USER_HOME="$(dirname "${PARENT_DIR}")"
for _candidate in \
    "${PARENT_DIR}/conda_envs/llm_steering/bin/python" \
    "${USER_HOME}/conda_envs/llm_steering/bin/python" \
    "${HOME}/conda_envs/llm_steering/bin/python" \
    "${HOME}/.conda/envs/llm_steering/bin/python" \
    "$(which python3 2>/dev/null)" \
    "$(which python 2>/dev/null)"; do
    if [ -f "${_candidate}" ]; then
        PYTHON="${PYTHON:-${_candidate}}"
        break
    fi
done
unset _candidate USER_HOME
if [ -z "${PYTHON:-}" ] || [ ! -f "${PYTHON:-}" ]; then
    echo "ERROR: Python not found. Set PYTHON environment variable."
    exit 1
fi

export CACHE_DIR="${CACHE_DIR:-}"
export HF_HOME="${HF_HOME:-}"
export PYTHONUNBUFFERED=1

# --- Configuration ---
CONCEPT_CLASS="${CONCEPT_CLASS:-fears}"
STAGE="${STAGE:-0}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"

# --- Concept slicing ---
CONCEPT_FILE="${DATA_DIR}/concepts/${CONCEPT_CLASS}.txt"
if [ ! -f "${CONCEPT_FILE}" ]; then
    echo "ERROR: Concept file not found: ${CONCEPT_FILE}"
    exit 1
fi
TOTAL_CONCEPTS=$(wc -l < "${CONCEPT_FILE}")
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

PER_TASK=$(( (TOTAL_CONCEPTS + N_TASKS - 1) / N_TASKS ))
START=$(( TASK_ID * PER_TASK ))
END=$(( START + PER_TASK ))
if [ ${END} -gt ${TOTAL_CONCEPTS} ]; then
    END=${TOTAL_CONCEPTS}
fi

echo "============================================"
echo "  Array Job: ${CONCEPT_CLASS} stage ${STAGE}"
echo "  Task ${TASK_ID}/${N_TASKS}: concepts [${START}, ${END})"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: ${PYTHON}"
echo "  Data: ${DATA_DIR}"
echo "  Start: $(date)"
echo "============================================"

cd "${PROJECT_DIR}"

${PYTHON} -u -c "
import logging, torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')

from steering_fast.utils import load_config
from steering_fast.pipeline.runner import run_pipeline

cfg = load_config(
    data='${CONCEPT_CLASS}',
    overrides={
        'stages': [${STAGE}],
        'slicing.enabled': True,
        'slicing.start': ${START},
        'slicing.end': ${END},
        'timing.enabled': True,
        'paths.data_dir': '${DATA_DIR}',
        'paths.cache_dir': '${CACHE_DIR}' if '${CACHE_DIR}' else None,
        'paths.checkpoint_dir': '${PROJECT_DIR}/checkpoints/${CONCEPT_CLASS}_${STAGE}_${TASK_ID}',
        'paths.output_dir': '${PROJECT_DIR}/outputs/${CONCEPT_CLASS}',
    },
)

run_pipeline(cfg)
" 2>&1

echo ""
echo "============================================"
echo "  Task ${TASK_ID} complete: $(date)"
echo "============================================"
