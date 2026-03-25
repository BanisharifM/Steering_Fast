#!/bin/bash
# ============================================================================
# SLURM Array Job: Split concepts across N GPUs for parallel processing
#
# Usage:
#   # 10 tasks, each processing 1/10 of concepts
#   sbatch --array=0-9 scripts/slurm_array.sh fears 0
#
#   # 5 tasks for stage 1
#   sbatch --array=0-4 scripts/slurm_array.sh fears 1
#
# Args:
#   $1 = concept class (fears, moods, personas, personalities, places)
#   $2 = stage number (0, 1, 2)
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

# --- Arguments ---
CONCEPT_CLASS="${1:-fears}"
STAGE="${2:-0}"

# --- Environment ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="/work/hdd/bdau/mbanisharifdehkordi/conda_envs/llm_steering"
PYTHON="${ENV_PATH}/bin/python"

export CACHE_DIR="${CACHE_DIR:-/work/hdd/bdau/mbanisharifdehkordi/model_cache}"
export HF_HOME="${HF_HOME:-/work/hdd/bdau/mbanisharifdehkordi/.hf_cache}"
export PYTHONUNBUFFERED=1

DATA_DIR="${PROJECT_DIR}/../attention_guided_steering/data"

# --- Compute slice for this array task ---
CONCEPT_FILE="${DATA_DIR}/concepts/${CONCEPT_CLASS}.txt"
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
        'paths.cache_dir': '${CACHE_DIR}',
        'paths.checkpoint_dir': '${PROJECT_DIR}/checkpoints',
        'paths.output_dir': '${PROJECT_DIR}/outputs',
    },
)

run_pipeline(cfg)
" 2>&1

echo ""
echo "============================================"
echo "  Task ${TASK_ID} complete: $(date)"
echo "============================================"
