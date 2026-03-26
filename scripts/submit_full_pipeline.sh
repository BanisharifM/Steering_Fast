#!/bin/bash
# ============================================================================
# Submit full pipeline with SLURM array jobs for parallel concept processing
#
# Usage:
#   bash scripts/submit_full_pipeline.sh [N_GPUS] [CONCEPT_CLASSES...]
#   bash scripts/submit_full_pipeline.sh 10 fears moods personas personalities places
#   bash scripts/submit_full_pipeline.sh 5 fears  # single class with 5 GPUs
#
# Environment: source scripts/env.sh or set PYTHON, CACHE_DIR, etc.
# ============================================================================
set -euo pipefail

# Source shared environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

N_GPUS="${1:-5}"
shift || true
CLASSES=("${@:-fears moods personas personalities places}")
if [ ${#CLASSES[@]} -eq 0 ]; then
    CLASSES=(fears moods personas personalities places)
fi

echo "============================================"
echo "  Full Pipeline Submission"
echo "  GPUs per class: ${N_GPUS}"
echo "  Classes: ${CLASSES[*]}"
echo "  Data: ${STEERING_DATA_DIR}"
echo "  Python: ${PYTHON}"
echo "============================================"

mkdir -p "${PROJECT_DIR}/../logs/slurm" "${PROJECT_DIR}/checkpoints" "${PROJECT_DIR}/outputs"

ARRAY_MAX=$((N_GPUS - 1))

for CLASS in "${CLASSES[@]}"; do
    echo ""
    echo "--- ${CLASS} ---"

    # Stage 0: attention extraction (GPU, parallel across concepts)
    JOB0=$(CONCEPT_CLASS="${CLASS}" STAGE=0 DATA_DIR="${STEERING_DATA_DIR}" \
        sbatch --parsable \
        --array=0-${ARRAY_MAX} \
        --job-name="s0_${CLASS}" \
        --export=ALL \
        "${SCRIPT_DIR}/slurm_array.sh" 2>&1)
    echo "  Stage 0: array job ${JOB0} (${N_GPUS} tasks)"

    # Stage 1: direction training (GPU, parallel, depends on stage 0)
    JOB1=$(CONCEPT_CLASS="${CLASS}" STAGE=1 DATA_DIR="${STEERING_DATA_DIR}" \
        sbatch --parsable \
        --array=0-${ARRAY_MAX} \
        --dependency=afterok:${JOB0} \
        --job-name="s1_${CLASS}" \
        --export=ALL \
        "${SCRIPT_DIR}/slurm_array.sh" 2>&1)
    echo "  Stage 1: array job ${JOB1} (${N_GPUS} tasks)"

    # Stage 2: generation (GPU, one array per version, depends on stage 1)
    JOB2_IDS=""
    for VERSION in 1 2 3 4 5; do
        JOB2=$(CONCEPT_CLASS="${CLASS}" STAGE=2 DATA_DIR="${STEERING_DATA_DIR}" \
            sbatch --parsable \
            --array=0-${ARRAY_MAX} \
            --dependency=afterok:${JOB1} \
            --job-name="s2_${CLASS}_v${VERSION}" \
            --export=ALL \
            "${SCRIPT_DIR}/slurm_array.sh" 2>&1)
        JOB2_IDS="${JOB2_IDS:+${JOB2_IDS},}afterok:${JOB2}"
    done
    echo "  Stage 2: 5 version array jobs submitted"
done

echo ""
echo "============================================"
echo "All jobs submitted. Monitor: squeue -u \$USER"
echo "============================================"
