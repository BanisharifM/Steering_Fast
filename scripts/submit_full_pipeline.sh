#!/bin/bash
# ============================================================================
# Submit full pipeline with SLURM array jobs for parallel concept processing
#
# Usage:
#   bash scripts/submit_full_pipeline.sh [N_GPUS] [CONCEPT_CLASSES...]
#   bash scripts/submit_full_pipeline.sh 10 fears moods personas personalities places
#   bash scripts/submit_full_pipeline.sh 5 fears  # single class with 5 GPUs
# ============================================================================
set -euo pipefail

N_GPUS="${1:-10}"
shift || true
CLASSES=("${@:-fears moods personas personalities places}")
if [ ${#CLASSES[@]} -eq 0 ]; then
    CLASSES=(fears moods personas personalities places)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================"
echo "  Full Pipeline: ${N_GPUS} GPUs per class"
echo "  Classes: ${CLASSES[*]}"
echo "============================================"

mkdir -p "${PROJECT_DIR}/../logs/slurm" "${PROJECT_DIR}/checkpoints" "${PROJECT_DIR}/outputs"

ARRAY_MAX=$((N_GPUS - 1))

for CLASS in "${CLASSES[@]}"; do
    echo ""
    echo "--- ${CLASS} ---"

    # Stage 0: attention extraction (GPU, parallel)
    JOB0=$(sbatch --parsable \
        --array=0-${ARRAY_MAX} \
        --job-name="s0_${CLASS}" \
        "${SCRIPT_DIR}/slurm_array.sh" "${CLASS}" 0 2>&1)
    echo "  Stage 0: array job ${JOB0} (${N_GPUS} tasks)"

    # Stage 1: direction training (GPU, parallel, depends on stage 0)
    JOB1=$(sbatch --parsable \
        --array=0-${ARRAY_MAX} \
        --dependency=afterok:${JOB0} \
        --job-name="s1_${CLASS}" \
        "${SCRIPT_DIR}/slurm_array.sh" "${CLASS}" 1 2>&1)
    echo "  Stage 1: array job ${JOB1} (${N_GPUS} tasks)"

    # Stage 2: generation (GPU, one job per version, depends on stage 1)
    JOB2_IDS=""
    for VERSION in 1 2 3 4 5; do
        JOB2=$(sbatch --parsable \
            --array=0-${ARRAY_MAX} \
            --dependency=afterok:${JOB1} \
            --job-name="s2_${CLASS}_v${VERSION}" \
            "${SCRIPT_DIR}/slurm_array.sh" "${CLASS}" 2 2>&1)
        JOB2_IDS="${JOB2_IDS:+${JOB2_IDS},}afterok:${JOB2}"
    done
    echo "  Stage 2: 5 version array jobs submitted"

    # Stage 3: evaluation (CPU, sequential per version, depends on all stage 2)
    # Note: Stage 3 uses OpenAI API, not GPU. Run as single CPU jobs.
    for VERSION in 1 2 3 4 5; do
        JOB3=$(sbatch --parsable \
            --dependency=${JOB2_IDS} \
            --account=bdau-delta-cpu \
            --partition=cpu \
            --gpus-per-node=0 \
            --cpus-per-task=4 \
            --mem=16g \
            --time=06:00:00 \
            --job-name="s3_${CLASS}_v${VERSION}" \
            --output=logs/slurm/%x_%j.out \
            --error=logs/slurm/%x_%j.out \
            --wrap="cd ${PROJECT_DIR} && ${PROJECT_DIR}/../conda_envs/llm_steering/bin/python -u -c \"
import logging; logging.basicConfig(level=logging.INFO)
from steering_fast.utils import load_config
from steering_fast.pipeline.runner import run_pipeline
cfg = load_config(data='${CLASS}', overrides={
    'stages': [3],
    'generation.versions': [${VERSION}],
    'paths.data_dir': '${PROJECT_DIR}/../attention_guided_steering/data',
    'paths.checkpoint_dir': '${PROJECT_DIR}/checkpoints',
    'paths.output_dir': '${PROJECT_DIR}/outputs',
})
run_pipeline(cfg)
\"" 2>&1)
    done
    echo "  Stage 3: 5 evaluation jobs submitted"
done

echo ""
echo "============================================"
echo "All jobs submitted. Monitor:"
echo "  squeue -u \$USER"
echo "============================================"
