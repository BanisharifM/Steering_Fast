#!/bin/bash
# ============================================================================
# Multi-statement GCG test + generalization evaluation
# Tests the anti-overfitting fix: optimize across 5 statements, then test on 20
# ============================================================================
#SBATCH --job-name=gcg_multistmt
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.out

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
CACHE_DIR="${CACHE_DIR:-$(dirname "${DATA_DIR}")}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/prefix_optimization_5stmts"

echo "============================================"
echo "  Multi-Statement GCG (5 statements)"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start: $(date)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

# Part 1: GCG with 5 statements on 10 concepts
echo ""
echo ">>> GCG with n_statements=5, 10 concepts"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_all \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --all_concepts --smoke_test 10 \
    --output_dir "${OUTPUT_DIR}" \
    --n_steps 200 --gcg_topk 256 --gcg_batch_size 64 \
    --n_statements 5 \
    --log_every 10 --seed 42 \
    2>&1

echo ""
echo ">>> Multi-statement GCG complete: $(date)"

# Part 2: Test generalization on 20 statements
echo ""
echo ">>> GENERALIZATION TEST on multi-statement results"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.evaluate_generalization \
    --results_dir "${OUTPUT_DIR}/llama_3.1_8b/fears" \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --model_name llama_3.1_8b \
    --concept_class fears \
    --n_concepts 10 \
    --n_statements 20 \
    2>&1

echo ""
echo "============================================"
echo "  All complete: $(date)"
echo "============================================"
