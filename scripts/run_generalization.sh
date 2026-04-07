#!/bin/bash
#SBATCH --job-name=gcg_general
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
CACHE_DIR="${CACHE_DIR:-$(dirname "${DATA_DIR}")}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/prefix_optimization"

echo "============================================"
echo "  Generalization + Steering Evaluation"
echo "  Node: $(hostname)"
echo "  Start: $(date)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p logs/slurm

# Part 1: Multi-statement generalization (10 concepts, 20 statements)
echo ""
echo ">>> GENERALIZATION TEST: 10 concepts, 20 statements each"
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
echo ">>> Generalization complete: $(date)"

# Part 2: Steering evaluation (10 concepts)
echo ""
echo ">>> STEERING EVALUATION: 10 concepts"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.evaluate_steering \
    --results_dir "${OUTPUT_DIR}/llama_3.1_8b/fears" \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --model_name llama_3.1_8b \
    --n_concepts 10 \
    --coefficient 0.8 \
    2>&1

echo ""
echo "============================================"
echo "  All complete: $(date)"
echo "============================================"
