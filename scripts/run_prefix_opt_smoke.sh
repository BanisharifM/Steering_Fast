#!/bin/bash
# ============================================================================
# SLURM Job: Prefix Optimization Smoke Test
#
# Runs prefix optimization for a single concept on a single layer to verify
# the entire pipeline works: model loading, direction loading, gradient flow,
# loss computation, discrete token recovery.
#
# Then runs all three methods (gradient, jacobian, logit_lens) to compare.
#
# Usage:
#   sbatch scripts/run_prefix_opt_smoke.sh
# ============================================================================
#SBATCH --job-name=prefix_opt_smoke
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
# SBATCH copies scripts to /var/spool, so BASH_SOURCE is unreliable.
# Use SLURM_SUBMIT_DIR (where sbatch was invoked) or hardcode.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/work/hdd/bdau/mbanisharifdehkordi/LLM_Steering/steering_fast}"
PARENT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

# Python detection
PYTHON="${PYTHON:-$(find "${PARENT_DIR}" -path "*/conda_envs/*/bin/python" -type f 2>/dev/null | head -1)}"
if [ -z "${PYTHON}" ] || [ ! -f "${PYTHON}" ]; then
    PYTHON="$(which python)"
fi

export CACHE_DIR="${CACHE_DIR:-}"
export HF_HOME="${HF_HOME:-}"
export PYTHONUNBUFFERED=1

# Resolve symlink to absolute path for compute nodes
DATA_DIR="${DATA_DIR:-$(readlink -f "${PROJECT_DIR}/data" 2>/dev/null || echo "${PROJECT_DIR}/data")}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/prefix_optimization"

# Verify directions exist
echo "Data dir resolved to: ${DATA_DIR}"
DIRECTION_FILE="${DATA_DIR}/directions/rfm_bacteria_tokenidx_max_attn_per_layer_block_softlabels_llama_3.1_8b.pkl"
if [ ! -f "${DIRECTION_FILE}" ]; then
    echo "ERROR: No direction files found. Run the steering pipeline first."
    echo "Looked for: ${DIRECTION_FILE}"
    echo ""
    echo "Available direction files:"
    ls "${DATA_DIR}/directions/" 2>/dev/null | head -10
    exit 1
fi

# Get first available concept from direction files
CONCEPT=$(ls "${DATA_DIR}/directions/" | grep "^rfm_" | head -1 | sed 's/rfm_//; s/_tokenidx_.*//')
if [ -z "${CONCEPT}" ]; then
    echo "ERROR: Could not determine concept from direction files."
    exit 1
fi

echo "============================================"
echo "  Prefix Optimization Smoke Test"
echo "  Concept: ${CONCEPT}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: ${PYTHON}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Start: $(date)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

# ============================================================================
# Test 1: Single layer, gradient method, cosine loss
# ============================================================================
echo ""
echo ">>> TEST 1: Gradient method, single layer 16, cosine loss"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --method gradient \
    --layers 16 \
    --prefix_length 10 \
    --n_steps 200 \
    --lr 0.01 \
    --loss_type cosine \
    --init_strategy concept_name \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/test1_gradient" \
    --log_every 20 \
    --seed 42 \
    2>&1

echo ""
echo ">>> TEST 1 complete: $(date)"

# ============================================================================
# Test 2: Logit lens method (no optimization, just analysis)
# ============================================================================
echo ""
echo ">>> TEST 2: Logit lens analysis"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --method logit_lens \
    --layers 16 \
    --prefix_length 10 \
    --output_dir "${OUTPUT_DIR}/test2_logit_lens" \
    --seed 42 \
    2>&1

echo ""
echo ">>> TEST 2 complete: $(date)"

# ============================================================================
# Test 3: Jacobian reachability analysis
# ============================================================================
echo ""
echo ">>> TEST 3: Jacobian reachability analysis"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --method jacobian \
    --layers 16 \
    --prefix_length 10 \
    --jacobian_rank 32 \
    --output_dir "${OUTPUT_DIR}/test3_jacobian" \
    --seed 42 \
    2>&1

echo ""
echo ">>> TEST 3 complete: $(date)"

# ============================================================================
# Test 4: Loss function comparison (gradient method, vary loss)
# ============================================================================
echo ""
echo ">>> TEST 4: Loss function comparison"
echo "-----------------------------------------------------------"
for LOSS in cosine angular normalized_projection projection; do
    echo ""
    echo "  ... Loss: ${LOSS}"
    ${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
        --concept "${CONCEPT}" \
        --concept_class fears \
        --data_dir "${DATA_DIR}" \
        --cache_dir "${CACHE_DIR}" \
        --method gradient \
        --layers 16 \
        --prefix_length 10 \
        --n_steps 200 \
        --lr 0.01 \
        --loss_type "${LOSS}" \
        --init_strategy concept_name \
        --target_position last_prefix \
        --output_dir "${OUTPUT_DIR}/test4_loss_${LOSS}" \
        --log_every 50 \
        --seed 42 \
        2>&1
done

echo ""
echo ">>> TEST 4 complete: $(date)"

# ============================================================================
# Test 5: Multi-layer optimization
# ============================================================================
echo ""
echo ">>> TEST 5: Multi-layer (all layers)"
echo "-----------------------------------------------------------"
${PYTHON} -u -m steering_fast.prefix_optimization.run_experiment \
    --concept "${CONCEPT}" \
    --concept_class fears \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --method gradient \
    --layers all \
    --prefix_length 10 \
    --n_steps 300 \
    --lr 0.01 \
    --loss_type cosine \
    --init_strategy concept_name \
    --target_position last_prefix \
    --output_dir "${OUTPUT_DIR}/test5_multilayer" \
    --log_every 50 \
    --seed 42 \
    2>&1

echo ""
echo ">>> TEST 5 complete: $(date)"

echo ""
echo "============================================"
echo "  All smoke tests complete: $(date)"
echo "  Results in: ${OUTPUT_DIR}"
echo "============================================"
