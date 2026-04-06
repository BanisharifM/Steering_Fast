# Prefix Optimization for Concept Vector Alignment

## What This Is

A research tool for discovering optimal prefix tokens that make LLM activations maximally aligned with pre-computed concept steering vectors.

**Research question**: Given a concept vector v (learned by RFM or other methods from the existing steering pipeline), what prefix tokens, when prepended to a general statement, produce activations closest to v?

This is an interpretability/discovery project. It does not aim to improve steering accuracy; it aims to understand what input patterns best activate concept representations.

## How It Works

1. **Input**: A frozen LLM, a pre-computed concept vector v^(l) at layer l, a general statement s
2. **Optimize**: Continuous prefix embeddings P = [p_1, ..., p_K] via gradient descent
3. **Loss**: Cosine similarity between activation h^(l)(P; s) and concept vector v^(l)
4. **Output**: Optimal continuous embeddings + nearest discrete tokens

```
Loss = 1 - cos(h^(l)(P; s), v^(l)) + lambda * regularization
```

Gradients flow from the loss through frozen transformer layers back to the prefix embeddings. Only P is updated; the LLM remains frozen.

## Prerequisites

- Pre-computed concept directions from the existing pipeline (stage 0 + stage 1)
- Data directory (symlinked at `./data`)
- GPU with enough memory for forward + backward pass through the model

## Quick Start

```bash
# Activate environment
source scripts/env.sh

# Run single-concept, single-layer smoke test
python -m steering_fast.prefix_optimization.optimize \
    --concept "spiders" \
    --concept_class fears \
    --layer 16 \
    --prefix_length 10 \
    --n_steps 500 \
    --lr 0.01

# Run multi-layer optimization
python -m steering_fast.prefix_optimization.optimize \
    --concept "spiders" \
    --concept_class fears \
    --layers all \
    --prefix_length 10 \
    --n_steps 1000

# Run full sweep (SLURM)
sbatch scripts/submit_prefix_optimization.sh
```

## Project Structure

```
steering_fast/
  prefix_optimization/        # New module for this project
    __init__.py
    optimize.py               # Main optimization loop
    loss.py                   # Loss functions (cosine sim, regularization)
    prefix_embeddings.py      # Prefix embedding initialization and management
    discrete_recovery.py      # Convert continuous embeddings to discrete tokens
    evaluate.py               # Compare optimal vs hand-crafted prefixes
    config.py                 # Hyperparameter configuration
  docs/
    algorithm_design.md       # Full mathematical formulation
    paper_analysis.md         # Analysis of the base paper
    codebase_math_details.md  # Implementation details of existing pipeline
    related_work_survey.md    # Survey of related techniques
    ROADMAP.md                # Phase-by-phase development plan
```

## Documentation

- [Algorithm Design](docs/algorithm_design.md) -- Full mathematical formulation and implementation plan
- [Roadmap](docs/ROADMAP.md) -- Phase-by-phase development plan
- [Paper Analysis](docs/paper_analysis.md) -- Deep analysis of the base paper
- [Related Work](docs/related_work_survey.md) -- Survey of prefix tuning, concept vectors, and related techniques

## Key Design Decisions

1. **Cosine similarity loss** (not L2): scale-invariant, measures directional alignment
2. **Input-layer-only optimization** (not per-layer prefixes): simpler, interpretable, discrete tokens recoverable
3. **Frozen LLM**: only prefix embeddings are optimized
4. **Phased development**: single layer -> multi-layer -> multi-concept -> analysis
