# steering_fast

Optimized pipeline for attention-guided LLM steering, plus a research module that
discovers optimal token prefixes for activating pre-trained concept directions.

> Based on the reference implementation at
> [pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering)
> by [Parmida Davarmanesh](https://github.com/pdavar) (MIT).
> The original algorithms live verbatim in `steering_fast/core/`. See the paper
> citation below and [LICENSE](LICENSE) for terms.
>
> **What's different from the original pipeline?** See [CHANGES.md](CHANGES.md) for
> a per-file walkthrough of every optimization.

This codebase has two layers:

1. **Optimized steering pipeline** (`main` branch, `steering_fast/pipeline/`) — a refactored
   implementation of the 5-stage pipeline from Davarmanesh et al.
   (*"Efficient and accurate steering of LLMs through attention-guided feature learning"*,
   arXiv:2602.00333). Adds batched extraction, per-concept checkpointing, statement
   caching, SLURM array support, safetensors fastload, GPU cleanup, and Hydra-based
   configuration. The original algorithms live verbatim in `steering_fast/core/`.

2. **Prefix optimization** (`prefix-optimization` branch, `steering_fast/prefix_optimization/`) —
   research module that asks: given a pre-computed concept direction `v^(l)`, what discrete
   prefix tokens produce hidden states maximally aligned with `v^(l)`? Implements GCG
   (Greedy Coordinate Gradient), PEZ v2, and stubs for Jacobian-reachability,
   logit-lens, and gradient-only baselines.

## Quick start

```bash
source scripts/env.sh
pip install -e .

# Smoke test for prefix optimization (single GPU, ~2h)
sbatch scripts/run_prefix_opt_smoke.sh

# GCG prefix optimization on the fears concept class
sbatch scripts/run_gcg.sh

# Run the full optimized pipeline (stages 0–4) as a SLURM array
sbatch scripts/slurm_array.sh
```

Requires Python ≥3.10, an NVIDIA GPU with CUDA, and pre-computed concept directions in
`data/directions/`. The `data/` directory is a symlink — point it at a directory containing
`concepts/`, `general_statements/`, `evaluation_prompts/`, and `directions/` produced by
the original attention-guided pipeline.

## Repository layout

```
steering_fast/
├── pipeline/                # stages 0–4 + runner
├── extraction/              # batched attention + hidden-state extraction
├── generation/              # steering hooks
├── evaluation/              # GPT-4o evaluator (incl. Batch API)
├── tracking/                # checkpoint, timer, wandb
├── core/                    # verbatim originals — do not modify
├── prefix_optimization/     # research module (GCG, PEZ v2, etc.)
└── conf/                    # Hydra config tree (model/data/steering/experiment)

scripts/                     # SLURM submission scripts
tests/                       # smoke tests
```

## Configuration

All configs are Hydra dataclasses (`steering_fast/config.py`). Run the pipeline via the
CLI entrypoint or directly:

```python
from steering_fast.pipeline.runner import run_pipeline
from steering_fast.utils import load_config

cfg = load_config(
    model="llama_3_1_8b",
    data="fears",
    steering="rfm",
    experiment="full",
    overrides={"training.batch_size": 32},
)
run_pipeline(cfg)
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@misc{davarmanesh2026efficient,
  title  = {Efficient and accurate steering of Large Language Models through
            attention-guided feature learning},
  author = {Davarmanesh, Parmida and Wilson, Ashia and Radhakrishnan, Adityanarayanan},
  year   = {2026},
  eprint = {2602.00333},
  archivePrefix = {arXiv},
}
```

## License

MIT — see [LICENSE](LICENSE).
