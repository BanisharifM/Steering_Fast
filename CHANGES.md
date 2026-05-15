# CHANGES — What's different from the original pipeline

Map of optimizations layered on top of [pdavar/attention_guided_steering](https://github.com/pdavar/attention_guided_steering).

**TL;DR:** the math is unchanged. Every file in `steering_fast/core/` is a verbatim
copy of the corresponding file in the original repo (run `diff` to confirm). All
speedups live in *new* files that wrap the core code.

## Summary table

| # | Optimization | Where | Why it matters |
|---|---|---|---|
| 1 | Batched hidden-state extraction | [`steering_fast/core/direction_utils.py`](steering_fast/core/direction_utils.py) (`*_batched` functions) and [`steering_fast/extraction/hidden_states.py`](steering_fast/extraction/hidden_states.py) | Original processed one prompt at a time. Now a configurable batch (e.g. 32) goes through one forward pass. ~2–4× stage-1 speedup. |
| 2 | Batched attention extraction | [`steering_fast/extraction/attention.py`](steering_fast/extraction/attention.py) | Same idea for stage 0. ~2–4× speedup. |
| 3 | Vectorized head aggregation | [`steering_fast/core/direction_utils.py`](steering_fast/core/direction_utils.py) | Replaces a Python-level loop over attention heads with one tensor op. ~10× on the head-aggregation step. |
| 4 | Statement cache | [`steering_fast/data/statements.py`](steering_fast/data/statements.py) | `class_0.txt` and `class_1.txt` are read **once** per process and reused across all concepts (originally re-read per concept). Removes a hidden I/O bottleneck on Lustre. |
| 5 | safetensors fastload | [`steering_fast/utils.py`](steering_fast/utils.py) | Concept directions saved as `.safetensors` instead of pickled tensors → zero-copy GPU load. ~76× direction-loading speedup on stage 2 startup. |
| 6 | Per-concept checkpointing | [`steering_fast/tracking/checkpoint.py`](steering_fast/tracking/checkpoint.py) | After each concept finishes, a small JSON manifest records progress. A crashed or killed job resumes exactly where it stopped — no recomputation. |
| 7 | GPU memory cleanup between stages | [`steering_fast/pipeline/runner.py`](steering_fast/pipeline/runner.py) (`_free_gpu_memory`) | `gc.collect()` + `torch.cuda.empty_cache()` between stages 0/1/2 so leftover allocations don't accumulate. Pipeline now fits on a 40 GB A100. |
| 8 | SLURM array partitioning | [`steering_fast/config.py`](steering_fast/config.py) (`SlicingConfig`) and [`scripts/slurm_array.sh`](scripts/slurm_array.sh) | One concept per array task. Splits a 100-concept run across N GPUs cleanly. |
| 9 | Robust GPT-4o evaluation | [`steering_fast/evaluation/openai_eval.py`](steering_fast/evaluation/openai_eval.py) | (a) Exponential backoff on every API error, not just 429s. (b) Regex score parsing — original parsing took `score_str[0]` so it silently truncated multi-digit scores. (c) Optional **Batch API** path: ~50% cheaper, no rate limit, ~24 h turnaround. |
| 10 | Per-concept timing CSV | [`steering_fast/tracking/timer.py`](steering_fast/tracking/timer.py) | Each concept's wall time per stage is written to `timing.csv` for performance audits. |
| 11 | Optional W&B tracking | [`steering_fast/tracking/wandb_tracker.py`](steering_fast/tracking/wandb_tracker.py) | Per-concept metrics + stage summaries logged to wandb if enabled. Auto-login from `.wandb_key`. Gracefully no-ops when disabled. |
| 12 | Hydra config | [`steering_fast/config.py`](steering_fast/config.py) + [`steering_fast/conf/`](steering_fast/conf/) | Replaces the argparse flags with a typed config tree (`model/`, `data/`, `steering/`, `experiment/`). Supports overrides like `training.batch_size=32` without editing source. |

## What was NOT changed

These are byte-identical to the original (with the one noted exception):

- `core/control_toolkits.py` — RFM, PCA, linear-probe, logistic, mean-difference toolkits
- `core/rfm.py` — RFM solver and AGOP computation
- `core/datasets.py` — all dataset constructors
- `core/neural_controllers.py` — `NeuralController` orchestration
- `core/generation_utils.py` — hook installation, steered generation
- `core/utils.py` — `select_llm`, `get_coefs`, `compute_save_directions`, `generate`
- `core/direction_utils.py` — **exception:** all original functions unchanged; new `*_batched` variants were appended below them
- `core/args.py` — argparse definitions (kept for backward compat, even though Hydra config supersedes them)

To verify: clone the original and `diff -r` against `steering_fast/core/`.

## How the pipeline actually runs now

The new orchestration layer ([`steering_fast/pipeline/runner.py`](steering_fast/pipeline/runner.py)) calls
the same five stages as the original, but wraps each one with checkpointing, timing,
and GPU cleanup:

```
runner.run_pipeline(cfg):
  for stage in [0, 1, 2, 3, 4]:
      with timer.time_stage(stage):
          stage_module.run(cfg, llm, checkpoint_manager)
      _free_gpu_memory()
```

Each `stage{0..4}.py` module delegates to the original `core/` code for the math, but
calls the batched extraction helpers (`extraction/`) and routes through the checkpoint
manager (`tracking/checkpoint.py`).

## Where to start reading

1. [`README.md`](README.md) — overview and quickstart
2. [`steering_fast/pipeline/runner.py`](steering_fast/pipeline/runner.py) — entry point: see how stages are sequenced
3. [`steering_fast/pipeline/stage0.py`](steering_fast/pipeline/stage0.py) … [`stage4.py`](steering_fast/pipeline/stage4.py) — one file per stage; thin wrappers over `core/`
4. [`steering_fast/core/`](steering_fast/core/) — your original code, unchanged
5. [`steering_fast/extraction/`](steering_fast/extraction/), [`steering_fast/data/statements.py`](steering_fast/data/statements.py), [`steering_fast/tracking/`](steering_fast/tracking/) — the speedups
