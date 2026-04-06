"""Single-process runner: load model ONCE, run all experiments.

Applies all steering_fast optimizations:
- Single model load (select_llm)
- Statement caching (load once, reuse)
- GPU cleanup between experiments
- Pickle protocol 5
- Proper seed management (set_seed)
- Per-experiment timing with CSV export
- Checkpointing (resume from crashed concept)
- Smoke test mode (n_concepts flag)
- Multi-concept support via concept list or SLURM array slicing

Usage:
    # Single concept
    python -m steering_fast.prefix_optimization.run_all \
        --concept bacteria --concept_class fears --data_dir ./data

    # All concepts in a class
    python -m steering_fast.prefix_optimization.run_all \
        --concept_class fears --data_dir ./data --all_concepts

    # SLURM array slice
    python -m steering_fast.prefix_optimization.run_all \
        --concept_class fears --data_dir ./data --all_concepts \
        --slice_start 0 --slice_end 20
"""

import argparse
import csv
import gc
import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import PrefixOptConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Statement cache (optimization #3/#18: load files once, reuse)
# ---------------------------------------------------------------------------
_statement_cache: Dict[str, List[str]] = {}


def load_statements_cached(data_dir: str) -> List[str]:
    """Load general statements with caching. Files read only once per process."""
    if data_dir in _statement_cache:
        return _statement_cache[data_dir]

    statements = []
    for fname in ["class_0.txt", "class_1.txt"]:
        path = os.path.join(data_dir, "general_statements", fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                statements.extend(line.strip() for line in f if line.strip())
    if not statements:
        statements = ["What do you think about this topic?"]

    _statement_cache[data_dir] = statements
    logger.info("Loaded and cached %d general statements", len(statements))
    return statements


# ---------------------------------------------------------------------------
# Model loading (optimization #1: load once)
# ---------------------------------------------------------------------------


def load_model_once(config: PrefixOptConfig):
    """Load the model a single time using the pipeline's select_llm."""
    from ..utils import core_imports_and_cwd

    data_dir = os.path.abspath(config.data_dir)
    with core_imports_and_cwd(data_dir):
        from utils import select_llm
        import utils as orig_utils
        orig_utils.DATA_DIR = data_dir
        if config.cache_dir:
            orig_utils.CACHE_DIR = config.cache_dir

        logger.info("Loading model via select_llm: %s", config.model_name)
        llm = select_llm(config.model_name)

    model = llm.language_model
    tokenizer = llm.tokenizer

    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model loaded. Vocab: %d, Hidden: %d, Layers: %d",
                len(tokenizer), model.config.hidden_size, model.config.num_hidden_layers)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Direction loading
# ---------------------------------------------------------------------------


def load_directions(config: PrefixOptConfig, concept: Optional[str] = None) -> Dict[int, torch.Tensor]:
    """Load pre-computed concept direction vectors."""
    c = concept or config.concept
    filename = config.get_direction_filename(c)
    path = os.path.join(config.data_dir, "directions", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Direction file not found: {path}")
    with open(path, "rb") as f:
        directions = pickle.load(f)
    logger.info("Loaded %d layer directions from %s", len(directions), os.path.basename(path))
    return directions


# ---------------------------------------------------------------------------
# Concept list loading + slicing (optimization #5: SLURM array support)
# ---------------------------------------------------------------------------


def load_concept_list(data_dir: str, concept_class: str) -> List[str]:
    """Load sorted, deduplicated concept list from file."""
    path = os.path.join(data_dir, "concepts", f"{concept_class}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept file not found: {path}")
    with open(path, encoding="utf-8") as f:
        concepts = sorted(set(line.strip().lower() for line in f if line.strip()))
    return concepts


# ---------------------------------------------------------------------------
# GPU cleanup (optimization #4)
# ---------------------------------------------------------------------------


def free_gpu():
    """Aggressive GPU cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ---------------------------------------------------------------------------
# Timing CSV export (optimization #8)
# ---------------------------------------------------------------------------


class ExperimentTimer:
    """Per-experiment timing with CSV export."""

    def __init__(self):
        self.records: List[Dict] = []

    def record(self, concept: str, experiment: str, elapsed: float, metrics: Dict):
        self.records.append({
            "concept": concept,
            "experiment": experiment,
            "elapsed_seconds": round(elapsed, 2),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        })

    def to_csv(self, path: str):
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keys = list(self.records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        logger.info("Timing saved to %s (%d records)", path, len(self.records))


# ---------------------------------------------------------------------------
# Checkpointing (optimization #6)
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: str) -> set:
    """Load set of completed concepts from checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            completed = set(line.strip() for line in f if line.strip())
        logger.info("Checkpoint loaded: %d concepts completed", len(completed))
        return completed
    return set()


def save_checkpoint(checkpoint_path: str, completed: set):
    """Save completed concepts to checkpoint file (one per line)."""
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w") as f:
        for c in sorted(completed):
            f.write(c + "\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all_experiments(config: PrefixOptConfig, concepts: Optional[List[str]] = None):
    """Load model once, run experiments for one or many concepts.

    Applies all steering_fast optimizations:
    - Single model load
    - Statement caching
    - GPU cleanup
    - Pickle protocol 5
    - Proper seeding
    - Timing CSV
    - Checkpointing
    """
    from ..utils import set_seed
    set_seed(config.seed)

    total_start = time.time()
    timer = ExperimentTimer()

    # ---- Single model load ----
    model, tokenizer = load_model_once(config)
    statements = load_statements_cached(config.data_dir)
    load_time = time.time() - total_start
    logger.info("Model + data loaded in %.1f seconds", load_time)

    # ---- Resolve concept list ----
    if concepts is None:
        concepts = [config.concept]

    # ---- Checkpointing ----
    checkpoint_dir = os.path.join(config.output_dir, config.model_name, config.concept_class)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "completed_concepts.txt")
    completed = load_checkpoint(checkpoint_path)

    from .methods.gcg import optimize_prefix_gcg

    experiment_configs = [
        ("gcg_layer16", {"layers": "16", "loss_type": "cosine", "gcg_multi_swap": False}),
        ("gcg_all_layers", {"layers": "all", "loss_type": "cosine", "gcg_multi_swap": False}),
    ]

    for concept in concepts:
        if concept in completed:
            logger.info("Skipping %s (already completed)", concept)
            continue

        logger.info("\n" + "=" * 60)
        logger.info("CONCEPT: %s (%d/%d)", concept, concepts.index(concept) + 1, len(concepts))
        logger.info("=" * 60)

        # Load directions for this concept
        try:
            directions = load_directions(config, concept)
        except FileNotFoundError:
            logger.warning("No directions for %s, skipping.", concept)
            continue

        concept_results = {
            "concept": concept,
            "concept_class": config.concept_class,
            "model": config.model_name,
            "experiments": {},
        }

        for exp_name, overrides in experiment_configs:
            logger.info("\n--- %s: %s ---", concept, exp_name)
            exp_config = PrefixOptConfig(
                **{**vars(config), "concept": concept, "n_steps": config.n_steps,
                   "method": "gcg", **overrides}
            )

            exp_start = time.time()
            result = optimize_prefix_gcg(model, tokenizer, directions, concept, statements, exp_config)
            exp_elapsed = time.time() - exp_start

            concept_results["experiments"][exp_name] = result
            timer.record(concept, exp_name, exp_elapsed, {
                "handcrafted_cos": result.get("handcrafted_mean_cos_sim", 0),
                "optimized_cos": result.get("optimized_mean_cos_sim", 0),
                "improvement": result.get("improvement_over_handcrafted", 0),
                "n_swaps": result.get("n_improvements", 0),
            })
            free_gpu()

        # Save per-concept results (pickle protocol 5)
        concept_dir = os.path.join(checkpoint_dir, concept.replace(" ", "_"))
        os.makedirs(concept_dir, exist_ok=True)

        pkl_path = os.path.join(concept_dir, "results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(concept_results, f, protocol=5)

        # Save per-concept JSON summary
        summary = {"concept": concept, "model": config.model_name}
        for exp_name, result in concept_results["experiments"].items():
            summary[f"{exp_name}_handcrafted"] = result.get("handcrafted_mean_cos_sim")
            summary[f"{exp_name}_no_prefix"] = result.get("no_prefix_mean_cos_sim")
            summary[f"{exp_name}_optimized"] = result.get("optimized_mean_cos_sim")
            summary[f"{exp_name}_improvement"] = result.get("improvement_over_handcrafted")
            summary[f"{exp_name}_prefix"] = result.get("optimized_prefix")
            summary[f"{exp_name}_time"] = result.get("total_time_seconds")

        json_path = os.path.join(concept_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Update checkpoint
        completed.add(concept)
        save_checkpoint(checkpoint_path, completed)
        logger.info("Saved results for %s. %d/%d concepts done.", concept, len(completed), len(concepts))

    # ---- Save timing CSV ----
    total_time = time.time() - total_start
    timer.to_csv(os.path.join(checkpoint_dir, "timing.csv"))

    logger.info("\n" + "=" * 60)
    logger.info("ALL COMPLETE: %d concepts in %.1f seconds (model load: %.1f s)",
                len(completed), total_time, load_time)
    logger.info("Results: %s", checkpoint_dir)
    logger.info("=" * 60)

    return completed


def main():
    parser = argparse.ArgumentParser(description="Prefix optimization: single model load, all experiments")

    # Concept selection
    parser.add_argument("--concept", "-c", default=None, help="Single concept (or use --all_concepts)")
    parser.add_argument("--concept_class", default="fears")
    parser.add_argument("--all_concepts", action="store_true", help="Run all concepts in the class")
    parser.add_argument("--smoke_test", type=int, default=0, help="Run only N concepts (0=disabled)")

    # SLURM array slicing
    parser.add_argument("--slice_start", type=int, default=None, help="Start index for concept slice")
    parser.add_argument("--slice_end", type=int, default=None, help="End index for concept slice")

    # Paths
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--model_name", "-m", default="llama_3.1_8b")
    parser.add_argument("--output_dir", default="outputs/prefix_optimization")

    # Optimization
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)

    # GCG params
    parser.add_argument("--gcg_topk", type=int, default=256)
    parser.add_argument("--gcg_batch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    MODEL_DIMS = {
        "llama_3.1_8b": (4096, 32),
        "llama_3.1_70b": (8192, 80),
        "llama_3.3_70b": (8192, 80),
        "qwen-14b": (5120, 24),
        "qwen-32b": (5120, 64),
    }
    hidden_dim, n_layers = MODEL_DIMS.get(args.model_name, (4096, 32))

    # Resolve concept list
    concepts = None
    if args.all_concepts or args.slice_start is not None:
        all_concepts = load_concept_list(args.data_dir, args.concept_class)

        # Smoke test: limit to N concepts
        if args.smoke_test > 0:
            all_concepts = all_concepts[:args.smoke_test]
            logger.info("SMOKE TEST: %d concepts", len(all_concepts))

        # SLURM array slicing
        if args.slice_start is not None:
            end = args.slice_end if args.slice_end is not None else len(all_concepts)
            all_concepts = all_concepts[args.slice_start:end]
            logger.info("SLICE [%d:%d]: %d concepts", args.slice_start, end, len(all_concepts))

        concepts = all_concepts
        concept_for_config = concepts[0] if concepts else "unknown"
    elif args.concept:
        concept_for_config = args.concept
    else:
        parser.error("Provide --concept or --all_concepts")

    config = PrefixOptConfig(
        model_name=args.model_name,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        cache_dir=args.cache_dir,
        concept=concept_for_config,
        concept_class=args.concept_class,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        seed=args.seed,
        log_every=args.log_every,
        gcg_topk=args.gcg_topk,
        gcg_batch_size=args.gcg_batch_size,
        grad_clip=args.grad_clip,
    )

    run_all_experiments(config, concepts=concepts)


if __name__ == "__main__":
    main()
