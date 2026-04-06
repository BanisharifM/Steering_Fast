"""Single-process runner: load model ONCE, run all experiments.

This replaces the SLURM script pattern of calling run_experiment.py multiple
times (each reloading the 8B model from scratch, wasting ~7 min per load).

Usage:
    python -m steering_fast.prefix_optimization.run_all \
        --concept bacteria --concept_class fears --data_dir ./data

Runs: GCG (single layer + all layers), hand-crafted baseline, no-prefix baseline.
All with a single model load.
"""

import argparse
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

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model loaded. Vocab: %d, Hidden: %d, Layers: %d",
                len(tokenizer), model.config.hidden_size, model.config.num_hidden_layers)
    return model, tokenizer


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


def load_statements(config: PrefixOptConfig) -> List[str]:
    """Load general statements."""
    statements = []
    for fname in ["class_0.txt", "class_1.txt"]:
        path = os.path.join(config.data_dir, "general_statements", fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                statements.extend(line.strip() for line in f if line.strip())
    if not statements:
        statements = ["What do you think about this topic?"]
    return statements


def free_gpu():
    """Aggressive GPU cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def run_all_experiments(config: PrefixOptConfig):
    """Load model once, run all experiments sequentially."""
    torch.manual_seed(config.seed)
    total_start = time.time()

    # ---- Single model load ----
    model, tokenizer = load_model_once(config)
    directions = load_directions(config)
    statements = load_statements(config)
    load_time = time.time() - total_start
    logger.info("Model + data loaded in %.1f seconds", load_time)

    all_results = {
        "concept": config.concept,
        "concept_class": config.concept_class,
        "model": config.model_name,
        "load_time_seconds": load_time,
        "experiments": {},
    }

    # ---- Experiment 1: GCG single layer (layer 16) ----
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1: GCG single layer 16")
    logger.info("=" * 60)

    config_l16 = PrefixOptConfig(
        **{**vars(config), "layers": "16", "n_steps": 200, "method": "gcg", "loss_type": "cosine"}
    )
    from .methods.gcg import optimize_prefix_gcg
    result_l16 = optimize_prefix_gcg(model, tokenizer, directions, config.concept, statements, config_l16)
    all_results["experiments"]["gcg_layer16"] = result_l16
    free_gpu()

    # ---- Experiment 2: GCG all layers ----
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: GCG all layers")
    logger.info("=" * 60)

    config_all = PrefixOptConfig(
        **{**vars(config), "layers": "all", "n_steps": 200, "method": "gcg", "loss_type": "cosine"}
    )
    result_all = optimize_prefix_gcg(model, tokenizer, directions, config.concept, statements, config_all)
    all_results["experiments"]["gcg_all_layers"] = result_all
    free_gpu()

    # ---- Experiment 3: GCG layer 16, angular loss ----
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3: GCG layer 16, angular loss")
    logger.info("=" * 60)

    config_ang = PrefixOptConfig(
        **{**vars(config), "layers": "16", "n_steps": 200, "method": "gcg", "loss_type": "angular"}
    )
    result_ang = optimize_prefix_gcg(model, tokenizer, directions, config.concept, statements, config_ang)
    all_results["experiments"]["gcg_layer16_angular"] = result_ang
    free_gpu()

    # ---- Save all results ----
    output_dir = os.path.join(config.output_dir, config.model_name, config.concept_class, config.concept.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)

    pkl_path = os.path.join(output_dir, "all_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f, protocol=5)

    # Summary JSON
    total_time = time.time() - total_start
    summary = {
        "concept": config.concept,
        "model": config.model_name,
        "load_time": load_time,
        "total_time": total_time,
    }
    for exp_name, exp_result in all_results["experiments"].items():
        summary[f"{exp_name}_handcrafted"] = exp_result.get("handcrafted_mean_cos_sim")
        summary[f"{exp_name}_no_prefix"] = exp_result.get("no_prefix_mean_cos_sim")
        summary[f"{exp_name}_optimized"] = exp_result.get("optimized_mean_cos_sim")
        summary[f"{exp_name}_improvement"] = exp_result.get("improvement_over_handcrafted")
        summary[f"{exp_name}_delta"] = exp_result.get("optimized_delta")
        summary[f"{exp_name}_n_swaps"] = exp_result.get("n_improvements")
        summary[f"{exp_name}_prefix"] = exp_result.get("optimized_prefix")
        summary[f"{exp_name}_time"] = exp_result.get("total_time_seconds")

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("  Total time: %.1f seconds (model load: %.1f s)", total_time, load_time)
    logger.info("  Results: %s", output_dir)
    for exp_name in all_results["experiments"]:
        r = all_results["experiments"][exp_name]
        logger.info("  %s: handcrafted=%.4f -> optimized=%.4f (%+.4f)",
                    exp_name,
                    r.get("handcrafted_mean_cos_sim", 0),
                    r.get("optimized_mean_cos_sim", 0),
                    r.get("improvement_over_handcrafted", 0))
    logger.info("=" * 60)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Prefix optimization: single model load, all experiments")

    parser.add_argument("--concept", "-c", required=True)
    parser.add_argument("--concept_class", default="fears")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--model_name", "-m", default="llama_3.1_8b")
    parser.add_argument("--output_dir", default="outputs/prefix_optimization")
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

    config = PrefixOptConfig(
        model_name=args.model_name,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        cache_dir=args.cache_dir,
        concept=args.concept,
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

    run_all_experiments(config)


if __name__ == "__main__":
    main()
