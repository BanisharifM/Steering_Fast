"""Main experiment runner for prefix optimization.

Loads model, concept directions, and runs the specified optimization method(s).
Designed to be called from SLURM jobs or directly.
"""

import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import PrefixOptConfig

logger = logging.getLogger(__name__)


def load_directions(config: PrefixOptConfig, concept: Optional[str] = None) -> Dict[int, torch.Tensor]:
    """Load pre-computed concept direction vectors from pickle.

    Direction pickles are Dict[int, Tensor] where keys are layer indices
    and values are direction tensors of shape (1, d_model) or (d_model,).
    """
    c = concept or config.concept
    filename = config.get_direction_filename(c)
    path = os.path.join(config.data_dir, "directions", filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Direction file not found: {path}\n"
            f"Run the steering pipeline (stages 0-1) first to generate directions."
        )

    with open(path, "rb") as f:
        directions = pickle.load(f)

    logger.info("Loaded %d layer directions from %s", len(directions), path)

    # Verify shapes
    for layer_idx, v in list(directions.items())[:3]:
        if hasattr(v, "shape"):
            logger.info("  Layer %d: shape=%s, norm=%.4f", layer_idx, v.shape, v.flatten().norm().item())

    return directions


def load_statements(config: PrefixOptConfig) -> List[str]:
    """Load general statements for multi-statement optimization."""
    class_0_path = os.path.join(config.data_dir, "general_statements", "class_0.txt")
    class_1_path = os.path.join(config.data_dir, "general_statements", "class_1.txt")

    statements = []
    for path in [class_0_path, class_1_path]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                statements.extend(line.strip() for line in f if line.strip())

    if not statements:
        # Fallback: generic statement
        statements = ["What do you think about this topic?"]
        logger.warning("No statement files found; using fallback statement.")

    logger.info("Loaded %d general statements", len(statements))
    return statements


def load_model(config: PrefixOptConfig, core_ctx):
    """Load the frozen LLM and tokenizer using the existing pipeline's select_llm.

    Uses core/utils.py:select_llm() -- the same function the steering pipeline uses.
    Must be called inside a core_imports_and_cwd context.

    Args:
        config: PrefixOptConfig
        core_ctx: active core_imports_and_cwd context (for import resolution)

    Returns:
        (model, tokenizer) where model has all params frozen.
    """
    from utils import select_llm
    import utils as orig_utils

    # Set DATA_DIR and CACHE_DIR so core code resolves paths correctly
    orig_utils.DATA_DIR = config.data_dir
    if config.cache_dir:
        orig_utils.CACHE_DIR = config.cache_dir

    logger.info("Loading model via select_llm: %s", config.model_name)
    llm = select_llm(config.model_name)

    model = llm.language_model
    tokenizer = llm.tokenizer

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        "Model loaded. Vocab size: %d, Hidden dim: %d, Layers: %d",
        len(tokenizer),
        model.config.hidden_size,
        model.config.num_hidden_layers,
    )

    return model, tokenizer


def evaluate_handcrafted_prefix(
    model, tokenizer, directions, concept, config, statements
) -> Dict:
    """Evaluate the alignment of the hand-crafted prefix from the paper.

    This provides a baseline to compare optimized prefixes against.
    """
    from .methods.gradient import extract_activations

    device = next(model.parameters()).device
    embedding_matrix = model.model.embed_tokens.weight.detach()

    # Build the hand-crafted prefix based on concept class
    templates = {
        "fears": f"Personify someone who is terrified of {concept}.",
        "moods": f"Act as someone who is feeling {concept}.",
        "personas": f"You are {concept}.",
        "personalities": f"Act as someone who is {concept}.",
        "places": f"Describe what it is like to be in {concept}.",
    }
    prefix_text = templates.get(config.concept_class, f"Think about {concept}.")

    # Tokenize and embed
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt")
    prefix_ids = prefix_ids.to(device)
    with torch.no_grad():
        prefix_emb = model.model.embed_tokens(prefix_ids).float()

    prefix_len = prefix_ids.shape[1]

    # Normalize directions
    clean_directions = {}
    for layer_idx, v in directions.items():
        v_flat = v.flatten().float().to(device)
        v_flat = v_flat / v_flat.norm()
        clean_directions[layer_idx]= v_flat

    layers = config.get_layers()
    active_directions = {l: clean_directions[l] for l in layers if l in clean_directions}
    active_layers = sorted(active_directions.keys())

    # Evaluate
    stmt = statements[0]
    token_ids = tokenizer.encode(stmt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        stmt_emb = model.model.embed_tokens(token_ids).float()

    activations = extract_activations(
        model, prefix_emb, stmt_emb, active_layers, config.target_position, prefix_len
    )

    cos_sims = {}
    for layer_idx in active_layers:
        if layer_idx in activations:
            import torch.nn.functional as F
            cs = F.cosine_similarity(
                activations[layer_idx].float().unsqueeze(0),
                active_directions[layer_idx].unsqueeze(0),
            ).item()
            cos_sims[layer_idx] = cs

    return {
        "prefix_text": prefix_text,
        "prefix_length": prefix_len,
        "cosine_similarities": cos_sims,
        "mean_cosine_similarity": sum(cos_sims.values()) / len(cos_sims) if cos_sims else 0.0,
    }


def run_single_experiment(config: PrefixOptConfig) -> Dict:
    """Run a complete prefix optimization experiment for one concept.

    Uses core_imports_and_cwd to load the model via the same infrastructure
    as the existing steering pipeline (select_llm from core/utils.py).

    Returns a results dict with all method outputs.
    """
    from ..utils import core_imports_and_cwd

    torch.manual_seed(config.seed)

    logger.info("=" * 60)
    logger.info("EXPERIMENT: concept=%s, method=%s", config.concept, config.method)
    logger.info("  layers=%s, prefix_length=%d, loss=%s, init=%s",
                config.layers, config.prefix_length, config.loss_type, config.init_strategy)
    logger.info("=" * 60)

    # Use core_imports_and_cwd to set up the import environment
    # This is the same pattern used by stage0.py, stage1.py, stage2.py
    data_dir = os.path.abspath(config.data_dir)

    with core_imports_and_cwd(data_dir):
        # Load model using the pipeline's select_llm
        model, tokenizer = load_model(config, core_ctx=True)

        # Load directions
        directions = load_directions(config)

        # Load statements
        statements = load_statements(config)

    # Evaluate hand-crafted baseline
    logger.info("Evaluating hand-crafted prefix baseline...")
    baseline = evaluate_handcrafted_prefix(
        model, tokenizer, directions, config.concept, config, statements
    )
    logger.info("Hand-crafted baseline mean cos_sim: %.4f", baseline["mean_cosine_similarity"])

    results = {
        "config": vars(config),
        "baseline": baseline,
        "methods": {},
    }

    methods_to_run = (
        ["pez_v2", "gradient", "pez", "jacobian", "logit_lens"]
        if config.method == "all"
        else [config.method]
    )

    for method_name in methods_to_run:
        logger.info("Running method: %s", method_name)

        if method_name == "gradient":
            from .methods.gradient import optimize_prefix
            method_result = optimize_prefix(
                model, tokenizer, directions, config.concept, statements, config
            )
        elif method_name == "jacobian":
            from .methods.jacobian import run_jacobian_analysis
            method_result = run_jacobian_analysis(
                model, tokenizer, directions, config.concept, statements, config
            )
        elif method_name == "pez":
            from .methods.pez import optimize_prefix_pez
            method_result = optimize_prefix_pez(
                model, tokenizer, directions, config.concept, statements, config
            )
        elif method_name == "pez_v2":
            from .methods.pez_v2 import optimize_prefix_pez_v2
            method_result = optimize_prefix_pez_v2(
                model, tokenizer, directions, config.concept, statements, config
            )
        elif method_name == "gcg":
            from .methods.gcg import optimize_prefix_gcg
            method_result = optimize_prefix_gcg(
                model, tokenizer, directions, config.concept, statements, config
            )
        elif method_name == "logit_lens":
            from .methods.logit_lens import run_logit_lens_analysis
            method_result = run_logit_lens_analysis(
                model, tokenizer, directions, config.concept, statements, config
            )
        else:
            logger.warning("Unknown method: %s, skipping.", method_name)
            continue

        results["methods"][method_name] = method_result

        # GPU cleanup between methods
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    output_dir = os.path.join(
        config.output_dir,
        config.model_name,
        config.concept_class,
        config.concept.replace(" ", "_"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save full results as pickle (includes tensors)
    pkl_path = os.path.join(output_dir, f"results_{config.method}_{config.loss_type}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=5)
    logger.info("Results saved to %s", pkl_path)

    # Save summary as JSON (no tensors)
    summary = {
        "concept": config.concept,
        "concept_class": config.concept_class,
        "model": config.model_name,
        "baseline_mean_cos_sim": baseline["mean_cosine_similarity"],
        "baseline_prefix": baseline["prefix_text"],
    }
    for method_name, method_result in results["methods"].items():
        if method_name == "gradient":
            summary[f"{method_name}_best_cos_sim"] = method_result["best_cosine_similarity"]
            summary[f"{method_name}_discrete_text"] = method_result["discrete_text"]
            summary[f"{method_name}_discrete_cos_sims"] = method_result["discrete_cosine_similarities"]
            summary[f"{method_name}_time"] = method_result["total_time_seconds"]
        elif method_name == "pez":
            summary[f"{method_name}_best_discrete_cos_sim"] = method_result["best_discrete_cosine_similarity"]
            summary[f"{method_name}_discrete_text"] = method_result["discrete_text"]
            summary[f"{method_name}_per_layer_cos_sims"] = method_result["per_layer_discrete_cosine_similarities"]
            summary[f"{method_name}_time"] = method_result["total_time_seconds"]
        elif method_name in ("pez_v2", "gcg"):
            summary[f"{method_name}_handcrafted_mean"] = method_result["handcrafted_mean_cos_sim"]
            summary[f"{method_name}_optimized_mean"] = method_result["optimized_mean_cos_sim"]
            summary[f"{method_name}_improvement"] = method_result["improvement"]
            summary[f"{method_name}_original_prefix"] = method_result["original_prefix"]
            summary[f"{method_name}_optimized_prefix"] = method_result["optimized_prefix"]
            summary[f"{method_name}_time"] = method_result["total_time_seconds"]
        elif method_name == "jacobian":
            summary[f"{method_name}_reachability"] = method_result["reachability_scores"]
            summary[f"{method_name}_time"] = method_result["total_time_seconds"]
        elif method_name == "logit_lens":
            summary[f"{method_name}_prefix"] = method_result.get("prefix_text", "")
            summary[f"{method_name}_alignment"] = method_result.get("best_prefix_alignment", {})

    json_path = os.path.join(output_dir, f"summary_{config.method}_{config.loss_type}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", json_path)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prefix optimization for concept vector alignment")

    # Model
    parser.add_argument("--model_name", "-m", default="llama_3.1_8b")
    parser.add_argument("--model_hf_id", default=None, help="Override HF model ID")
    parser.add_argument("--cache_dir", default=None)

    # Concept
    parser.add_argument("--concept", "-c", required=True, help="Concept to optimize for")
    parser.add_argument("--concept_class", default="fears")
    parser.add_argument("--data_dir", default="./data")

    # Direction loading
    parser.add_argument("--steering_method", default="rfm")
    parser.add_argument("--label_type", default="soft")
    parser.add_argument("--rep_token", default="max_attn_per_layer")

    # Optimization
    parser.add_argument("--method", default="all", choices=["gradient", "pez", "pez_v2", "gcg", "jacobian", "logit_lens", "all"])
    parser.add_argument("--prefix_length", "-k", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--loss_type", default="cosine",
                        choices=["cosine", "projection", "normalized_projection", "angular"])
    parser.add_argument("--init_strategy", default="concept_name",
                        choices=["random", "concept_name", "logit_lens"])
    parser.add_argument("--target_position", default="last_prefix",
                        choices=["last_prefix", "last_token", "mean_prefix", "mean_all"])
    parser.add_argument("--layers", default="16", help="Layer(s): 'all', '16', '12-24', '5,10,15,20'")
    parser.add_argument("--n_statements", type=int, default=1)
    parser.add_argument("--lambda_prox", type=float, default=0.01)
    parser.add_argument("--lambda_norm", type=float, default=0.001)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Jacobian
    parser.add_argument("--jacobian_rank", type=int, default=64)

    # Output
    parser.add_argument("--output_dir", default="outputs/prefix_optimization")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Build config
    MODEL_HF_IDS = {
        "llama_3.1_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama_3.1_70b": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "llama_3.3_70b": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "qwen-14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "qwen-32b": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    }
    MODEL_DIMS = {
        "llama_3.1_8b": (4096, 32),
        "llama_3.1_70b": (8192, 80),
        "llama_3.3_70b": (8192, 80),
        "qwen-14b": (5120, 24),
        "qwen-32b": (5120, 64),
    }

    hf_id = args.model_hf_id or MODEL_HF_IDS.get(args.model_name, args.model_name)
    hidden_dim, n_layers = MODEL_DIMS.get(args.model_name, (4096, 32))

    config = PrefixOptConfig(
        model_name=args.model_name,
        model_hf_id=hf_id,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        cache_dir=args.cache_dir,
        concept=args.concept,
        concept_class=args.concept_class,
        data_dir=args.data_dir,
        steering_method=args.steering_method,
        label_type=args.label_type,
        rep_token=args.rep_token,
        method=args.method,
        prefix_length=args.prefix_length,
        n_steps=args.n_steps,
        lr=args.lr,
        optimizer=args.optimizer,
        loss_type=args.loss_type,
        init_strategy=args.init_strategy,
        target_position=args.target_position,
        layers=args.layers,
        n_statements=args.n_statements,
        lambda_prox=args.lambda_prox,
        lambda_norm=args.lambda_norm,
        grad_clip=args.grad_clip,
        jacobian_rank=args.jacobian_rank,
        output_dir=args.output_dir,
        log_every=args.log_every,
        seed=args.seed,
    )

    run_single_experiment(config)


if __name__ == "__main__":
    main()
