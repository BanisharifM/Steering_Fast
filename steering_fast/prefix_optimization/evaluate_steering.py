"""Evaluate whether GCG-optimized prefixes produce better steered outputs.

Takes the optimized prefixes from GCG results and runs them through the
steering pipeline (generation with concept direction hooks) to see if
higher cos_sim translates to better concept-aligned text.

Usage:
    python -m steering_fast.prefix_optimization.evaluate_steering \
        --results_dir outputs/prefix_optimization/llama_3.1_8b/fears \
        --data_dir ./data \
        --n_concepts 5
"""

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


def load_gcg_results(results_dir: str, n_concepts: int = 5) -> List[Dict]:
    """Load GCG results for evaluation."""
    rows = []
    for concept_dir in sorted(os.listdir(results_dir)):
        summary_path = os.path.join(results_dir, concept_dir, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            d = json.load(f)
        if "gcg_layer16_prefix" in d:
            rows.append(d)
        if len(rows) >= n_concepts:
            break
    return rows


def generate_steered_text(model, tokenizer, concept, prefix_text, directions,
                          layers_to_control, coefficient, max_tokens=50):
    """Generate text with steering hooks applied.

    Uses the same hook mechanism as the original pipeline.
    """
    # Build the full prompt
    statement = "What are your thoughts on the following statement?\nStatement: Life is full of interesting experiences."
    content = prefix_text + statement
    chat = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Register hooks
    hooks = []
    layers = model.model.layers

    for layer_idx in layers_to_control:
        if layer_idx not in directions:
            continue
        control_vec = directions[layer_idx]
        if not torch.is_tensor(control_vec):
            control_vec = torch.tensor(control_vec)
        if control_vec.ndim == 1:
            control_vec = control_vec.view(1, 1, -1)
        elif control_vec.ndim == 2:
            control_vec = control_vec.unsqueeze(0)

        block = layers[layer_idx]

        def make_hook(cv=control_vec, coef=coefficient):
            def hook_fn(module, inputs, output):
                new_hidden = output[0] if isinstance(output, tuple) else output
                delta = coef * cv.to(dtype=new_hidden.dtype, device=new_hidden.device)
                new_hidden = new_hidden + delta
                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]
                return new_hidden
            return hook_fn

        h = block.register_forward_hook(make_hook())
        hooks.append(h)

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--n_concepts", type=int, default=5)
    parser.add_argument("--model_name", default="llama_3.1_8b")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--coefficient", type=float, default=0.8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Load model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
    data_dir = os.path.abspath(args.data_dir)

    from ..utils import core_imports_and_cwd
    with core_imports_and_cwd(data_dir):
        from utils import select_llm
        import utils as orig_utils
        orig_utils.DATA_DIR = data_dir
        if args.cache_dir:
            orig_utils.CACHE_DIR = args.cache_dir
        llm = select_llm(args.model_name)

    model = llm.language_model
    tokenizer = llm.tokenizer

    # Load GCG results
    gcg_results = load_gcg_results(args.results_dir, args.n_concepts)
    logger.info("Loaded %d concept results for steering evaluation", len(gcg_results))

    layers_to_control = list(range(1, model.config.num_hidden_layers))
    coef = args.coefficient

    results = []

    for r in gcg_results:
        concept = r["concept"]
        handcrafted_prefix = f"Personify someone who is terrified of {concept}.  "
        gcg_prefix = r.get("gcg_layer16_prefix", handcrafted_prefix)

        # Load directions
        label_suffix = "_softlabels"
        direction_file = os.path.join(
            data_dir, "directions",
            f"rfm_{concept}_tokenidx_max_attn_per_layer_block{label_suffix}_{args.model_name}.pkl"
        )
        if not os.path.exists(direction_file):
            logger.warning("No directions for %s, skipping", concept)
            continue

        with open(direction_file, "rb") as f:
            directions = pickle.load(f)

        logger.info("=" * 60)
        logger.info("CONCEPT: %s", concept)
        logger.info("  Hand-crafted prefix: %s", handcrafted_prefix[:60])
        logger.info("  GCG prefix: %s", gcg_prefix[:60])

        # Generate with hand-crafted prefix (no steering)
        text_handcrafted_no_steer = generate_steered_text(
            model, tokenizer, concept, handcrafted_prefix, {},
            [], 0, max_tokens=50
        )

        # Generate with hand-crafted prefix + steering
        text_handcrafted_steered = generate_steered_text(
            model, tokenizer, concept, handcrafted_prefix, directions,
            layers_to_control, coef, max_tokens=50
        )

        # Generate with GCG prefix + steering
        text_gcg_steered = generate_steered_text(
            model, tokenizer, concept, gcg_prefix, directions,
            layers_to_control, coef, max_tokens=50
        )

        # Generate with GCG prefix (no steering)
        text_gcg_no_steer = generate_steered_text(
            model, tokenizer, concept, gcg_prefix, {},
            [], 0, max_tokens=50
        )

        logger.info("  --- No steering ---")
        logger.info("  Hand-crafted: %s", text_handcrafted_no_steer[:120])
        logger.info("  GCG prefix:   %s", text_gcg_no_steer[:120])
        logger.info("  --- With steering (coef=%.2f) ---", coef)
        logger.info("  Hand-crafted: %s", text_handcrafted_steered[:120])
        logger.info("  GCG prefix:   %s", text_gcg_steered[:120])

        results.append({
            "concept": concept,
            "handcrafted_prefix": handcrafted_prefix,
            "gcg_prefix": gcg_prefix,
            "l16_handcrafted_cos": r.get("gcg_layer16_handcrafted"),
            "l16_optimized_cos": r.get("gcg_layer16_optimized"),
            "l16_improvement": r.get("gcg_layer16_improvement"),
            "text_handcrafted_no_steer": text_handcrafted_no_steer,
            "text_handcrafted_steered": text_handcrafted_steered,
            "text_gcg_no_steer": text_gcg_no_steer,
            "text_gcg_steered": text_gcg_steered,
        })

        torch.cuda.empty_cache()

    # Save results
    output_dir = os.path.join(args.results_dir, "steering_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "steering_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("STEERING EVALUATION SUMMARY")
    logger.info("=" * 70)
    for r in results:
        logger.info("\nConcept: %s (cos improvement: %+.3f)", r["concept"], r.get("l16_improvement", 0))
        logger.info("  Hand-crafted + steer: %s", r["text_handcrafted_steered"][:100])
        logger.info("  GCG prefix + steer:   %s", r["text_gcg_steered"][:100])

    logger.info("\nResults saved to %s", output_dir)


if __name__ == "__main__":
    main()
