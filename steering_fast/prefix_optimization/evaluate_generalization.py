"""Evaluate how well GCG-optimized prefixes generalize across statements.

The GCG optimization used a single statement. This script tests whether
the optimized prefix maintains its improvement across different statements.

Also tests layer-group analysis: which layer groups benefit most.

Usage:
    python -m steering_fast.prefix_optimization.evaluate_generalization \
        --results_dir outputs/prefix_optimization/llama_3.1_8b/fears \
        --data_dir ./data --n_concepts 10 --n_statements 20
"""

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--n_statements", type=int, default=20)
    parser.add_argument("--model_name", default="llama_3.1_8b")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--concept_class", default="fears")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Load model
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
    for param in model.parameters():
        param.requires_grad = False

    # Load statements
    statements = []
    for fname in ["class_0.txt", "class_1.txt"]:
        path = os.path.join(data_dir, "general_statements", fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                statements.extend(line.strip() for line in f if line.strip())

    test_statements = statements[:args.n_statements]
    logger.info("Testing generalization across %d statements", len(test_statements))

    # Load GCG results
    from .methods.pez_v2 import build_prompt_parts, load_layer_to_token, tokenize_prompt_parts

    concept_dirs = sorted(os.listdir(args.results_dir))
    concepts_tested = 0
    all_results = []

    for concept_dir in concept_dirs:
        summary_path = os.path.join(args.results_dir, concept_dir, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            summary = json.load(f)

        concept = summary.get("concept")
        if not concept:
            continue

        gcg_prefix_text = summary.get("gcg_layer16_prefix")
        if not gcg_prefix_text:
            continue

        # Load directions
        label_suffix = "_softlabels"
        direction_file = os.path.join(
            data_dir, "directions",
            f"rfm_{concept}_tokenidx_max_attn_per_layer_block{label_suffix}_{args.model_name}.pkl"
        )
        if not os.path.exists(direction_file):
            continue

        with open(direction_file, "rb") as f:
            directions = pickle.load(f)

        # Load layer_to_token
        layer_to_token = load_layer_to_token(data_dir, concept, args.model_name)

        # Normalize directions
        device = next(model.parameters()).device
        clean_directions = {}
        for layer_idx, v in directions.items():
            v_flat = v.flatten().float().to(device)
            v_flat = v_flat / v_flat.norm()
            clean_directions[layer_idx] = v_flat

        active_layers = sorted(l for l in clean_directions if l >= 1)

        # Define layer groups
        layer_groups = {
            "early (1-10)": [l for l in active_layers if 1 <= l <= 10],
            "middle (11-20)": [l for l in active_layers if 11 <= l <= 20],
            "late (21-31)": [l for l in active_layers if 21 <= l <= 31],
        }

        logger.info("=" * 60)
        logger.info("CONCEPT: %s", concept)

        # Test across multiple statements
        handcrafted_scores = []
        gcg_scores = []
        handcrafted_per_group = {g: [] for g in layer_groups}
        gcg_per_group = {g: [] for g in layer_groups}

        for stmt_idx, stmt in enumerate(test_statements):
            # Build prompts with this statement
            prefix_text, suffix_text, full_positive, full_negative = build_prompt_parts(
                concept, args.concept_class, stmt, tokenizer
            )
            all_ids, prefix_start, prefix_end = tokenize_prompt_parts(
                full_positive, prefix_text, tokenizer, device
            )

            # Handcrafted: just forward pass with original prefix
            with torch.no_grad():
                mask = torch.ones(1, len(all_ids), device=device, dtype=torch.long)
                outputs = model.model(input_ids=all_ids.unsqueeze(0), attention_mask=mask,
                                      output_hidden_states=True, use_cache=False)

                hc_cos_sims = {}
                for layer_idx in active_layers:
                    hs_idx = layer_idx + 1
                    if hs_idx >= len(outputs.hidden_states):
                        continue
                    h = outputs.hidden_states[hs_idx]
                    tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
                    act = h[0, tok_pos, :].to(torch.float32)
                    cs = F.cosine_similarity(act.unsqueeze(0), clean_directions[layer_idx].unsqueeze(0)).item()
                    hc_cos_sims[layer_idx] = cs

            # GCG prefix: tokenize GCG prefix and replace in the prompt
            gcg_token_ids = tokenizer.encode(gcg_prefix_text, add_special_tokens=False)
            # Ensure same length as original prefix
            orig_prefix_len = prefix_end - prefix_start
            if len(gcg_token_ids) > orig_prefix_len:
                gcg_token_ids = gcg_token_ids[:orig_prefix_len]
            elif len(gcg_token_ids) < orig_prefix_len:
                # Pad with the last token
                gcg_token_ids = gcg_token_ids + [gcg_token_ids[-1]] * (orig_prefix_len - len(gcg_token_ids))

            gcg_ids = all_ids.clone()
            gcg_ids[prefix_start:prefix_end] = torch.tensor(gcg_token_ids, device=device)

            with torch.no_grad():
                mask = torch.ones(1, len(gcg_ids), device=device, dtype=torch.long)
                outputs = model.model(input_ids=gcg_ids.unsqueeze(0), attention_mask=mask,
                                      output_hidden_states=True, use_cache=False)

                gcg_cos_sims = {}
                for layer_idx in active_layers:
                    hs_idx = layer_idx + 1
                    if hs_idx >= len(outputs.hidden_states):
                        continue
                    h = outputs.hidden_states[hs_idx]
                    tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
                    act = h[0, tok_pos, :].to(torch.float32)
                    cs = F.cosine_similarity(act.unsqueeze(0), clean_directions[layer_idx].unsqueeze(0)).item()
                    gcg_cos_sims[layer_idx] = cs

            # Layer 16 scores
            hc_l16 = hc_cos_sims.get(16, 0)
            gcg_l16 = gcg_cos_sims.get(16, 0)
            handcrafted_scores.append(hc_l16)
            gcg_scores.append(gcg_l16)

            # Per group scores
            for group_name, group_layers in layer_groups.items():
                hc_group = [hc_cos_sims.get(l, 0) for l in group_layers if l in hc_cos_sims]
                gcg_group = [gcg_cos_sims.get(l, 0) for l in group_layers if l in gcg_cos_sims]
                if hc_group:
                    handcrafted_per_group[group_name].append(sum(hc_group) / len(hc_group))
                if gcg_group:
                    gcg_per_group[group_name].append(sum(gcg_group) / len(gcg_group))

        # Aggregate
        import statistics
        hc_mean = statistics.mean(handcrafted_scores)
        gcg_mean = statistics.mean(gcg_scores)
        hc_std = statistics.stdev(handcrafted_scores) if len(handcrafted_scores) > 1 else 0
        gcg_std = statistics.stdev(gcg_scores) if len(gcg_scores) > 1 else 0

        logger.info("  Layer 16 across %d statements:", len(test_statements))
        logger.info("    Handcrafted: %.4f +/- %.4f", hc_mean, hc_std)
        logger.info("    GCG:         %.4f +/- %.4f", gcg_mean, gcg_std)
        logger.info("    Improvement: %+.4f (generalizes: %s)",
                    gcg_mean - hc_mean,
                    "YES" if gcg_mean > hc_mean else "NO")

        # Layer group analysis
        logger.info("  Layer group analysis:")
        for group_name in layer_groups:
            hc_g = statistics.mean(handcrafted_per_group[group_name]) if handcrafted_per_group[group_name] else 0
            gcg_g = statistics.mean(gcg_per_group[group_name]) if gcg_per_group[group_name] else 0
            logger.info("    %s: HC=%.4f, GCG=%.4f, improvement=%+.4f",
                        group_name, hc_g, gcg_g, gcg_g - hc_g)

        result = {
            "concept": concept,
            "n_statements": len(test_statements),
            "l16_handcrafted_mean": hc_mean,
            "l16_handcrafted_std": hc_std,
            "l16_gcg_mean": gcg_mean,
            "l16_gcg_std": gcg_std,
            "l16_improvement": gcg_mean - hc_mean,
            "generalizes": gcg_mean > hc_mean,
        }
        for group_name in layer_groups:
            hc_g = statistics.mean(handcrafted_per_group[group_name]) if handcrafted_per_group[group_name] else 0
            gcg_g = statistics.mean(gcg_per_group[group_name]) if gcg_per_group[group_name] else 0
            safe_name = group_name.split("(")[0].strip()
            result[f"{safe_name}_handcrafted"] = hc_g
            result[f"{safe_name}_gcg"] = gcg_g
            result[f"{safe_name}_improvement"] = gcg_g - hc_g

        all_results.append(result)
        concepts_tested += 1
        if concepts_tested >= args.n_concepts:
            break

    # Save results
    output_dir = os.path.join(args.results_dir, "generalization_analysis")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "generalization_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("GENERALIZATION SUMMARY (%d concepts, %d statements each)", concepts_tested, args.n_statements)
    logger.info("=" * 70)

    n_generalize = sum(1 for r in all_results if r["generalizes"])
    improvements = [r["l16_improvement"] for r in all_results]

    logger.info("  Generalizes: %d/%d concepts (%.1f%%)", n_generalize, len(all_results),
                n_generalize / len(all_results) * 100)
    logger.info("  Mean improvement across statements: %+.4f", statistics.mean(improvements))

    for group_name in ["early", "middle", "late"]:
        group_imps = [r.get(f"{group_name}_improvement", 0) for r in all_results]
        logger.info("  %s layers improvement: %+.4f", group_name, statistics.mean(group_imps))

    logger.info("\nResults saved to %s", output_dir)


if __name__ == "__main__":
    main()
