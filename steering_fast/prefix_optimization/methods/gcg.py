"""GCG (Greedy Coordinate Gradient) for discrete prefix optimization.

Based on Zou et al. (2023). Never leaves discrete token space.
At each step: compute token-level gradients, select top-k candidates
for one position, evaluate all candidates via BATCHED forward pass,
keep the best.

v2 improvements:
- Batched candidate evaluation (single forward pass for all k candidates)
- Random position selection (breaks cycling plateau)
- Multi-position candidates per step (evaluate across positions)
"""

import logging
import random
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .pez_v2 import (
    build_prompt_parts,
    load_layer_to_token,
    tokenize_prompt_parts,
)
from ..losses import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)


def forward_batch_with_prefix_variants(
    model,
    all_ids: torch.Tensor,
    prefix_start: int,
    prefix_end: int,
    candidate_prefix_ids_batch: torch.Tensor,
    device: torch.device,
) -> list:
    """Batched forward pass for multiple prefix variants.

    Constructs a batch where each element is the full prompt with a different
    prefix token sequence. All share the same non-prefix tokens.

    Args:
        model: Frozen LLM
        all_ids: (seq_len,) base token IDs
        prefix_start, prefix_end: prefix boundaries
        candidate_prefix_ids_batch: (B, prefix_len) different prefix token IDs
        device: cuda device

    Returns:
        List of hidden_states, one per candidate (each is a tuple of layer tensors)
    """
    B = candidate_prefix_ids_batch.shape[0]
    seq_len = len(all_ids)
    prefix_len = prefix_end - prefix_start

    # Build batch of input IDs -- stay in model's native dtype (bf16)
    batch_ids = all_ids.unsqueeze(0).expand(B, -1).clone()  # (B, seq_len)
    batch_ids[:, prefix_start:prefix_end] = candidate_prefix_ids_batch

    attention_mask = torch.ones(B, seq_len, device=device, dtype=torch.long)

    with torch.no_grad():
        # Use model.model (base transformer) NOT model (CausalLM with lm_head).
        # We only need hidden_states, not logits. Skipping lm_head saves ~2 GB
        # (avoids allocating (B, seq_len, 128256) logits tensor).
        outputs = model.model(
            input_ids=batch_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    return outputs


def compute_token_gradients(
    model,
    all_ids: torch.Tensor,
    prefix_start: int,
    prefix_end: int,
    current_prefix_ids: torch.Tensor,
    active_layers: List[int],
    active_directions: Dict[int, torch.Tensor],
    layer_to_token: Optional[Dict[int, int]],
    loss_fn,
    device: torch.device,
) -> torch.Tensor:
    """Compute gradient of loss w.r.t. one-hot token indicators at prefix positions.

    Returns:
        token_gradients: (prefix_len, vocab_size)
    """
    embedding_matrix = model.model.embed_tokens.weight  # (V, d)
    vocab_size = embedding_matrix.shape[0]
    prefix_len = prefix_end - prefix_start

    # Build full embedding sequence with one-hot prefix
    with torch.no_grad():
        all_embeds = model.model.embed_tokens(all_ids.unsqueeze(0)).float()  # (1, seq_len, d)

    # Create one-hot for prefix positions
    one_hot = F.one_hot(current_prefix_ids, num_classes=vocab_size).float()
    one_hot = one_hot.detach().requires_grad_(True)
    prefix_embeds = one_hot @ embedding_matrix.float()  # (prefix_len, d)

    # Replace prefix in the full embedding
    combined = all_embeds.clone()
    combined[0, prefix_start:prefix_end, :] = prefix_embeds

    attention_mask = torch.ones(1, combined.shape[1], device=device, dtype=torch.long)

    # Use base transformer (no lm_head) -- saves memory, we only need hidden_states
    outputs = model.model(
        inputs_embeds=combined,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    for layer_idx in active_layers:
        hs_idx = layer_idx + 1
        if hs_idx >= len(outputs.hidden_states):
            continue
        h = outputs.hidden_states[hs_idx]
        tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
        act = h[0, tok_pos, :].float()
        target = active_directions[layer_idx]
        loss_val, _ = loss_fn(act, target)
        total_loss = total_loss + loss_val / len(active_layers)

    total_loss.backward()
    return one_hot.grad.detach()


def evaluate_candidates_batched(
    model,
    all_ids: torch.Tensor,
    prefix_start: int,
    prefix_end: int,
    current_prefix_ids: torch.Tensor,
    position: int,
    candidate_token_ids: torch.Tensor,
    active_layers: List[int],
    active_directions: Dict[int, torch.Tensor],
    layer_to_token: Optional[Dict[int, int]],
    loss_fn,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """Evaluate candidate token replacements at a specific position using BATCHED forward passes.

    Args:
        position: which prefix position to replace (0-indexed within prefix)
        candidate_token_ids: (n_candidates,) token IDs to try
        batch_size: how many candidates to evaluate per forward pass

    Returns:
        (best_token_id, best_loss, best_cos_sim)
    """
    n_candidates = len(candidate_token_ids)
    prefix_len = prefix_end - prefix_start

    best_loss = float("inf")
    best_token = current_prefix_ids[position].item()
    best_cos_sim = -1.0

    for batch_start in range(0, n_candidates, batch_size):
        batch_end = min(batch_start + batch_size, n_candidates)
        batch_cands = candidate_token_ids[batch_start:batch_end]
        B = len(batch_cands)

        # Build batch of prefix variants: each has one different token at `position`
        batch_prefix_ids = current_prefix_ids.unsqueeze(0).expand(B, -1).clone()  # (B, prefix_len)
        batch_prefix_ids[:, position] = batch_cands

        with torch.no_grad():
            outputs = forward_batch_with_prefix_variants(
                model, all_ids, prefix_start, prefix_end, batch_prefix_ids, device
            )

            # Evaluate each candidate
            for b in range(B):
                total_loss = 0.0
                step_cos_sims = []

                for layer_idx in active_layers:
                    hs_idx = layer_idx + 1
                    if hs_idx >= len(outputs.hidden_states):
                        continue
                    h = outputs.hidden_states[hs_idx]  # (B, seq_len, d)
                    tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
                    act = h[b, tok_pos, :].to(torch.float32)
                    target = active_directions[layer_idx]
                    loss_val, metrics = loss_fn(act, target)
                    total_loss += loss_val.item() / len(active_layers)
                    step_cos_sims.append(metrics["cosine_similarity"])

                mean_cos = sum(step_cos_sims) / len(step_cos_sims) if step_cos_sims else 0.0

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_token = batch_cands[b].item()
                    best_cos_sim = mean_cos

    return best_token, best_loss, best_cos_sim


def evaluate_two_position_swap(
    model,
    all_ids: torch.Tensor,
    prefix_start: int,
    prefix_end: int,
    current_prefix_ids: torch.Tensor,
    token_grads: torch.Tensor,
    active_layers: List[int],
    active_directions: Dict[int, torch.Tensor],
    layer_to_token: Optional[Dict[int, int]],
    loss_fn,
    device: torch.device,
    top_per_pos: int = 8,
    batch_size: int = 64,
) -> tuple:
    """Try swapping 2 positions simultaneously.

    For each pair of positions, take top_per_pos candidates per position,
    form all combinations (top_per_pos^2 per pair), evaluate the best pairs.

    Returns:
        (pos1, tok1, pos2, tok2, best_cos_sim) or None if no improvement
    """
    prefix_len = prefix_end - prefix_start

    # Get top candidates per position from gradients
    per_pos_candidates = {}
    for p in range(prefix_len):
        topk = (-token_grads[p]).topk(top_per_pos)
        per_pos_candidates[p] = topk.indices

    # Try all position pairs, but limit to most promising
    # Sort positions by gradient magnitude (most improvable first)
    pos_importance = [(-token_grads[p]).max().item() for p in range(prefix_len)]
    sorted_positions = sorted(range(prefix_len), key=lambda p: pos_importance[p], reverse=True)
    top_positions = sorted_positions[:5]  # only try top-5 most important positions

    best_overall_cos = -1.0
    best_swap = None

    for i, p1 in enumerate(top_positions):
        for p2 in top_positions[i + 1:]:
            cands_p1 = per_pos_candidates[p1]
            cands_p2 = per_pos_candidates[p2]

            # Form all combinations
            candidates = []
            for t1 in cands_p1:
                for t2 in cands_p2:
                    candidates.append((t1.item(), t2.item()))

            # Evaluate in batches
            for batch_start in range(0, len(candidates), batch_size):
                batch_end = min(batch_start + batch_size, len(candidates))
                batch = candidates[batch_start:batch_end]
                B = len(batch)

                batch_prefix_ids = current_prefix_ids.unsqueeze(0).expand(B, -1).clone()
                for b_idx, (t1, t2) in enumerate(batch):
                    batch_prefix_ids[b_idx, p1] = t1
                    batch_prefix_ids[b_idx, p2] = t2

                with torch.no_grad():
                    outputs = forward_batch_with_prefix_variants(
                        model, all_ids, prefix_start, prefix_end, batch_prefix_ids, device
                    )

                    for b_idx in range(B):
                        step_cos_sims = []
                        for layer_idx in active_layers:
                            hs_idx = layer_idx + 1
                            if hs_idx >= len(outputs.hidden_states):
                                continue
                            h = outputs.hidden_states[hs_idx]
                            tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
                            act = h[b_idx, tok_pos, :].to(torch.float32)
                            target = active_directions[layer_idx]
                            _, metrics = loss_fn(act, target)
                            step_cos_sims.append(metrics["cosine_similarity"])

                        mean_cos = sum(step_cos_sims) / len(step_cos_sims) if step_cos_sims else 0.0
                        if mean_cos > best_overall_cos:
                            best_overall_cos = mean_cos
                            t1, t2 = batch[b_idx]
                            best_swap = (p1, t1, p2, t2)

    return best_swap, best_overall_cos


def optimize_prefix_gcg(
    model,
    tokenizer,
    directions: Dict[int, torch.Tensor],
    concept: str,
    statements: List[str],
    config,
) -> Dict:
    """GCG: Greedy Coordinate Gradient for discrete prefix optimization.

    At each step:
    1. Compute token-level gradients for all prefix positions
    2. Pick a RANDOM position (avoids cycling plateau)
    3. Get top-k candidates from gradient
    4. Evaluate all candidates with BATCHED forward passes
    5. Keep the best replacement (only if it improves)
    """
    device = next(model.parameters()).device
    embedding_matrix = model.model.embed_tokens.weight.detach()

    layer_to_token = load_layer_to_token(
        config.data_dir, concept, config.model_name, config.head_agg
    )

    clean_directions = {}
    for layer_idx, v in directions.items():
        v_flat = v.flatten().float().to(device)
        v_flat = v_flat / v_flat.norm()
        clean_directions[layer_idx] = v_flat

    layers = config.get_layers()
    active_directions = {l: clean_directions[l] for l in layers if l in clean_directions}
    active_layers = sorted(active_directions.keys())

    # Build prompt
    stmt = statements[0].strip()
    prefix_text, suffix_text, full_positive, full_negative = build_prompt_parts(
        concept, config.concept_class, stmt, tokenizer
    )
    all_ids, prefix_start, prefix_end = tokenize_prompt_parts(
        full_positive, prefix_text, tokenizer, device
    )
    prefix_len = prefix_end - prefix_start

    logger.info("GCG: %d prefix tokens [%d:%d], %d layers, top_k=%d, batch=%d",
                prefix_len, prefix_start, prefix_end, len(active_layers),
                config.gcg_topk, config.gcg_batch_size)

    # Initialize from hand-crafted prefix
    current_prefix_ids = all_ids[prefix_start:prefix_end].clone()
    loss_fn = LOSS_FUNCTIONS[config.loss_type]

    # Evaluate initial state (hand-crafted baseline at correct positions)
    init_cos_sims = {}
    with torch.no_grad():
        mask = torch.ones(1, len(all_ids), device=device, dtype=torch.long)
        outputs = model.model(input_ids=all_ids.unsqueeze(0), attention_mask=mask,
                              output_hidden_states=True, use_cache=False)
        for layer_idx in active_layers:
            hs_idx = layer_idx + 1
            if hs_idx >= len(outputs.hidden_states):
                continue
            h = outputs.hidden_states[hs_idx]
            tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
            act = h[0, tok_pos, :].to(torch.float32)
            cs = F.cosine_similarity(act.unsqueeze(0), active_directions[layer_idx].unsqueeze(0)).item()
            init_cos_sims[layer_idx] = cs

    init_mean_cos = sum(init_cos_sims.values()) / len(init_cos_sims) if init_cos_sims else 0
    logger.info("GCG initial (hand-crafted) mean cos_sim: %.4f", init_mean_cos)
    logger.info("  Per-layer: %s", {l: f"{v:.4f}" for l, v in init_cos_sims.items()})

    # Evaluate NO-PREFIX baseline (the negative prompt, without concept prefix)
    # This is the baseline the RFM was trained to discriminate FROM
    no_prefix_cos_sims = {}
    neg_ids = tokenizer.encode(full_negative, add_special_tokens=False, return_tensors="pt")[0].to(device)
    with torch.no_grad():
        neg_mask = torch.ones(1, len(neg_ids), device=device, dtype=torch.long)
        neg_outputs = model.model(input_ids=neg_ids.unsqueeze(0), attention_mask=neg_mask,
                                  output_hidden_states=True, use_cache=False)
        for layer_idx in active_layers:
            hs_idx = layer_idx + 1
            if hs_idx >= len(neg_outputs.hidden_states):
                continue
            h = neg_outputs.hidden_states[hs_idx]
            tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
            act = h[0, tok_pos, :].to(torch.float32)
            cs = F.cosine_similarity(act.unsqueeze(0), active_directions[layer_idx].unsqueeze(0)).item()
            no_prefix_cos_sims[layer_idx] = cs

    no_prefix_mean = sum(no_prefix_cos_sims.values()) / len(no_prefix_cos_sims) if no_prefix_cos_sims else 0
    handcrafted_delta = init_mean_cos - no_prefix_mean
    logger.info("No-prefix baseline mean cos_sim: %.4f", no_prefix_mean)
    logger.info("Hand-crafted delta (vs no-prefix): %.4f", handcrafted_delta)

    # Optimization
    rng = random.Random(config.seed)
    loss_curve = []
    metrics_history = []
    best_cos_sim = init_mean_cos
    best_prefix_ids = current_prefix_ids.clone()
    n_improvements = 0
    steps_without_improvement = 0
    start_time = time.time()

    for step in range(config.n_steps):
        # 1. Token gradients
        token_grads = compute_token_gradients(
            model, all_ids, prefix_start, prefix_end, current_prefix_ids,
            active_layers, active_directions, layer_to_token, loss_fn, device,
        )

        accepted = False

        # Try 2-position swap if single-position has plateaued for 20+ steps
        if config.gcg_multi_swap and steps_without_improvement >= 20:
            logger.info("  Step %d: trying 2-position swap (plateaued for %d steps)...",
                        step, steps_without_improvement)
            swap_result, swap_cos = evaluate_two_position_swap(
                model, all_ids, prefix_start, prefix_end, current_prefix_ids,
                token_grads, active_layers, active_directions,
                layer_to_token, loss_fn, device,
            )
            if swap_result and swap_cos > best_cos_sim:
                p1, t1, p2, t2 = swap_result
                current_prefix_ids[p1] = t1
                current_prefix_ids[p2] = t2
                best_cos_sim = swap_cos
                best_prefix_ids = current_prefix_ids.clone()
                n_improvements += 1
                steps_without_improvement = 0
                accepted = True
                logger.info("  2-position swap ACCEPTED: pos %d+%d, cos=%.4f", p1, p2, swap_cos)

        if not accepted:
            # Standard single-position GCG
            # 2. Random position selection
            pos = rng.randint(0, prefix_len - 1)

            # 3. Top-k candidates
            topk = (-token_grads[pos]).topk(config.gcg_topk)
            candidate_ids = topk.indices

            # 4. Batched evaluation
            best_token, best_loss, step_cos_sim = evaluate_candidates_batched(
                model, all_ids, prefix_start, prefix_end, current_prefix_ids,
                pos, candidate_ids, active_layers, active_directions,
                layer_to_token, loss_fn, device, config.gcg_batch_size,
            )

            # 5. Accept if improved
            old_token = current_prefix_ids[pos].item()
            if step_cos_sim > best_cos_sim:
                current_prefix_ids[pos] = best_token
                best_cos_sim = step_cos_sim
                best_prefix_ids = current_prefix_ids.clone()
                n_improvements += 1
                steps_without_improvement = 0
                accepted = True
            else:
                steps_without_improvement += 1

        loss_curve.append(best_loss)

        if step % config.log_every == 0 or step == config.n_steps - 1:
            current_text = tokenizer.decode(current_prefix_ids.tolist(), skip_special_tokens=False)
            elapsed = time.time() - start_time
            step_metrics = {
                "step": step,
                "position": pos,
                "accepted": accepted,
                "step_cos_sim": step_cos_sim,
                "best_cos_sim": best_cos_sim,
                "old_token": tokenizer.convert_ids_to_tokens([old_token])[0],
                "new_token": tokenizer.convert_ids_to_tokens([best_token])[0] if accepted else "(rejected)",
                "current_prefix": current_text,
                "elapsed_seconds": elapsed,
                "n_improvements": n_improvements,
            }
            metrics_history.append(step_metrics)
            logger.info(
                "GCG Step %d/%d pos=%d | cos=%.4f (best=%.4f) | %s%s | %s",
                step, config.n_steps, pos, step_cos_sim, best_cos_sim,
                "ACCEPT " if accepted else "reject ",
                tokenizer.convert_ids_to_tokens([best_token])[0] if accepted else "",
                current_text[:70],
            )

    # Final evaluation
    final_prefix_ids = best_prefix_ids.tolist()
    final_text = tokenizer.decode(final_prefix_ids, skip_special_tokens=True)
    final_cos_sims = {}

    with torch.no_grad():
        final_ids = all_ids.clone()
        final_ids[prefix_start:prefix_end] = best_prefix_ids
        mask = torch.ones(1, len(all_ids), device=device, dtype=torch.long)
        outputs = model.model(input_ids=final_ids.unsqueeze(0), attention_mask=mask,
                              output_hidden_states=True, use_cache=False)
        for layer_idx in active_layers:
            hs_idx = layer_idx + 1
            if hs_idx >= len(outputs.hidden_states):
                continue
            h = outputs.hidden_states[hs_idx]
            tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
            act = h[0, tok_pos, :].to(torch.float32)
            cs = F.cosine_similarity(act.unsqueeze(0), active_directions[layer_idx].unsqueeze(0)).item()
            final_cos_sims[layer_idx] = cs

    final_mean = sum(final_cos_sims.values()) / len(final_cos_sims) if final_cos_sims else 0
    total_time = time.time() - start_time

    optimized_delta = final_mean - no_prefix_mean

    logger.info("=" * 60)
    logger.info("GCG RESULTS:")
    logger.info("  No-prefix baseline cos_sim:  %.4f", no_prefix_mean)
    logger.info("  Hand-crafted prefix cos_sim: %.4f (delta=%+.4f)", init_mean_cos, handcrafted_delta)
    logger.info("  Optimized prefix cos_sim:    %.4f (delta=%+.4f)", final_mean, optimized_delta)
    logger.info("  Improvement over hand-crafted: %+.4f", final_mean - init_mean_cos)
    logger.info("  Improvement over no-prefix:    %+.4f", optimized_delta)
    logger.info("  Accepted swaps: %d / %d steps", n_improvements, config.n_steps)
    logger.info("  Original: '%s'", prefix_text)
    logger.info("  Optimized: '%s'", final_text)
    logger.info("  Time: %.1f seconds (%.1f s/step)", total_time, total_time / max(config.n_steps, 1))
    logger.info("=" * 60)

    return {
        "method": "gcg",
        "concept": concept,
        "original_prefix": prefix_text,
        "optimized_prefix": final_text,
        "optimized_token_ids": final_prefix_ids,
        "no_prefix_per_layer_cos_sims": no_prefix_cos_sims,
        "no_prefix_mean_cos_sim": no_prefix_mean,
        "handcrafted_per_layer_cos_sims": init_cos_sims,
        "handcrafted_mean_cos_sim": init_mean_cos,
        "handcrafted_delta": handcrafted_delta,
        "optimized_per_layer_cos_sims": final_cos_sims,
        "optimized_mean_cos_sim": final_mean,
        "optimized_delta": optimized_delta,
        "improvement_over_handcrafted": final_mean - init_mean_cos,
        "n_improvements": n_improvements,
        "loss_curve": loss_curve,
        "metrics_history": metrics_history,
        "total_time_seconds": total_time,
        "layer_to_token": layer_to_token,
        "config": {
            "loss_type": config.loss_type,
            "n_steps": config.n_steps,
            "gcg_topk": config.gcg_topk,
            "gcg_batch_size": config.gcg_batch_size,
            "layers": active_layers,
        },
    }
