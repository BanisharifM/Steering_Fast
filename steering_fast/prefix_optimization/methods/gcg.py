"""GCG (Greedy Coordinate Gradient) for discrete prefix optimization.

Based on Zou et al. (2023). Never leaves discrete token space.
At each step: compute token-level gradients, select top-k candidates
for one position, evaluate all candidates via forward pass, keep the best.

This avoids the PEZ projection gap entirely.
"""

import logging
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .pez_v2 import (
    build_prompt_parts,
    forward_with_prefix_embeds,
    load_layer_to_token,
    tokenize_prompt_parts,
)
from ..losses import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)


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
        token_gradients: (prefix_len, vocab_size) -- negative gradient means
        swapping to that token would decrease loss
    """
    embedding_matrix = model.model.embed_tokens.weight  # (V, d)
    vocab_size = embedding_matrix.shape[0]
    prefix_len = prefix_end - prefix_start

    # Create one-hot representation of current prefix tokens
    one_hot = F.one_hot(current_prefix_ids, num_classes=vocab_size).float()  # (prefix_len, V)
    one_hot = one_hot.detach().requires_grad_(True)

    # Compute embeddings via one_hot @ E (differentiable w.r.t. one_hot)
    prefix_embeds = one_hot @ embedding_matrix.float()  # (prefix_len, d)

    # Forward pass
    outputs = forward_with_prefix_embeds(
        model, all_ids, prefix_start, prefix_end, prefix_embeds, device
    )

    # Compute alignment loss
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

    # one_hot.grad: (prefix_len, vocab_size)
    # Negative gradient = tokens that would decrease loss
    return one_hot.grad.detach()  # (prefix_len, V)


def evaluate_candidates(
    model,
    all_ids: torch.Tensor,
    prefix_start: int,
    prefix_end: int,
    current_prefix_ids: torch.Tensor,
    position: int,
    candidate_ids: torch.Tensor,
    active_layers: List[int],
    active_directions: Dict[int, torch.Tensor],
    layer_to_token: Optional[Dict[int, int]],
    loss_fn,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """Evaluate candidate token replacements at a specific position.

    Args:
        position: which prefix position to replace (0-indexed within prefix)
        candidate_ids: (n_candidates,) token IDs to try

    Returns:
        (best_token_id, best_loss, best_cos_sim)
    """
    embedding_matrix = model.model.embed_tokens.weight.detach()
    n_candidates = len(candidate_ids)

    best_loss = float("inf")
    best_token = current_prefix_ids[position].item()
    best_cos_sim = -1.0

    # Evaluate in batches
    for batch_start in range(0, n_candidates, batch_size):
        batch_end = min(batch_start + batch_size, n_candidates)
        batch_candidates = candidate_ids[batch_start:batch_end]

        batch_losses = []
        batch_cos_sims = []

        for cand_id in batch_candidates:
            # Create modified prefix
            modified_ids = current_prefix_ids.clone()
            modified_ids[position] = cand_id

            # Get embeddings
            prefix_embeds = embedding_matrix[modified_ids].float()

            with torch.no_grad():
                outputs = forward_with_prefix_embeds(
                    model, all_ids, prefix_start, prefix_end, prefix_embeds, device
                )

                total_loss = 0.0
                step_cos_sims = []
                for layer_idx in active_layers:
                    hs_idx = layer_idx + 1
                    if hs_idx >= len(outputs.hidden_states):
                        continue
                    h = outputs.hidden_states[hs_idx]
                    tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
                    act = h[0, tok_pos, :].float()
                    target = active_directions[layer_idx]
                    loss_val, metrics = loss_fn(act, target)
                    total_loss += loss_val.item() / len(active_layers)
                    step_cos_sims.append(metrics["cosine_similarity"])

            batch_losses.append(total_loss)
            batch_cos_sims.append(sum(step_cos_sims) / len(step_cos_sims) if step_cos_sims else 0)

        # Find best in this batch
        for i, (loss, cos) in enumerate(zip(batch_losses, batch_cos_sims)):
            if loss < best_loss:
                best_loss = loss
                best_token = batch_candidates[i].item()
                best_cos_sim = cos

    return best_token, best_loss, best_cos_sim


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
    2. Pick one position (cycling or random)
    3. Get top-k candidates from gradient
    4. Evaluate all candidates with forward passes
    5. Keep the best replacement (only if it improves loss)

    This guarantees monotonic improvement and stays in discrete token space.
    """
    device = next(model.parameters()).device
    embedding_matrix = model.model.embed_tokens.weight.detach()

    # Load per-layer token positions
    layer_to_token = load_layer_to_token(
        config.data_dir, concept, config.model_name, config.head_agg
    )

    # Normalize directions
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

    logger.info("GCG: %d prefix tokens [%d:%d], %d layers, top_k=%d",
                prefix_len, prefix_start, prefix_end, len(active_layers), config.gcg_topk)

    # Initialize with hand-crafted prefix tokens
    current_prefix_ids = all_ids[prefix_start:prefix_end].clone()

    loss_fn = LOSS_FUNCTIONS[config.loss_type]

    # Evaluate initial state
    with torch.no_grad():
        init_embeds = embedding_matrix[current_prefix_ids].float()
        outputs = forward_with_prefix_embeds(
            model, all_ids, prefix_start, prefix_end, init_embeds, device
        )
        init_cos_sims = {}
        for layer_idx in active_layers:
            hs_idx = layer_idx + 1
            if hs_idx >= len(outputs.hidden_states):
                continue
            h = outputs.hidden_states[hs_idx]
            tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
            act = h[0, tok_pos, :].float()
            cs = F.cosine_similarity(act.unsqueeze(0), active_directions[layer_idx].unsqueeze(0)).item()
            init_cos_sims[layer_idx] = cs

    init_mean_cos = sum(init_cos_sims.values()) / len(init_cos_sims) if init_cos_sims else 0
    logger.info("GCG initial (hand-crafted) mean cos_sim: %.4f", init_mean_cos)

    # Optimization loop
    loss_curve = []
    metrics_history = []
    best_cos_sim = init_mean_cos
    best_prefix_ids = current_prefix_ids.clone()
    n_improvements = 0
    start_time = time.time()

    for step in range(config.n_steps):
        # 1. Compute token gradients
        token_grads = compute_token_gradients(
            model, all_ids, prefix_start, prefix_end, current_prefix_ids,
            active_layers, active_directions, layer_to_token, loss_fn, device,
        )

        # 2. Pick position to optimize (cycle through)
        pos = step % prefix_len

        # 3. Get top-k candidates (most negative gradient = biggest loss decrease)
        topk = (-token_grads[pos]).topk(config.gcg_topk)
        candidate_ids = topk.indices

        # 4. Evaluate candidates
        best_token, best_loss, step_cos_sim = evaluate_candidates(
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
            accepted = True
        else:
            accepted = False

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
                "GCG Step %d/%d pos=%d | cos=%.4f (best=%.4f) | %s | %s",
                step, config.n_steps, pos, step_cos_sim, best_cos_sim,
                "ACCEPT" if accepted else "reject",
                current_text[:60],
            )

    # Final evaluation
    final_prefix_ids = best_prefix_ids.tolist()
    final_text = tokenizer.decode(final_prefix_ids, skip_special_tokens=True)

    final_cos_sims = {}
    with torch.no_grad():
        final_embeds = embedding_matrix[best_prefix_ids].float()
        outputs = forward_with_prefix_embeds(
            model, all_ids, prefix_start, prefix_end, final_embeds, device
        )
        for layer_idx in active_layers:
            hs_idx = layer_idx + 1
            if hs_idx >= len(outputs.hidden_states):
                continue
            h = outputs.hidden_states[hs_idx]
            tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
            act = h[0, tok_pos, :].float()
            cs = F.cosine_similarity(act.unsqueeze(0), active_directions[layer_idx].unsqueeze(0)).item()
            final_cos_sims[layer_idx] = cs

    final_mean = sum(final_cos_sims.values()) / len(final_cos_sims) if final_cos_sims else 0
    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("GCG RESULTS:")
    logger.info("  Hand-crafted prefix mean cos_sim: %.4f", init_mean_cos)
    logger.info("  Optimized prefix mean cos_sim:    %.4f", final_mean)
    logger.info("  Improvement:                      %.4f", final_mean - init_mean_cos)
    logger.info("  Accepted swaps: %d / %d steps", n_improvements, config.n_steps)
    logger.info("  Original prefix: '%s'", prefix_text)
    logger.info("  Optimized prefix: '%s'", final_text)
    logger.info("  Time: %.1f seconds", total_time)
    logger.info("=" * 60)

    return {
        "method": "gcg",
        "concept": concept,
        "original_prefix": prefix_text,
        "optimized_prefix": final_text,
        "optimized_token_ids": final_prefix_ids,
        "handcrafted_per_layer_cos_sims": init_cos_sims,
        "handcrafted_mean_cos_sim": init_mean_cos,
        "optimized_per_layer_cos_sims": final_cos_sims,
        "optimized_mean_cos_sim": final_mean,
        "improvement": final_mean - init_mean_cos,
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
