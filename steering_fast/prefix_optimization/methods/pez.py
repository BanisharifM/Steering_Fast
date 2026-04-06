"""Method: PEZ (Projected Gradient Descent) for discrete token recovery.

Based on "Hard Prompts Made Easy" (Wen et al., NeurIPS 2023).
Optimizes continuous embeddings with periodic projection to nearest discrete tokens.
Uses Straight-Through Estimator (STE) so the model sees discrete tokens in
the forward pass while gradients flow to the continuous iterates.
"""

import logging
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..losses import LOSS_FUNCTIONS, embedding_proximity_regularization
from .gradient import extract_activations

logger = logging.getLogger(__name__)


def project_to_nearest_tokens(
    embeds: torch.Tensor, embedding_matrix: torch.Tensor
) -> tuple:
    """Project each embedding to its nearest token in the vocabulary.

    Args:
        embeds: (K, d_model) continuous embeddings
        embedding_matrix: (V, d_model) token embedding table

    Returns:
        (projected_embeds, token_ids): projected embeddings and their token IDs
    """
    dists = torch.cdist(embeds.float(), embedding_matrix.float())  # (K, V)
    token_ids = dists.argmin(dim=1)  # (K,)
    projected = embedding_matrix[token_ids].clone()
    return projected, token_ids


def optimize_prefix_pez(
    model,
    tokenizer,
    directions: Dict[int, torch.Tensor],
    concept: str,
    statements: List[str],
    config,
) -> Dict:
    """Run PEZ optimization: project-then-step with STE.

    Key difference from plain gradient method: at each forward pass,
    we project to nearest discrete tokens and use STE so the model
    sees real tokens while gradients update the continuous iterates.

    Args:
        model: Frozen LLM
        tokenizer: Tokenizer
        directions: Dict[layer -> direction vector]
        concept: Concept name
        statements: General statements
        config: PrefixOptConfig
    """
    device = next(model.parameters()).device
    embedding_matrix = model.model.embed_tokens.weight.detach()

    # Normalize directions
    clean_directions = {}
    for layer_idx, v in directions.items():
        v_flat = v.flatten().float().to(device)
        v_flat = v_flat / v_flat.norm()
        clean_directions[layer_idx] = v_flat

    layers = config.get_layers()
    active_directions = {l: clean_directions[l] for l in layers if l in clean_directions}
    active_layers = sorted(active_directions.keys())
    logger.info("PEZ optimizing for %d layers: %s", len(active_layers), active_layers)

    # Initialize prefix embeddings
    from ..initialization import init_concept_name, init_random

    if config.init_strategy == "concept_name":
        prefix_embeds = init_concept_name(
            embedding_matrix, tokenizer, concept, config.prefix_length, config.seed
        )
    else:
        prefix_embeds = init_random(embedding_matrix, config.prefix_length, config.seed)

    # Make prefix optimizable
    prefix_embeds = prefix_embeds.to(device).requires_grad_(True)

    # Pre-compute statement embeddings
    statement_embeds_list = []
    for stmt in statements[: config.n_statements]:
        token_ids = tokenizer.encode(stmt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            stmt_emb = model.model.embed_tokens(token_ids).float()
        statement_embeds_list.append(stmt_emb)

    # Optimizer: PEZ uses higher LR since projection re-discretizes
    optimizer = torch.optim.Adam([prefix_embeds], lr=config.lr)
    loss_fn = LOSS_FUNCTIONS[config.loss_type]

    # Tracking
    loss_curve = []
    metrics_history = []
    best_discrete_cos_sim = -1.0
    best_discrete_ids = None
    best_discrete_embeds = None
    start_time = time.time()

    for step in range(config.n_steps):
        optimizer.zero_grad()

        # STE: project to nearest tokens, but let gradients flow through
        with torch.no_grad():
            projected, token_ids_step = project_to_nearest_tokens(prefix_embeds, embedding_matrix)

        # Straight-Through Estimator: forward uses projected, backward uses prefix_embeds
        ste_embeds = prefix_embeds + (projected.float() - prefix_embeds).detach()
        ste_batch = ste_embeds.unsqueeze(0)  # (1, K, d)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        step_cos_sims = []

        for stmt_emb in statement_embeds_list:
            activations = extract_activations(
                model, ste_batch, stmt_emb, active_layers,
                config.target_position, config.prefix_length,
            )

            for layer_idx in active_layers:
                if layer_idx not in activations:
                    continue
                act = activations[layer_idx]
                target = active_directions[layer_idx]
                loss_val, metrics = loss_fn(act, target)
                total_loss = total_loss + loss_val / (len(active_layers) * len(statement_embeds_list))
                step_cos_sims.append(metrics["cosine_similarity"])

        # Backprop (gradients flow to prefix_embeds via STE)
        total_loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([prefix_embeds], config.grad_clip)

        optimizer.step()

        # Track discrete alignment (this is what we truly care about)
        mean_cos = sum(step_cos_sims) / len(step_cos_sims) if step_cos_sims else 0.0
        loss_curve.append(total_loss.item())

        if mean_cos > best_discrete_cos_sim:
            best_discrete_cos_sim = mean_cos
            best_discrete_ids = token_ids_step.tolist()
            best_discrete_embeds = projected.clone().detach()

        if step % config.log_every == 0 or step == config.n_steps - 1:
            tokens = tokenizer.convert_ids_to_tokens(token_ids_step.tolist())
            elapsed = time.time() - start_time
            step_metrics = {
                "step": step,
                "loss": total_loss.item(),
                "discrete_cosine_similarity": mean_cos,
                "best_discrete_cosine_similarity": best_discrete_cos_sim,
                "tokens": tokens,
                "elapsed_seconds": elapsed,
            }
            metrics_history.append(step_metrics)
            logger.info(
                "PEZ Step %d/%d | loss=%.4f | discrete_cos=%.4f (best=%.4f) | %s",
                step, config.n_steps, total_loss.item(), mean_cos,
                best_discrete_cos_sim, " ".join(tokens[:5]),
            )

    # Final results
    discrete_text = tokenizer.decode(best_discrete_ids, skip_special_tokens=True)

    # Evaluate best discrete prefix per layer
    final_cos_sims = {}
    final_embeds = embedding_matrix[best_discrete_ids].clone().detach().float().unsqueeze(0)
    with torch.no_grad():
        for stmt_emb in statement_embeds_list[:1]:
            acts = extract_activations(
                model, final_embeds.to(device), stmt_emb, active_layers,
                config.target_position, config.prefix_length,
            )
            for layer_idx in active_layers:
                if layer_idx in acts:
                    cs = F.cosine_similarity(
                        acts[layer_idx].unsqueeze(0),
                        active_directions[layer_idx].unsqueeze(0),
                    ).item()
                    final_cos_sims[layer_idx] = cs

    total_time = time.time() - start_time
    logger.info("PEZ completed in %.1f seconds", total_time)
    logger.info("Best discrete cos_sim: %.4f", best_discrete_cos_sim)
    logger.info("Discrete prefix: '%s'", discrete_text)
    logger.info("Per-layer discrete cos_sims: %s",
                {l: f"{v:.4f}" for l, v in final_cos_sims.items()})

    return {
        "method": "pez",
        "concept": concept,
        "config": {
            "loss_type": config.loss_type,
            "init_strategy": config.init_strategy,
            "lr": config.lr,
            "n_steps": config.n_steps,
            "prefix_length": config.prefix_length,
            "target_position": config.target_position,
            "layers": active_layers,
        },
        "discrete_token_ids": best_discrete_ids,
        "discrete_text": discrete_text,
        "best_discrete_cosine_similarity": best_discrete_cos_sim,
        "per_layer_discrete_cosine_similarities": final_cos_sims,
        "loss_curve": loss_curve,
        "metrics_history": metrics_history,
        "total_time_seconds": total_time,
    }
