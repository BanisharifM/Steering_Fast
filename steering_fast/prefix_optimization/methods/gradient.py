"""Method 1: Gradient-based continuous optimization of prefix embeddings.

Optimizes continuous prefix embeddings via backpropagation through the frozen model.
This is the primary baseline method.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..losses import (
    LOSS_FUNCTIONS,
    embedding_proximity_regularization,
    norm_regularization,
)

logger = logging.getLogger(__name__)


def extract_activations(
    model,
    prefix_embeds: torch.Tensor,
    statement_embeds: torch.Tensor,
    layers: List[int],
    target_position: str,
    prefix_length: int,
) -> Dict[int, torch.Tensor]:
    """Forward pass through the model and extract hidden states at target layers.

    Args:
        model: The frozen LLM
        prefix_embeds: (1, K, d_model) prefix embeddings (differentiable)
        statement_embeds: (1, T, d_model) statement token embeddings (frozen)
        layers: list of layer indices to extract
        target_position: where to read activations from
        prefix_length: K

    Returns:
        Dict mapping layer_idx -> activation tensor (d_model,)
    """
    # Concatenate prefix + statement embeddings
    input_embeds = torch.cat([prefix_embeds, statement_embeds], dim=1)
    seq_len = input_embeds.shape[1]

    # Attention mask: all ones (no padding)
    attention_mask = torch.ones(1, seq_len, device=input_embeds.device, dtype=torch.long)

    # Forward pass with hidden states
    with torch.set_grad_enabled(prefix_embeds.requires_grad):
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Extract activations at target position for each layer
    # outputs.hidden_states has L+1 entries: [embedding_output, layer_1, ..., layer_L]
    activations = {}
    for layer_idx in layers:
        # hidden_states index is layer_idx + 1 (index 0 = embedding output)
        # But check: some models index from 0. Let's be safe.
        hs_idx = layer_idx  # hidden_states[0] = embeddings, [1] = layer 1, etc.
        if hs_idx >= len(outputs.hidden_states):
            logger.warning(
                "Layer %d out of range (model has %d hidden states). Skipping.",
                layer_idx,
                len(outputs.hidden_states) - 1,
            )
            continue

        h = outputs.hidden_states[hs_idx]  # (1, seq_len, d_model)

        if target_position == "last_prefix":
            # Activation at the last prefix token position
            act = h[0, prefix_length - 1, :]
        elif target_position == "last_token":
            # Activation at the last token of the full sequence
            act = h[0, -1, :]
        elif target_position == "mean_prefix":
            # Mean-pool over all prefix positions
            act = h[0, :prefix_length, :].mean(dim=0)
        elif target_position == "mean_all":
            # Mean-pool over all positions
            act = h[0, :, :].mean(dim=0)
        else:
            raise ValueError(f"Unknown target_position: {target_position}")

        activations[layer_idx] = act

    return activations


def optimize_prefix(
    model,
    tokenizer,
    directions: Dict[int, torch.Tensor],
    concept: str,
    statements: List[str],
    config,
) -> Dict:
    """Run gradient-based prefix optimization.

    Args:
        model: Frozen LLM (all params requires_grad=False)
        tokenizer: The model's tokenizer
        directions: Dict[layer_idx -> direction_vector (1, d) or (d,)]
        concept: Concept name (for initialization)
        statements: List of general statement strings
        config: PrefixOptConfig

    Returns:
        Dict with results: optimized_embeds, discrete_tokens, metrics, loss_curve, etc.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embedding_matrix = model.model.embed_tokens.weight.detach()

    # Normalize directions to unit vectors
    clean_directions = {}
    for layer_idx, v in directions.items():
        v_flat = v.flatten().float().to(device)
        v_flat = v_flat / v_flat.norm()
        clean_directions[layer_idx] = v_flat

    # Filter to requested layers
    layers = config.get_layers()
    active_directions = {l: clean_directions[l] for l in layers if l in clean_directions}
    if not active_directions:
        raise ValueError(
            f"No directions found for requested layers {layers}. "
            f"Available: {sorted(clean_directions.keys())}"
        )
    active_layers = sorted(active_directions.keys())
    logger.info("Optimizing for %d layers: %s", len(active_layers), active_layers)

    # Initialize prefix embeddings
    from ..initialization import (
        init_concept_name,
        init_logit_lens,
        init_random,
    )

    if config.init_strategy == "concept_name":
        prefix_embeds = init_concept_name(
            embedding_matrix, tokenizer, concept, config.prefix_length, config.seed
        )
    elif config.init_strategy == "logit_lens":
        # Use the direction from the middle active layer for logit lens init
        mid_layer = active_layers[len(active_layers) // 2]
        lm_head_weight = model.lm_head.weight.detach()
        lm_head_bias = getattr(model.lm_head, "bias", None)
        if lm_head_bias is not None:
            lm_head_bias = lm_head_bias.detach()
        prefix_embeds = init_logit_lens(
            embedding_matrix,
            lm_head_weight,
            lm_head_bias,
            active_directions[mid_layer],
            config.prefix_length,
        )
    elif config.init_strategy == "random":
        prefix_embeds = init_random(embedding_matrix, config.prefix_length, config.seed)
    else:
        prefix_embeds = init_random(embedding_matrix, config.prefix_length, config.seed)

    # Make prefix optimizable
    prefix_embeds = prefix_embeds.to(device).requires_grad_(True)

    # Pre-compute statement embeddings (frozen)
    statement_embeds_list = []
    for stmt in statements[:config.n_statements]:
        token_ids = tokenizer.encode(stmt, add_special_tokens=False, return_tensors="pt")
        token_ids = token_ids.to(device)
        with torch.no_grad():
            stmt_emb = model.model.embed_tokens(token_ids).float()  # (1, T, d)
        statement_embeds_list.append(stmt_emb)

    # Setup optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW([prefix_embeds], lr=config.lr)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam([prefix_embeds], lr=config.lr)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD([prefix_embeds], lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    loss_fn = LOSS_FUNCTIONS[config.loss_type]

    # Optimization loop
    loss_curve = []
    metrics_history = []
    best_cos_sim = -1.0
    best_embeds = None
    start_time = time.time()

    for step in range(config.n_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        step_cos_sims = []

        # Average over statements
        for stmt_emb in statement_embeds_list:
            prefix_batch = prefix_embeds.unsqueeze(0)  # (1, K, d)

            activations = extract_activations(
                model,
                prefix_batch,
                stmt_emb,
                active_layers,
                config.target_position,
                config.prefix_length,
            )

            # Per-layer alignment loss
            for layer_idx in active_layers:
                if layer_idx not in activations:
                    continue
                act = activations[layer_idx]
                target = active_directions[layer_idx]

                loss_val, metrics = loss_fn(act, target)
                total_loss = total_loss + loss_val / (len(active_layers) * len(statement_embeds_list))
                step_cos_sims.append(metrics["cosine_similarity"])

        # Regularization
        if config.lambda_prox > 0:
            reg_prox = embedding_proximity_regularization(prefix_embeds, embedding_matrix)
            total_loss = total_loss + config.lambda_prox * reg_prox

        if config.lambda_norm > 0:
            reg_norm = norm_regularization(prefix_embeds, embedding_matrix)
            total_loss = total_loss + config.lambda_norm * reg_norm

        # Backpropagate
        total_loss.backward()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([prefix_embeds], config.grad_clip)

        optimizer.step()

        # Track metrics
        mean_cos = sum(step_cos_sims) / len(step_cos_sims) if step_cos_sims else 0.0
        loss_curve.append(total_loss.item())

        if mean_cos > best_cos_sim:
            best_cos_sim = mean_cos
            best_embeds = prefix_embeds.clone().detach()

        if step % config.log_every == 0 or step == config.n_steps - 1:
            # Find nearest tokens for current embeddings
            with torch.no_grad():
                dists = torch.cdist(prefix_embeds.float(), embedding_matrix.float())
                nearest_ids = dists.argmin(dim=1).tolist()
                nearest_tokens = tokenizer.convert_ids_to_tokens(nearest_ids)

            elapsed = time.time() - start_time
            step_metrics = {
                "step": step,
                "loss": total_loss.item(),
                "mean_cosine_similarity": mean_cos,
                "best_cosine_similarity": best_cos_sim,
                "nearest_tokens": nearest_tokens,
                "elapsed_seconds": elapsed,
            }
            metrics_history.append(step_metrics)
            logger.info(
                "Step %d/%d | loss=%.4f | cos_sim=%.4f (best=%.4f) | tokens=%s",
                step,
                config.n_steps,
                total_loss.item(),
                mean_cos,
                best_cos_sim,
                " ".join(nearest_tokens[:5]),
            )

    # Final discrete recovery
    with torch.no_grad():
        final_embeds = best_embeds if best_embeds is not None else prefix_embeds.detach()
        dists = torch.cdist(final_embeds.float(), embedding_matrix.float())
        nearest_ids = dists.argmin(dim=1).tolist()
        top5_per_position = []
        for k in range(config.prefix_length):
            topk = dists[k].topk(5, largest=False)
            top5_ids = topk.indices.tolist()
            top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids)
            top5_dists = topk.values.tolist()
            top5_per_position.append(
                {"token_ids": top5_ids, "tokens": top5_tokens, "distances": top5_dists}
            )

    discrete_text = tokenizer.decode(nearest_ids, skip_special_tokens=True)

    # Evaluate discrete prefix alignment
    discrete_embeds = embedding_matrix[nearest_ids].clone().detach().float().unsqueeze(0)
    discrete_cos_sims = {}
    with torch.no_grad():
        for stmt_emb in statement_embeds_list[:1]:  # evaluate on first statement
            discrete_acts = extract_activations(
                model,
                discrete_embeds,
                stmt_emb,
                active_layers,
                config.target_position,
                config.prefix_length,
            )
            for layer_idx in active_layers:
                if layer_idx in discrete_acts:
                    cs = F.cosine_similarity(
                        discrete_acts[layer_idx].unsqueeze(0),
                        active_directions[layer_idx].unsqueeze(0),
                    ).item()
                    discrete_cos_sims[layer_idx] = cs

    total_time = time.time() - start_time
    logger.info("Optimization completed in %.1f seconds", total_time)
    logger.info("Best continuous cos_sim: %.4f", best_cos_sim)
    logger.info("Discrete prefix: '%s'", discrete_text)
    logger.info(
        "Discrete cos_sims: %s",
        {l: f"{v:.4f}" for l, v in discrete_cos_sims.items()},
    )

    return {
        "method": "gradient",
        "concept": concept,
        "config": {
            "loss_type": config.loss_type,
            "init_strategy": config.init_strategy,
            "lr": config.lr,
            "n_steps": config.n_steps,
            "prefix_length": config.prefix_length,
            "target_position": config.target_position,
            "layers": active_layers,
            "optimizer": config.optimizer,
            "lambda_prox": config.lambda_prox,
            "lambda_norm": config.lambda_norm,
        },
        "continuous_embeds": final_embeds.cpu(),
        "discrete_token_ids": nearest_ids,
        "discrete_text": discrete_text,
        "top5_per_position": top5_per_position,
        "best_cosine_similarity": best_cos_sim,
        "discrete_cosine_similarities": discrete_cos_sims,
        "loss_curve": loss_curve,
        "metrics_history": metrics_history,
        "total_time_seconds": total_time,
    }
