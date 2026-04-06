"""PEZ v2: Corrected prefix optimization matching original pipeline structure.

Key fixes over v1:
1. Full prompt wrapped in chat template (same as original pipeline)
2. Token position from max_attn_per_layer (layer_to_token), not last_prefix
3. Proper input construction: [chat_header_ids | OPTIMIZABLE_PREFIX | statement_ids | chat_footer_ids]
4. STE on the prefix portion only; rest uses real token IDs
"""

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..losses import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)


def load_layer_to_token(data_dir: str, concept: str, model_name: str, head_agg: str = "mean") -> Dict[int, int]:
    """Load per-layer token indices from pre-computed attention files.

    These are the token positions with maximum attention to the prefix,
    exactly as used by the original pipeline in stage 1.

    Returns:
        Dict[layer_idx -> negative_token_index] e.g. {0: -3, 1: -2, ...}
    """
    attn_dir = os.path.join(data_dir, "attention_to_prompt")
    outpath = os.path.join(attn_dir, f"attentions_{head_agg}head_{model_name}_{concept}_paired_statements.npy")

    if not os.path.exists(outpath):
        logger.warning("Attention file not found: %s. Falling back to last token (-1).", outpath)
        return None

    magnitudes = np.load(outpath).max(axis=0)  # (400, n_layers, n_tokens) -> (n_layers, n_tokens)
    n_tokens = magnitudes.shape[-1]
    max_token_idxs = np.argmax(magnitudes, axis=1) - n_tokens  # negative indexing

    layer_to_token = {int(k): int(v) for k, v in enumerate(max_token_idxs)}
    logger.info("Loaded layer_to_token: %d layers, e.g. layer 0->%d, layer 15->%d",
                len(layer_to_token),
                layer_to_token.get(0, "?"),
                layer_to_token.get(15, "?"))
    return layer_to_token


def build_prompt_parts(concept: str, concept_class: str, statement: str, tokenizer):
    """Build the full prompt matching original pipeline structure.

    The original pipeline uses:
        Positive: "Personify someone who is terrified of {fear}.  What are your thoughts on...\\nStatement: {statement}"
        Negative: "What are your thoughts on...\\nStatement: {statement}"

    Both wrapped in chat template with add_generation_prompt=True.

    Returns:
        prefix_text: the concept prefix string (what we optimize)
        suffix_text: the statement portion
        full_positive_text: the complete chat-templated positive prompt
        full_negative_text: the complete chat-templated negative prompt
    """
    TEMPLATES = {
        "fears": ("Personify someone who is terrified of {concept}.  ", "What are your thoughts on the following statement? \nStatement: {statement}"),
        "moods": ("Act as someone who is feeling {concept}.  ", "What are your thoughts on the following statement? \nStatement: {statement}"),
        "personalities": ("Act as someone who is {concept}.  ", "What are your thoughts on the following statement? \nStatement: {statement}"),
        "personas": ("You are {concept}.  ", "What are your thoughts on the following statement? \nStatement: {statement}"),
        "places": ("Describe what it is like to be in {concept}.  ", "What are your thoughts on the following statement? \nStatement: {statement}"),
    }

    prefix_template, suffix_template = TEMPLATES.get(concept_class, ("Think about {concept}.  ", "{statement}"))
    prefix_text = prefix_template.format(concept=concept)
    suffix_text = suffix_template.format(statement=statement)

    # Full prompts wrapped in chat template
    pos_content = prefix_text + suffix_text
    neg_content = suffix_text

    pos_chat = [{"role": "user", "content": pos_content}]
    neg_chat = [{"role": "user", "content": neg_content}]

    full_positive = tokenizer.apply_chat_template(pos_chat, tokenize=False, add_generation_prompt=True)
    full_negative = tokenizer.apply_chat_template(neg_chat, tokenize=False, add_generation_prompt=True)

    return prefix_text, suffix_text, full_positive, full_negative


def tokenize_prompt_parts(full_positive: str, prefix_text: str, tokenizer, device):
    """Tokenize the full positive prompt and identify which tokens are the prefix.

    Returns:
        all_ids: (seq_len,) full token IDs
        prefix_start: index where the concept prefix starts
        prefix_end: index where the concept prefix ends (exclusive)
    """
    all_ids = tokenizer.encode(full_positive, add_special_tokens=False, return_tensors="pt")[0].to(device)

    # Find prefix boundaries using the same method as the original pipeline:
    # prefix_start = number of chat template prepend tokens
    # prefix_end = position of " What" token
    chat_empty = [{"role": "user", "content": ""}]
    empty_ids = tokenizer.apply_chat_template(chat_empty, tokenize=True, add_generation_prompt=False, return_tensors="pt")
    if isinstance(empty_ids, torch.Tensor) and empty_ids.ndim == 2:
        empty_ids = empty_ids[0]

    # Find longest common prefix (chat template header)
    L = min(len(all_ids), len(empty_ids))
    prefix_start = 0
    while prefix_start < L and all_ids[prefix_start].item() == empty_ids[prefix_start].item():
        prefix_start += 1

    # Find " What" token to determine prefix end
    what_token_id = tokenizer.encode(" What", add_special_tokens=False)[0]
    what_positions = (all_ids == what_token_id).nonzero(as_tuple=True)[0]
    if len(what_positions) > 0:
        prefix_end = what_positions[0].item()
    else:
        # Fallback: estimate from prefix text length
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_end = prefix_start + len(prefix_ids)

    logger.info("Prompt tokenized: %d tokens, prefix [%d:%d] = '%s'",
                len(all_ids), prefix_start, prefix_end,
                tokenizer.decode(all_ids[prefix_start:prefix_end]))

    return all_ids, prefix_start, prefix_end


def forward_with_prefix_embeds(
    model, all_ids, prefix_start, prefix_end, prefix_embeds, device
):
    """Forward pass replacing the prefix tokens with optimizable embeddings.

    Constructs inputs_embeds by:
    1. Embedding all token IDs normally
    2. Replacing positions [prefix_start:prefix_end] with prefix_embeds

    Args:
        model: Frozen LLM
        all_ids: (seq_len,) full token IDs
        prefix_start, prefix_end: prefix boundary indices
        prefix_embeds: (prefix_len, d_model) optimizable embeddings
        device: cuda device

    Returns:
        outputs with hidden_states
    """
    with torch.no_grad():
        all_embeds = model.model.embed_tokens(all_ids.unsqueeze(0)).float()  # (1, seq_len, d)

    # Replace prefix region with optimizable embeddings
    combined = all_embeds.clone()
    prefix_len = prefix_end - prefix_start
    actual_len = min(prefix_len, prefix_embeds.shape[0])
    combined[0, prefix_start:prefix_start + actual_len, :] = prefix_embeds[:actual_len].float()

    attention_mask = torch.ones(1, combined.shape[1], device=device, dtype=torch.long)

    outputs = model(
        inputs_embeds=combined,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs


def optimize_prefix_pez_v2(
    model,
    tokenizer,
    directions: Dict[int, torch.Tensor],
    concept: str,
    statements: List[str],
    config,
) -> Dict:
    """PEZ v2: Proper prefix optimization matching original pipeline structure.

    Key improvements:
    - Full chat-templated prompt
    - Per-layer token position from max_attn_per_layer
    - STE on prefix portion only
    - Measures alignment at the exact position the original pipeline uses
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
    logger.info("PEZ v2: optimizing for %d layers: %s", len(active_layers), active_layers)

    # Build prompt from first statement
    stmt = statements[0].strip()
    prefix_text, suffix_text, full_positive, full_negative = build_prompt_parts(
        concept, config.concept_class, stmt, tokenizer
    )

    # Tokenize and find prefix boundaries
    all_ids, prefix_start, prefix_end = tokenize_prompt_parts(
        full_positive, prefix_text, tokenizer, device
    )
    prefix_len = prefix_end - prefix_start
    seq_len = len(all_ids)

    logger.info("Full prompt: %d tokens. Prefix: %d tokens [%d:%d].",
                seq_len, prefix_len, prefix_start, prefix_end)

    # Initialize prefix embeddings from the actual prefix tokens
    with torch.no_grad():
        init_embeds = model.model.embed_tokens(all_ids[prefix_start:prefix_end]).float()
    prefix_embeds = init_embeds.clone().detach().requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.Adam([prefix_embeds], lr=config.lr)
    loss_fn = LOSS_FUNCTIONS[config.loss_type]

    # Tracking
    loss_curve = []
    metrics_history = []
    best_discrete_cos_sim = -1.0
    best_discrete_ids = None
    start_time = time.time()

    for step in range(config.n_steps):
        optimizer.zero_grad()

        # STE: project to nearest tokens
        with torch.no_grad():
            dists = torch.cdist(prefix_embeds.float(), embedding_matrix.float())
            token_ids_step = dists.argmin(dim=1)
            projected = embedding_matrix[token_ids_step].clone().float()

        # Straight-Through Estimator
        ste_embeds = prefix_embeds + (projected - prefix_embeds).detach()

        # Forward pass with STE prefix
        outputs = forward_with_prefix_embeds(
            model, all_ids, prefix_start, prefix_end, ste_embeds, device
        )

        # Compute alignment loss at per-layer token positions
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        step_cos_sims = []

        # hidden_states[0] = embedding output, [1] = layer 0, [2] = layer 1, ...
        for layer_idx in active_layers:
            hs_idx = layer_idx + 1  # +1 because index 0 is embedding output
            if hs_idx >= len(outputs.hidden_states):
                continue

            h = outputs.hidden_states[hs_idx]  # (1, seq_len, d_model)

            # Use per-layer token position (negative index) or fallback to -1
            if layer_to_token is not None and layer_idx in layer_to_token:
                tok_pos = layer_to_token[layer_idx]  # negative index, e.g. -3
            else:
                tok_pos = -1

            act = h[0, tok_pos, :].float()  # (d_model,)
            target = active_directions[layer_idx]

            loss_val, metrics = loss_fn(act, target)
            total_loss = total_loss + loss_val / len(active_layers)
            step_cos_sims.append(metrics["cosine_similarity"])

        total_loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([prefix_embeds], config.grad_clip)

        optimizer.step()

        # Track
        mean_cos = sum(step_cos_sims) / len(step_cos_sims) if step_cos_sims else 0.0
        loss_curve.append(total_loss.item())

        if mean_cos > best_discrete_cos_sim:
            best_discrete_cos_sim = mean_cos
            best_discrete_ids = token_ids_step.tolist()

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
                "PEZv2 Step %d/%d | loss=%.4f | cos=%.4f (best=%.4f) | %s",
                step, config.n_steps, total_loss.item(), mean_cos,
                best_discrete_cos_sim,
                tokenizer.decode(token_ids_step.tolist(), skip_special_tokens=False)[:60],
            )

    # Final evaluation of best discrete prefix
    discrete_text = tokenizer.decode(best_discrete_ids, skip_special_tokens=True)
    final_cos_sims = {}

    with torch.no_grad():
        best_embeds = embedding_matrix[best_discrete_ids].float()
        outputs = forward_with_prefix_embeds(
            model, all_ids, prefix_start, prefix_end, best_embeds, device
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

    # Also evaluate the ORIGINAL hand-crafted prefix at the correct positions
    handcrafted_cos_sims = {}
    with torch.no_grad():
        orig_embeds = model.model.embed_tokens(all_ids[prefix_start:prefix_end]).float()
        outputs = forward_with_prefix_embeds(
            model, all_ids, prefix_start, prefix_end, orig_embeds, device
        )
        for layer_idx in active_layers:
            hs_idx = layer_idx + 1
            if hs_idx >= len(outputs.hidden_states):
                continue
            h = outputs.hidden_states[hs_idx]
            tok_pos = layer_to_token[layer_idx] if layer_to_token and layer_idx in layer_to_token else -1
            act = h[0, tok_pos, :].float()
            cs = F.cosine_similarity(act.unsqueeze(0), active_directions[layer_idx].unsqueeze(0)).item()
            handcrafted_cos_sims[layer_idx] = cs

    total_time = time.time() - start_time

    hc_mean = sum(handcrafted_cos_sims.values()) / len(handcrafted_cos_sims) if handcrafted_cos_sims else 0
    opt_mean = sum(final_cos_sims.values()) / len(final_cos_sims) if final_cos_sims else 0

    logger.info("=" * 60)
    logger.info("PEZ v2 RESULTS:")
    logger.info("  Hand-crafted prefix mean cos_sim: %.4f", hc_mean)
    logger.info("  Optimized prefix mean cos_sim:    %.4f", opt_mean)
    logger.info("  Improvement:                      %.4f", opt_mean - hc_mean)
    logger.info("  Original prefix: '%s'", prefix_text)
    logger.info("  Optimized prefix: '%s'", discrete_text)
    logger.info("  Time: %.1f seconds", total_time)
    logger.info("=" * 60)

    return {
        "method": "pez_v2",
        "concept": concept,
        "original_prefix": prefix_text,
        "optimized_prefix": discrete_text,
        "optimized_token_ids": best_discrete_ids,
        "handcrafted_per_layer_cos_sims": handcrafted_cos_sims,
        "handcrafted_mean_cos_sim": hc_mean,
        "optimized_per_layer_cos_sims": final_cos_sims,
        "optimized_mean_cos_sim": opt_mean,
        "improvement": opt_mean - hc_mean,
        "best_during_training_cos_sim": best_discrete_cos_sim,
        "loss_curve": loss_curve,
        "metrics_history": metrics_history,
        "total_time_seconds": total_time,
        "layer_to_token": layer_to_token,
        "prefix_boundaries": (prefix_start, prefix_end),
        "config": {
            "loss_type": config.loss_type,
            "lr": config.lr,
            "n_steps": config.n_steps,
            "prefix_length": prefix_len,
            "layers": active_layers,
        },
    }
