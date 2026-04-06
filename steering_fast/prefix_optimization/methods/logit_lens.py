"""Method 3: Logit lens / unembedding inversion.

Project concept directions through the model's language model head to find
tokens the model associates with each direction. No optimization required;
purely analytical.
"""

import logging
from typing import Dict, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def logit_lens_tokens(
    model,
    direction: torch.Tensor,
    tokenizer,
    top_k: int = 20,
) -> Dict:
    """Project a concept direction through the unembedding matrix.

    logits(v) = W_u @ v + b

    The top-K tokens represent what the model's output head "thinks"
    direction v means in token space.

    Args:
        model: The LLM (needs lm_head)
        direction: (d_model,) concept direction vector
        tokenizer: For decoding token IDs
        top_k: Number of top tokens to return

    Returns:
        Dict with top tokens, logits, and decoded strings
    """
    device = next(model.parameters()).device
    v = direction.flatten().float().to(device)

    # lm_head: (vocab_size, d_model) weight + optional bias
    lm_head_weight = model.lm_head.weight.detach().float()
    lm_head_bias = getattr(model.lm_head, "bias", None)

    logits = lm_head_weight @ v  # (vocab_size,)
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.detach().float()

    # Top-K and bottom-K tokens
    topk = torch.topk(logits, top_k)
    bottomk = torch.topk(logits, top_k, largest=False)

    top_tokens = tokenizer.convert_ids_to_tokens(topk.indices.tolist())
    bottom_tokens = tokenizer.convert_ids_to_tokens(bottomk.indices.tolist())

    return {
        "top_token_ids": topk.indices.tolist(),
        "top_tokens": top_tokens,
        "top_logits": topk.values.tolist(),
        "bottom_token_ids": bottomk.indices.tolist(),
        "bottom_tokens": bottom_tokens,
        "bottom_logits": bottomk.values.tolist(),
    }


def run_logit_lens_analysis(
    model,
    tokenizer,
    directions: Dict[int, torch.Tensor],
    concept: str,
    statements: List[str],
    config,
) -> Dict:
    """Run logit lens analysis for all layers.

    For each layer's concept direction, find the tokens most associated
    with that direction via the unembedding matrix. Then evaluate
    the alignment of those tokens when used as a prefix.

    Args:
        model: Frozen LLM
        tokenizer: Tokenizer
        directions: Dict[layer_idx -> direction vector]
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

    results = {
        "method": "logit_lens",
        "concept": concept,
        "per_layer_analysis": {},
        "best_prefix_alignment": {},
    }

    # Logit lens for each layer
    for layer_idx in active_layers:
        ll_result = logit_lens_tokens(
            model, active_directions[layer_idx], tokenizer, top_k=20
        )
        results["per_layer_analysis"][layer_idx] = ll_result
        logger.info(
            "Layer %d logit lens top-5: %s",
            layer_idx,
            ll_result["top_tokens"][:5],
        )

    # Use the mid-layer logit lens tokens as a prefix and evaluate alignment
    mid_layer = active_layers[len(active_layers) // 2]
    mid_ll = results["per_layer_analysis"][mid_layer]
    prefix_ids = mid_ll["top_token_ids"][:config.prefix_length]

    prefix_embeds = embedding_matrix[prefix_ids].clone().detach().float().unsqueeze(0)
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

    logger.info("Logit-lens prefix (layer %d): '%s'", mid_layer, prefix_text)

    # Evaluate this prefix's alignment across all layers
    stmt = statements[0]
    token_ids = tokenizer.encode(stmt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        stmt_emb = model.model.embed_tokens(token_ids).float()

        input_embeds = torch.cat([prefix_embeds.to(device), stmt_emb], dim=1)
        seq_len = input_embeds.shape[1]
        mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )

        for layer_idx in active_layers:
            h = outputs.hidden_states[layer_idx]
            if config.target_position == "last_prefix":
                act = h[0, config.prefix_length - 1, :].float()
            elif config.target_position == "last_token":
                act = h[0, -1, :].float()
            else:
                act = h[0, config.prefix_length - 1, :].float()

            cos_sim = F.cosine_similarity(
                act.unsqueeze(0), active_directions[layer_idx].unsqueeze(0)
            ).item()
            results["best_prefix_alignment"][layer_idx] = cos_sim

    results["prefix_text"] = prefix_text
    results["prefix_token_ids"] = prefix_ids
    logger.info("Logit-lens prefix alignment: %s",
                {l: f"{v:.4f}" for l, v in results["best_prefix_alignment"].items()})

    return results
