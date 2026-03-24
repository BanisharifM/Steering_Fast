"""Stage 0: Batched attention-to-prefix extraction.

Original: batch_size=1, Python loop over 32 heads per layer.
Optimized: configurable batch_size, fully vectorized attention aggregation.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_prefix_indices(prompt: str, tokenizer) -> Tuple[int, int]:
    """Find the token range of the concept prefix in a formatted prompt.

    Returns (prefix_start, prefix_end) as token indices.
    The prefix is everything between the user header and 'What are your thoughts'.
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]

    # Find "What" token that starts the shared statement portion
    what_idx = None
    for i, tok_str in enumerate(decoded):
        if "What" in tok_str or " What" in tok_str:
            what_idx = i
            break

    if what_idx is None:
        # Fallback: prefix is first 30% of user content
        what_idx = len(tokens) // 3

    # Find user content start (after system/user headers)
    user_start = 0
    for i, tok_str in enumerate(decoded):
        if "user" in tok_str.lower() and i > 5:
            user_start = i + 2  # skip past header tokens
            break

    return user_start, what_idx


def extract_attention_batched(
    pos_prompts: List[str],
    model,
    tokenizer,
    n_common_toks: int,
    head_agg: str = "mean",
    batch_size: int = 16,
) -> Dict[int, np.ndarray]:
    """Extract attention-to-prefix for positive prompts, batched.

    Args:
        pos_prompts: List of formatted positive prompts
        model: The language model
        tokenizer: The tokenizer (padding_side='left')
        n_common_toks: Number of tokens added by chat template at end
        head_agg: 'mean' or 'max' over attention heads
        batch_size: Batch size for forward passes

    Returns:
        Dict mapping layer_idx to numpy array of shape (N, n_common_toks)
        where each entry is the summed attention from that token to prefix tokens.
    """
    n_layers = model.config.num_hidden_layers
    N = len(pos_prompts)

    # Pre-compute prefix indices from first prompt
    prefix_start, prefix_end = get_prefix_indices(pos_prompts[0], tokenizer)
    log.info("Prefix token range: [%d, %d)", prefix_start, prefix_end)

    # Pre-allocate output arrays
    layer_attns = {layer: np.zeros((N, n_common_toks), dtype=np.float32) for layer in range(n_layers)}

    for batch_start in tqdm(range(0, N, batch_size), desc="Extracting attention"):
        batch_end = min(batch_start + batch_size, N)
        batch_prompts = pos_prompts[batch_start:batch_end]
        B = len(batch_prompts)

        # Tokenize with left-padding
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True, return_dict=True)

        # For each layer, extract attention to prefix (vectorized over heads)
        for layer_idx in range(n_layers):
            attn = outputs.attentions[layer_idx]  # (B, n_heads, seq_len, seq_len)

            # Attention from last N common tokens to prefix range
            # Shape: (B, n_heads, n_common_toks, prefix_len)
            attn_slice = attn[:, :, -n_common_toks:, prefix_start:prefix_end]

            # Sum over prefix tokens: (B, n_heads, n_common_toks)
            attn_to_prefix = attn_slice.sum(dim=-1)

            # Aggregate over heads: (B, n_common_toks)
            if head_agg == "mean":
                agg = attn_to_prefix.mean(dim=1)
            else:  # max
                agg = attn_to_prefix.amax(dim=1)

            layer_attns[layer_idx][batch_start:batch_end] = agg.cpu().numpy()

        # Free GPU memory
        del outputs, encoded
        torch.cuda.empty_cache()

    return layer_attns


def compute_token_indices_per_layer(
    attn_array: np.ndarray,
) -> Dict[int, int]:
    """Compute the max-attention token index per layer from saved attention arrays.

    Args:
        attn_array: Shape (N_samples, n_layers, n_common_toks) from .npy file

    Returns:
        Dict mapping layer_idx to negative token index (e.g., -1, -2, etc.)
    """
    # Max over samples: (n_layers, n_common_toks)
    magnitudes = attn_array.max(axis=0)
    n_tokens = magnitudes.shape[-1]
    # Argmax per layer, convert to negative indexing
    max_token_idxs = np.argmax(magnitudes, axis=1) - n_tokens
    return {int(k): int(v) for k, v in enumerate(max_token_idxs)}
