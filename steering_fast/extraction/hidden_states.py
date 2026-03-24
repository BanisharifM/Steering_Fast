"""Batched hidden state extraction for direction training (Stage 1).

Original: batch_size=1, extracts hidden states + attention in same pass.
Optimized: configurable batch_size, separates hidden states from attention,
GPU-resident tensors (no unnecessary CPU transfers).
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def extract_hidden_states_batched(
    prompts: List[str],
    model,
    tokenizer,
    hidden_layers: List[int],
    rep_token: Union[int, str],
    layer_to_token: Optional[Dict[int, int]],
    batch_size: int = 16,
    device: str = "cuda",
    need_attention: bool = False,
    head_agg: str = "mean",
    prefix_start: int = 0,
    prefix_end: int = 0,
) -> Tuple[Dict[int, torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
    """Extract hidden states (and optionally attention) for all prompts, batched.

    Args:
        prompts: List of formatted prompt strings
        model: Language model
        tokenizer: Tokenizer (padding_side='left')
        hidden_layers: List of layer indices to extract
        rep_token: Token position index (int like -1, -2) or 'max_attn_per_layer'
        layer_to_token: Mapping {layer_idx: token_idx} when rep_token='max_attn_per_layer'
        batch_size: Forward pass batch size
        device: Target device for output tensors
        need_attention: Whether to also extract attention-based soft labels
        head_agg: 'mean' or 'max' for attention aggregation
        prefix_start: Start index of prefix tokens (for attention)
        prefix_end: End index of prefix tokens (for attention)

    Returns:
        Tuple of:
        - hidden_states: {layer_idx: Tensor(N, hidden_dim)} on device
        - soft_labels: {layer_idx: Tensor(N, 1)} on device, or None if not need_attention
    """
    N = len(prompts)
    hidden_dim = model.config.hidden_size
    use_layer_tokens = rep_token == "max_attn_per_layer"

    # Pre-allocate on GPU
    all_hidden = {layer: torch.empty(N, hidden_dim, device=device) for layer in hidden_layers}
    all_soft = None
    if need_attention:
        all_soft = {layer: torch.empty(N, 1, device=device) for layer in hidden_layers}

    for batch_start in tqdm(range(0, N, batch_size), desc="Extracting hidden states"):
        batch_end = min(batch_start + batch_size, N)
        batch_prompts = prompts[batch_start:batch_end]
        B = len(batch_prompts)

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                output_attentions=need_attention,
                return_dict=True,
            )

        for layer_idx in hidden_layers:
            # Determine token position for this layer
            if use_layer_tokens and layer_to_token is not None:
                tok_idx = layer_to_token.get(layer_idx, -1)
            elif isinstance(rep_token, int):
                tok_idx = rep_token
            else:
                tok_idx = -1

            # Extract hidden states: (B, hidden_dim)
            hs = outputs.hidden_states[layer_idx + 1][:, tok_idx, :]
            all_hidden[layer_idx][batch_start:batch_end] = hs

            # Extract attention-based soft labels if needed
            if need_attention and all_soft is not None:
                attn = outputs.attentions[layer_idx]  # (B, n_heads, seq_len, seq_len)
                attn_to_prefix = attn[:, :, tok_idx, prefix_start:prefix_end].sum(dim=-1)  # (B, n_heads)
                if head_agg == "mean":
                    soft = attn_to_prefix.mean(dim=1)  # (B,)
                else:
                    soft = attn_to_prefix.amax(dim=1)
                all_soft[layer_idx][batch_start:batch_end, 0] = soft

        del outputs, encoded
        torch.cuda.empty_cache()

    return all_hidden, all_soft


def load_soft_labels_from_npy(
    concept: str,
    model_name: str,
    n_layers: int,
    layer_to_token: Dict[int, int],
    attn_dir: str,
    n_pos_samples: int,
    n_neg_samples: int,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """Load pre-computed soft labels from Stage 0's .npy attention files.

    This avoids needing output_attentions=True in Stage 1, enabling Flash Attention.

    Args:
        concept: Concept name
        model_name: Model short name
        n_layers: Total number of layers
        layer_to_token: {layer_idx: token_idx} mapping
        attn_dir: Directory containing .npy attention files
        n_pos_samples: Number of positive samples
        n_neg_samples: Number of negative samples
        device: Target device

    Returns:
        {layer_idx: Tensor(N_total, 1)} where N_total = n_pos + n_neg,
        positive samples get their attention score, negative get 0.
    """
    import os
    attn_path = os.path.join(
        attn_dir,
        f"attentions_meanhead_{model_name}_{concept}_paired_statements.npy",
    )

    if not os.path.exists(attn_path):
        log.warning("Attention file not found: %s", attn_path)
        return {}

    attns = np.load(attn_path)  # (N_pos_paired, n_layers, n_common_toks)
    # Take every other sample (same as original stage 0 which does pairs[::2])
    n_attn_samples = attns.shape[0]

    soft_labels = {}
    n_total = n_pos_samples + n_neg_samples

    for layer_idx in range(1, n_layers):
        tok_idx = layer_to_token.get(layer_idx, -1)
        # Get attention for this layer at the selected token
        layer_attns = attns[:, layer_idx, tok_idx]  # (N_pos_paired,)

        # Build interleaved labels: pos gets attention score, neg gets 0
        labels = torch.zeros(n_total, 1, device=device)
        # Positive samples are at even indices (0, 2, 4, ...)
        n_use = min(n_pos_samples, len(layer_attns))
        labels[:n_use * 2:2, 0] = torch.tensor(layer_attns[:n_use], device=device)

        soft_labels[layer_idx] = labels

    return soft_labels
