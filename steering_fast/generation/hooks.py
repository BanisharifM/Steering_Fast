"""Steering hooks for transformer blocks.

Same algorithm as original generation_utils.py, but cleaned up with
type hints and proper hook management.
"""
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def hook_model(
    model,
    directions: Dict[int, torch.Tensor],
    layers_to_control: List[int],
    control_coef: float,
    start_from_token: int = 0,
) -> List:
    """Register forward hooks on transformer blocks for steering.

    Args:
        model: The language model
        directions: {layer_idx: Tensor(1, hidden_dim)} steering vectors
        layers_to_control: Which layers to steer
        control_coef: Scaling coefficient for the steering vector
        start_from_token: Only steer positions >= this index

    Returns:
        List of hook handles (call clear_hooks() to remove them)
    """
    hooks = []

    for layer_idx in layers_to_control:
        if layer_idx not in directions:
            continue

        direction = directions[layer_idx]
        if direction is None:
            continue

        # Get the transformer block
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            block = model.model.layers[layer_idx]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            block = model.transformer.h[layer_idx]
        else:
            log.warning("Cannot find transformer block %d", layer_idx)
            continue

        def make_hook(dir_vec, coef, start_tok):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                # Determine offset for start_from_token
                seq_len = hidden.shape[1]
                if start_tok > 0 and seq_len > start_tok:
                    mask = torch.zeros(1, seq_len, 1, device=hidden.device)
                    mask[:, start_tok:, :] = 1.0
                    control_vec = coef * dir_vec.to(hidden.device, hidden.dtype)
                    hidden = hidden + mask * control_vec
                else:
                    control_vec = coef * dir_vec.to(hidden.device, hidden.dtype)
                    hidden = hidden + control_vec

                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            return hook_fn

        handle = block.register_forward_hook(
            make_hook(direction, control_coef, start_from_token)
        )
        hooks.append(handle)

    return hooks


def clear_hooks(hooks: List) -> None:
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()


def generate_steered(
    model,
    tokenizer,
    prompt: str,
    directions: Dict[int, torch.Tensor],
    layers_to_control: List[int],
    control_coef: float,
    max_new_tokens: int = 50,
    start_from_token: int = 0,
    do_sample: bool = False,
) -> str:
    """Generate text with steering hooks applied.

    Returns the full generated text (including prompt).
    """
    # Register hooks
    hooks = hook_model(model, directions, layers_to_control, control_coef, start_from_token)

    try:
        # Format and tokenize
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        return tokenizer.decode(output_ids[0], skip_special_tokens=False)
    finally:
        clear_hooks(hooks)
