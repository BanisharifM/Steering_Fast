"""Model loading with configurable attention implementation."""
import logging
from collections import namedtuple
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

LLM = namedtuple("LLM", ["language_model", "tokenizer", "model_name", "n_added_tokens"])


def get_n_common_toks(tokenizer, verbose: bool = False) -> int:
    """Count tokens added at the end by the chat template (generation prompt)."""
    random_word = "This is a random sentence"
    chat = [{"role": "user", "content": random_word}]

    ids_no_gen = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )[0]
    ids_with_gen = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )[0]

    added_ids = ids_with_gen[len(ids_no_gen) - 1 :]
    n = len(added_ids)
    if verbose:
        log.info("Tokens added at end: %s", tokenizer.convert_ids_to_tokens(added_ids.tolist()))
    return n


def load_model(
    model_name: str,
    hf_id: str,
    cache_dir: Optional[str] = None,
    attn_implementation: str = "eager",
) -> LLM:
    """Load model and tokenizer. Returns LLM namedtuple.

    Args:
        model_name: Short name (e.g., 'llama_3.1_8b')
        hf_id: HuggingFace model ID
        cache_dir: Directory for model downloads (None = HF default)
        attn_implementation: 'eager' (returns attention weights) or 'flash_attention_2'
    """
    log.info("Loading model %s (%s) with attn=%s", model_name, hf_id, attn_implementation)

    language_model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map="cuda",
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
    ).eval()

    archs = getattr(language_model.config, "architectures", []) or []
    use_fast_tokenizer = all("LlamaForCausalLM" not in a for a in archs)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        legacy=False,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    n = get_n_common_toks(tokenizer, verbose=True)
    log.info("Model loaded: %d layers, %d hidden dim", language_model.config.num_hidden_layers, language_model.config.hidden_size)

    return LLM(language_model, tokenizer, model_name, n)
