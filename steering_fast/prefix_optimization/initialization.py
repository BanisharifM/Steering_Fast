"""Initialization strategies for prefix embeddings.

Each function returns a (K, d_model) tensor of initial prefix embeddings.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def init_random(
    embedding_matrix: torch.Tensor,
    prefix_length: int,
    seed: int = 42,
) -> torch.Tensor:
    """Sample K random token embeddings from the vocabulary.

    This is the simplest baseline. Each prefix position gets a random
    real token's embedding.
    """
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, embedding_matrix.shape[0], (prefix_length,), generator=gen)
    embeds = embedding_matrix[indices].clone().detach().float()
    logger.info(
        "Random init: sampled %d token embeddings (indices: %s)",
        prefix_length,
        indices[:5].tolist(),
    )
    return embeds


def init_concept_name(
    embedding_matrix: torch.Tensor,
    tokenizer,
    concept: str,
    prefix_length: int,
    seed: int = 42,
) -> torch.Tensor:
    """Initialize from the concept name's token embeddings.

    If the concept name has fewer tokens than prefix_length, pad with
    random vocabulary embeddings. If more, truncate.
    """
    # Tokenize the concept name (without special tokens)
    token_ids = tokenizer.encode(concept, add_special_tokens=False)
    logger.info(
        "Concept '%s' tokenizes to %d tokens: %s",
        concept,
        len(token_ids),
        tokenizer.convert_ids_to_tokens(token_ids),
    )

    if len(token_ids) >= prefix_length:
        # Truncate to prefix_length
        token_ids = token_ids[:prefix_length]
        embeds = embedding_matrix[token_ids].clone().detach().float()
    else:
        # Use concept tokens + pad with random
        concept_embeds = embedding_matrix[token_ids].clone().detach().float()
        n_pad = prefix_length - len(token_ids)
        pad_embeds = init_random(embedding_matrix, n_pad, seed=seed)
        embeds = torch.cat([concept_embeds, pad_embeds], dim=0)

    return embeds


def init_logit_lens(
    embedding_matrix: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor,
    direction: torch.Tensor,
    prefix_length: int,
) -> torch.Tensor:
    """Initialize from tokens that the unembedding matrix associates with the direction.

    Project the concept direction through the model's language model head (unembedding)
    to find which tokens the model considers most relevant to this direction.

    Args:
        embedding_matrix: (V, d_model) token embeddings
        lm_head_weight: (V, d_model) output head weight matrix
        lm_head_bias: (V,) or None, output head bias
        direction: (d_model,) concept direction vector
        prefix_length: K
    """
    # Project direction through lm_head: logits = W_u @ v + b
    logits = lm_head_weight.float() @ direction.float()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.float()

    # Top-K tokens by logit score
    topk_values, topk_indices = torch.topk(logits, prefix_length)
    embeds = embedding_matrix[topk_indices].clone().detach().float()

    logger.info(
        "Logit-lens init: top-%d tokens with logits %s",
        prefix_length,
        topk_values[:5].tolist(),
    )
    return embeds


def init_agop(
    embedding_matrix: torch.Tensor,
    agop_matrix: torch.Tensor,
    prefix_length: int,
) -> torch.Tensor:
    """Initialize from tokens with highest Mahalanobis norm in AGOP feature space.

    Tokens with high E[t]^T M E[t] are "concept-relevant" according to the
    RFM feature matrix.

    Args:
        embedding_matrix: (V, d_model)
        agop_matrix: (d_model, d_model) the AGOP/feature matrix M from RFM
        prefix_length: K
    """
    # Compute Mahalanobis norm for each token: E[t]^T M E[t]
    E = embedding_matrix.float()
    M = agop_matrix.float()

    # ME = M @ E^T -> (d, V), then element-wise multiply with E^T and sum
    ME = M @ E.t()  # (d, V)
    relevance = (E.t() * ME).sum(dim=0)  # (V,)

    topk_values, topk_indices = torch.topk(relevance, prefix_length)
    embeds = embedding_matrix[topk_indices].clone().detach().float()

    logger.info(
        "AGOP init: top-%d tokens with relevance scores %s",
        prefix_length,
        topk_values[:5].tolist(),
    )
    return embeds


INIT_STRATEGIES = {
    "random": init_random,
    "concept_name": init_concept_name,
    "logit_lens": init_logit_lens,
    "agop": init_agop,
}
