"""Loss functions for prefix optimization.

All loss functions take:
    activations: (batch, d_model) or (d_model,) -- hidden states at target position(s)
    target: (d_model,) -- concept direction (unit vector)

And return:
    loss: scalar tensor (lower is better)
    metrics: dict of scalar metrics for logging
"""

import torch
import torch.nn.functional as F


def cosine_loss(activations: torch.Tensor, target: torch.Tensor):
    """1 - cosine_similarity(activations, target).

    Scale-invariant. Measures purely directional alignment.
    Range: [0, 2] where 0 = perfect alignment.
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    target = target.view(1, -1).expand_as(activations)
    cos_sim = F.cosine_similarity(activations, target, dim=-1)
    loss = (1.0 - cos_sim).mean()
    return loss, {"cosine_similarity": cos_sim.mean().item(), "loss": loss.item()}


def projection_loss(activations: torch.Tensor, target: torch.Tensor):
    """Negative dot product with unit target. Rewards magnitude along target direction.

    Not scale-invariant: larger activations along target = lower loss.
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    target_unit = target / target.norm()
    proj = (activations * target_unit.unsqueeze(0)).sum(dim=-1)
    loss = -proj.mean()
    cos_sim = F.cosine_similarity(
        activations, target.view(1, -1).expand_as(activations), dim=-1
    )
    return loss, {
        "projection": proj.mean().item(),
        "cosine_similarity": cos_sim.mean().item(),
        "loss": loss.item(),
    }


def normalized_projection_loss(activations: torch.Tensor, target: torch.Tensor):
    """Negative dot product after normalizing activations. Equivalent to cosine sim
    when target is unit vector, but with different gradient dynamics (no quadratic
    slowdown near alignment).
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    a_norm = F.normalize(activations, dim=-1)
    target_unit = target / target.norm()
    proj = (a_norm * target_unit.unsqueeze(0)).sum(dim=-1)
    loss = -proj.mean()
    return loss, {"cosine_similarity": proj.mean().item(), "loss": loss.item()}


def angular_loss(activations: torch.Tensor, target: torch.Tensor):
    """arccos(cosine_similarity). Directly minimizes the angle.

    Avoids the quadratic slowdown of cosine loss near perfect alignment.
    Range: [0, pi] where 0 = perfect alignment.
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    target = target.view(1, -1).expand_as(activations)
    cos_sim = F.cosine_similarity(activations, target, dim=-1)
    # Clamp for numerical stability with arccos
    cos_sim_clamped = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_sim_clamped)
    loss = angle.mean()
    return loss, {
        "angle_rad": angle.mean().item(),
        "cosine_similarity": cos_sim.mean().item(),
        "loss": loss.item(),
    }


def embedding_proximity_regularization(
    prefix_embeds: torch.Tensor, embedding_matrix: torch.Tensor
) -> torch.Tensor:
    """L2 distance from each prefix embedding to its nearest real token embedding.

    Args:
        prefix_embeds: (K, d_model) optimizable prefix embeddings
        embedding_matrix: (V, d_model) model's token embedding matrix

    Returns:
        Scalar: mean squared distance to nearest token
    """
    # Compute pairwise distances efficiently
    # (K, 1, d) - (1, V, d) -> (K, V) squared distances
    # Use cdist for memory efficiency
    dists = torch.cdist(prefix_embeds.float(), embedding_matrix.float())  # (K, V)
    min_dists, _ = dists.min(dim=1)  # (K,)
    return (min_dists**2).mean()


def norm_regularization(
    prefix_embeds: torch.Tensor, embedding_matrix: torch.Tensor
) -> torch.Tensor:
    """Penalize prefix embeddings whose norm deviates from typical token embedding norm.

    Args:
        prefix_embeds: (K, d_model)
        embedding_matrix: (V, d_model)

    Returns:
        Scalar: mean squared deviation from average token embedding norm
    """
    target_norm = embedding_matrix.float().norm(dim=1).mean()
    prefix_norms = prefix_embeds.float().norm(dim=1)
    return ((prefix_norms - target_norm) ** 2).mean()


LOSS_FUNCTIONS = {
    "cosine": cosine_loss,
    "projection": projection_loss,
    "normalized_projection": normalized_projection_loss,
    "angular": angular_loss,
}
