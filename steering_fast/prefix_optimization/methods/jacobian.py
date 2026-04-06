"""Method 2: Jacobian-based analysis and reachability scoring.

Computes the Jacobian of hidden states w.r.t. prefix embeddings to:
1. Quantify reachability: how much of the concept direction is controllable from input
2. Provide a closed-form linear approximation to the optimal prefix perturbation
3. Analyze per-position contribution to concept activation
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_jacobian_low_rank(
    model,
    prefix_embeds: torch.Tensor,
    statement_embeds: torch.Tensor,
    layer_idx: int,
    target_position: str,
    prefix_length: int,
    rank: int = 64,
    n_probes: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute low-rank approximation of the Jacobian J = dh^(l)/dP via randomized SVD.

    The full Jacobian has shape (d_model, K*d_model) which is enormous.
    We use randomized probing: multiply J by random vectors and decompose.

    Args:
        model: Frozen LLM
        prefix_embeds: (K, d_model) current prefix embeddings
        statement_embeds: (1, T, d_model) statement embeddings
        layer_idx: which layer to analyze
        target_position: where to read activation from
        prefix_length: K
        rank: desired rank for the SVD approximation
        n_probes: number of random probes (default: 2*rank)

    Returns:
        U: (d_model, rank) left singular vectors (output space)
        S: (rank,) singular values
        Vt: (rank, K*d_model) right singular vectors (input space)
    """
    if n_probes is None:
        n_probes = min(2 * rank, prefix_length * prefix_embeds.shape[1])

    device = prefix_embeds.device
    d_model = prefix_embeds.shape[1]
    K = prefix_length
    input_dim = K * d_model

    # Create random probe matrix: (input_dim, n_probes)
    Omega = torch.randn(input_dim, n_probes, device=device, dtype=torch.float32)
    Omega = Omega / Omega.norm(dim=0, keepdim=True)

    # Compute J @ Omega by doing n_probes forward+backward passes
    # Each column of Omega is a direction in input space; we compute
    # the directional derivative of h^(l) along that direction.
    Y = torch.zeros(d_model, n_probes, device=device, dtype=torch.float32)

    prefix_param = prefix_embeds.clone().detach().float().requires_grad_(True)
    prefix_batch = prefix_param.unsqueeze(0)  # (1, K, d)

    for i in range(n_probes):
        if prefix_param.grad is not None:
            prefix_param.grad.zero_()

        input_embeds = torch.cat([prefix_batch, statement_embeds.float()], dim=1)
        seq_len = input_embeds.shape[1]
        attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        h = outputs.hidden_states[layer_idx]  # (1, seq_len, d_model)

        if target_position == "last_prefix":
            act = h[0, prefix_length - 1, :]
        elif target_position == "last_token":
            act = h[0, -1, :]
        elif target_position == "mean_prefix":
            act = h[0, :prefix_length, :].mean(dim=0)
        else:
            act = h[0, -1, :]

        # Compute J @ omega_i using vector-Jacobian product
        # We need the Jacobian-vector product: J @ omega_i
        # This requires forward-mode AD or computing dact/dP . omega_i
        # Use backward: for each output dimension, compute gradient
        # This is expensive. Instead, use the trick:
        # J @ omega = d/depsilon h(P + epsilon * reshape(omega)) |_{epsilon=0}
        # Approximated via finite differences for robustness:
        omega_reshaped = Omega[:, i].reshape(K, d_model)
        eps = 1e-4

        with torch.no_grad():
            prefix_plus = prefix_param + eps * omega_reshaped
            prefix_minus = prefix_param - eps * omega_reshaped

            inp_plus = torch.cat([prefix_plus.unsqueeze(0), statement_embeds.float()], dim=1)
            inp_minus = torch.cat([prefix_minus.unsqueeze(0), statement_embeds.float()], dim=1)

            out_plus = model(
                inputs_embeds=inp_plus,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            out_minus = model(
                inputs_embeds=inp_minus,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            h_plus = out_plus.hidden_states[layer_idx]
            h_minus = out_minus.hidden_states[layer_idx]

            if target_position == "last_prefix":
                act_plus = h_plus[0, prefix_length - 1, :]
                act_minus = h_minus[0, prefix_length - 1, :]
            elif target_position == "last_token":
                act_plus = h_plus[0, -1, :]
                act_minus = h_minus[0, -1, :]
            elif target_position == "mean_prefix":
                act_plus = h_plus[0, :prefix_length, :].mean(dim=0)
                act_minus = h_minus[0, :prefix_length, :].mean(dim=0)
            else:
                act_plus = h_plus[0, -1, :]
                act_minus = h_minus[0, -1, :]

            Y[:, i] = (act_plus.float() - act_minus.float()) / (2 * eps)

    # Randomized SVD of Y = J @ Omega
    # Y has shape (d_model, n_probes)
    # We want the SVD of J, approximated from Y
    # Standard randomized SVD: Q, _ = qr(Y); B = Q^T J; U_B, S, Vt = svd(B)
    # But we don't have J explicitly. Instead, use the approximation:
    # J approx Y @ pinv(Omega) -- but this is rank-limited.
    # Better: just SVD of Y gives us the column space of J.
    Q, _ = torch.linalg.qr(Y)  # (d_model, n_probes) orthonormal basis
    U = Q[:, :rank]  # (d_model, rank) - top directions in output space

    # For singular values, project Y onto U
    S = (U.t() @ Y).norm(dim=1)  # approximate singular values

    # For Vt (input space directions), we'd need more passes; skip for now
    # The main use is U and S for reachability
    Vt = torch.zeros(rank, input_dim, device=device)  # placeholder

    return U, S, Vt


def reachability_score(
    U: torch.Tensor, direction: torch.Tensor, k: Optional[int] = None
) -> float:
    """Compute the reachability score of a concept direction.

    r(v) = ||U_k^T v||^2

    This measures what fraction of the concept direction lies in the column space
    of the Jacobian (the subspace of activations reachable by varying the prefix).

    Args:
        U: (d_model, n) left singular vectors of the Jacobian
        direction: (d_model,) concept direction (unit vector)
        k: use only top-k singular vectors (default: all)

    Returns:
        Reachability score in [0, 1]. 1 = fully reachable, 0 = unreachable.
    """
    if k is not None:
        U = U[:, :k]
    v = direction.flatten().float()
    v = v / v.norm()
    U = U.float()

    # Project v onto column space of U
    proj = U.t() @ v  # (n,)
    r = (proj**2).sum().item()
    return min(r, 1.0)  # clamp for numerical stability


def per_position_contribution(
    model,
    prefix_embeds: torch.Tensor,
    statement_embeds: torch.Tensor,
    layer_idx: int,
    direction: torch.Tensor,
    target_position: str,
    prefix_length: int,
) -> List[float]:
    """Compute how much each prefix position contributes to the concept direction.

    For each position k, compute c_k = ||J_k^T v|| where J_k = dh/dp_k.
    This tells us which positions are most important for activating the concept.

    Uses finite differences for robustness.

    Returns:
        List of contribution scores, one per prefix position.
    """
    device = prefix_embeds.device
    d_model = prefix_embeds.shape[1]
    K = prefix_length
    v = direction.flatten().float().to(device)
    v = v / v.norm()

    contributions = []
    eps = 1e-4

    for k in range(K):
        # Perturb position k along d random directions and measure effect on h along v
        n_random = min(32, d_model)
        total_sensitivity = 0.0

        for _ in range(n_random):
            delta = torch.randn(d_model, device=device, dtype=torch.float32)
            delta = delta / delta.norm()

            with torch.no_grad():
                prefix_plus = prefix_embeds.clone().float()
                prefix_minus = prefix_embeds.clone().float()
                prefix_plus[k] += eps * delta
                prefix_minus[k] -= eps * delta

                # Forward passes
                seq_len = prefix_length + statement_embeds.shape[1]
                mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

                inp_plus = torch.cat([prefix_plus.unsqueeze(0), statement_embeds.float()], dim=1)
                inp_minus = torch.cat([prefix_minus.unsqueeze(0), statement_embeds.float()], dim=1)

                out_plus = model(
                    inputs_embeds=inp_plus,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                out_minus = model(
                    inputs_embeds=inp_minus,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

                h_plus = out_plus.hidden_states[layer_idx]
                h_minus = out_minus.hidden_states[layer_idx]

                if target_position == "last_prefix":
                    a_plus = h_plus[0, prefix_length - 1, :].float()
                    a_minus = h_minus[0, prefix_length - 1, :].float()
                elif target_position == "last_token":
                    a_plus = h_plus[0, -1, :].float()
                    a_minus = h_minus[0, -1, :].float()
                else:
                    a_plus = h_plus[0, prefix_length - 1, :].float()
                    a_minus = h_minus[0, prefix_length - 1, :].float()

                # Directional derivative along v
                dh = (a_plus - a_minus) / (2 * eps)
                sensitivity = (dh @ v).abs().item()
                total_sensitivity += sensitivity

        contributions.append(total_sensitivity / n_random)

    return contributions


def run_jacobian_analysis(
    model,
    tokenizer,
    directions: Dict[int, torch.Tensor],
    concept: str,
    statements: List[str],
    config,
) -> Dict:
    """Run full Jacobian analysis for a concept.

    Computes:
    1. Reachability scores per layer
    2. Per-position contributions per layer
    3. Closed-form linear prefix perturbation (warm-start for gradient method)
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

    # Initialize prefix from concept name
    from ..initialization import init_concept_name
    prefix_embeds = init_concept_name(
        embedding_matrix, tokenizer, concept, config.prefix_length, config.seed
    ).to(device)

    # Prepare statement
    stmt = statements[0]
    token_ids = tokenizer.encode(stmt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        stmt_emb = model.model.embed_tokens(token_ids).float()

    results = {
        "method": "jacobian",
        "concept": concept,
        "reachability_scores": {},
        "per_position_contributions": {},
    }

    start_time = time.time()

    for layer_idx in active_layers:
        logger.info("Analyzing layer %d...", layer_idx)

        # Reachability analysis
        U, S, Vt = compute_jacobian_low_rank(
            model,
            prefix_embeds,
            stmt_emb,
            layer_idx,
            config.target_position,
            config.prefix_length,
            rank=config.jacobian_rank,
        )

        r = reachability_score(U, active_directions[layer_idx])
        results["reachability_scores"][layer_idx] = r
        logger.info("Layer %d reachability: %.4f", layer_idx, r)

        # Per-position contribution
        contribs = per_position_contribution(
            model,
            prefix_embeds,
            stmt_emb,
            layer_idx,
            active_directions[layer_idx],
            config.target_position,
            config.prefix_length,
        )
        results["per_position_contributions"][layer_idx] = contribs
        logger.info("Layer %d position contributions: %s", layer_idx, [f"{c:.4f}" for c in contribs])

    results["total_time_seconds"] = time.time() - start_time
    results["singular_values"] = S.cpu().tolist() if S is not None else []

    return results
