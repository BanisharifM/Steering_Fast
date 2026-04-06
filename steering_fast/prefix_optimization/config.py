"""Configuration for prefix optimization experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PrefixOptConfig:
    """Configuration for a single prefix optimization run."""

    # Model
    model_name: str = "llama_3.1_8b"
    model_hf_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hidden_dim: int = 4096
    n_layers: int = 32
    cache_dir: Optional[str] = None

    # Concept
    concept: str = "spiders"
    concept_class: str = "fears"
    data_dir: str = "./data"

    # Direction loading
    steering_method: str = "rfm"
    label_type: str = "soft"
    rep_token: str = "max_attn_per_layer"
    head_agg: str = "mean"

    # Prefix optimization
    prefix_length: int = 10
    n_steps: int = 500
    lr: float = 0.01
    optimizer: str = "adamw"  # adamw, adam, sgd
    grad_clip: float = 1.0

    # Loss
    loss_type: str = "cosine"  # cosine, projection, normalized_projection, angular
    lambda_prox: float = 0.01  # embedding proximity regularization
    lambda_norm: float = 0.001  # norm regularization

    # Initialization
    init_strategy: str = "concept_name"  # random, concept_name, logit_lens, agop

    # Token position for alignment measurement
    target_position: str = "last_prefix"  # last_prefix, last_token, mean_prefix, mean_all

    # Layer selection
    layers: str = "all"  # "all", "16", "12-24", single int, or comma-separated
    layer_weighting: str = "uniform"  # uniform, uncertainty, progressive

    # Multi-statement
    n_statements: int = 1  # number of general statements to average over

    # Method selection
    method: str = "gradient"  # gradient, jacobian, logit_lens, agop, gcg, all

    # Jacobian-specific
    jacobian_rank: int = 64  # rank for randomized SVD

    # GCG-specific
    gcg_topk: int = 256  # top-k candidates per position
    gcg_batch_size: int = 64  # batch size for candidate evaluation (uses model.model to skip logits)

    # Output
    output_dir: str = "outputs/prefix_optimization"
    save_checkpoints: bool = True
    log_every: int = 50  # log metrics every N steps

    # Reproducibility
    seed: int = 42

    def get_layers(self) -> List[int]:
        """Parse layer specification into list of layer indices."""
        if self.layers == "all":
            return list(range(1, self.n_layers))  # skip layer 0
        if "-" in self.layers:
            start, end = self.layers.split("-")
            return list(range(int(start), int(end) + 1))
        if "," in self.layers:
            return [int(x) for x in self.layers.split(",")]
        return [int(self.layers)]

    def get_direction_filename(self, concept: Optional[str] = None) -> str:
        """Build direction pickle filename matching existing pipeline convention."""
        c = concept or self.concept
        label_suffix = "_softlabels" if self.label_type == "soft" else ""
        return (
            f"{self.steering_method}_{c}_tokenidx_{self.rep_token}"
            f"_block{label_suffix}_{self.model_name}.pkl"
        )
