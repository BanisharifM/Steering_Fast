"""Shared utilities: seeding, paths, file I/O."""
import os
import random
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_concept_list(filepath: str, lowercase: bool = True) -> List[str]:
    """Read concepts from file, one per line, sorted and deduplicated."""
    concepts = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if lowercase:
                text = text.lower()
            concepts.append(text)
    return sorted(set(concepts))


def config_hash(cfg: Any) -> str:
    """Compute a short hash of the config for checkpoint validation."""
    serialized = json.dumps(
        {
            "model": cfg.model.name,
            "steering": cfg.steering.method,
            "data": cfg.data.concept_class,
            "label_type": cfg.training.label_type,
            "rep_token": cfg.training.rep_token,
            "batch_size": cfg.training.batch_size,
            "seed": cfg.seed,
        },
        sort_keys=True,
    )
    return hashlib.md5(serialized.encode()).hexdigest()[:12]


def safe_load_pickle(path: str) -> Optional[Any]:
    """Load pickle file safely, returning None on any error."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError, FileNotFoundError):
        return None


def save_pickle(data: Any, path: str) -> None:
    """Save data to pickle file, creating parent dirs. Uses Protocol 5 for speed."""
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=5)


def save_directions_safetensors(directions: dict, path: str) -> None:
    """Save direction vectors using safetensors (76x faster load, zero-copy GPU).

    Args:
        directions: Dict mapping layer_idx (int) -> torch.Tensor
        path: Output file path (should end in .safetensors)
    """
    try:
        from safetensors.torch import save_file
        tensors = {f"layer_{k}": v.contiguous() for k, v in directions.items()}
        ensure_dir(os.path.dirname(path))
        save_file(tensors, path)
    except ImportError:
        # Fallback to pickle if safetensors not installed
        save_pickle(directions, path.replace(".safetensors", ".pkl"))


def load_directions_safetensors(path: str, device: str = "cpu") -> Optional[dict]:
    """Load direction vectors from safetensors with zero-copy to device.

    Returns dict mapping layer_idx (int) -> torch.Tensor, or None on error.
    """
    try:
        from safetensors.torch import load_file
        if not os.path.exists(path):
            return None
        tensors = load_file(path, device=device)
        return {int(k.replace("layer_", "")): v for k, v in tensors.items()}
    except ImportError:
        # Fallback to pickle
        pkl_path = path.replace(".safetensors", ".pkl")
        return safe_load_pickle(pkl_path)
    except Exception:
        return None


def get_coefficients(cfg) -> List[float]:
    """Get steering coefficients based on model and label type."""
    if cfg.training.label_type == "soft":
        return list(cfg.model.coefficients_soft)
    return list(cfg.model.coefficients_hard)


def get_concept_slice(concepts: List[str], cfg) -> List[str]:
    """Apply concept slicing for SLURM array jobs and smoke tests.

    Priority: smoke_test.enabled > slicing.enabled > all concepts.
    """
    if cfg.smoke_test.enabled:
        return concepts[: cfg.smoke_test.n_concepts]

    if hasattr(cfg, "slicing") and cfg.slicing.enabled:
        start = cfg.slicing.start
        end = cfg.slicing.end if cfg.slicing.end is not None else len(concepts)
        return concepts[start:end]

    return concepts


def load_config(
    model: str = "llama_3_1_8b",
    steering: str = "rfm",
    data: str = "fears",
    experiment: str = "full",
    overrides: Optional[dict] = None,
) -> Any:
    """Load and merge config YAMLs without Hydra process wrapper.

    This replicates what Hydra does (merging defaults) but without the
    process management that causes CUDA conflicts on HPC.

    Args:
        model: Model config name (filename without .yaml in conf/model/)
        steering: Steering config name
        data: Data config name
        experiment: Experiment preset name
        overrides: Dict of dot-path overrides (e.g. {"training.batch_size": 32})

    Returns:
        OmegaConf DictConfig with all defaults merged
    """
    from omegaconf import OmegaConf

    conf_dir = os.path.join(os.path.dirname(__file__), "conf")

    # Load each config group
    base = OmegaConf.load(os.path.join(conf_dir, "config.yaml"))
    model_cfg = OmegaConf.load(os.path.join(conf_dir, "model", f"{model}.yaml"))
    steering_cfg = OmegaConf.load(os.path.join(conf_dir, "steering", f"{steering}.yaml"))
    data_cfg = OmegaConf.load(os.path.join(conf_dir, "data", f"{data}.yaml"))
    exp_cfg = OmegaConf.load(os.path.join(conf_dir, "experiment", f"{experiment}.yaml"))

    # Merge: base <- model/steering/data <- experiment <- overrides
    cfg = OmegaConf.merge(
        base,
        {"model": model_cfg},
        {"steering": steering_cfg},
        {"data": data_cfg},
        exp_cfg,  # experiment uses @package _global_ so keys are at root level
    )

    # Apply overrides (supports dot-path like "paths.data_dir")
    if overrides:
        override_cfg = OmegaConf.create({})
        for key, val in overrides.items():
            OmegaConf.update(override_cfg, key, val, merge=True)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Remove hydra key if present (not needed at runtime)
    if "hydra" in cfg:
        del cfg["hydra"]
    # Remove defaults key if present
    if "defaults" in cfg:
        del cfg["defaults"]

    # Resolve environment variables
    OmegaConf.resolve(cfg)

    return cfg
