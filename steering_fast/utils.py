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
    """Save data to pickle file, creating parent dirs."""
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_coefficients(cfg) -> List[float]:
    """Get steering coefficients based on model and label type."""
    if cfg.training.label_type == "soft":
        return list(cfg.model.coefficients_soft)
    return list(cfg.model.coefficients_hard)
