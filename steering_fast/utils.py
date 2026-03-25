"""Shared utilities: seeding, paths, file I/O, config loading."""
import logging
import os
import random
import hashlib
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Note: Full determinism on GPU is not guaranteed due to parallel reduction
    in operations like softmax. This minimizes variance, not eliminates it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # benchmark=False for determinism; benchmark=True for speed
    # We choose determinism since result equivalence matters
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_concept_list(filepath: str, lowercase: bool = True) -> List[str]:
    """Read concepts from file, one per line, sorted and deduplicated.

    Args:
        filepath: Path to concept list file
        lowercase: Whether to lowercase all concepts

    Raises:
        FileNotFoundError: If filepath doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Concept file not found: {filepath}")

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


def safe_load_pickle(path: str) -> Optional[Any]:
    """Load pickle file safely, returning None on any error.

    Warning: pickle can execute arbitrary code. Only load trusted files.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError, FileNotFoundError) as e:
        log.warning("Failed to load pickle %s: %s", path, e)
        return None


def save_pickle(data: Any, path: str) -> None:
    """Save data to pickle file with Protocol 5 (O20: 7x faster for large tensors)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=5)


def save_directions_safetensors(directions: Dict[int, torch.Tensor], path: str) -> None:
    """Save direction vectors using safetensors (O18: 76x faster load, zero-copy GPU).

    Falls back to pickle if safetensors is not installed.
    """
    try:
        from safetensors.torch import save_file
        tensors = {f"layer_{k}": v.contiguous().float() for k, v in directions.items()}
        ensure_dir(os.path.dirname(path) or ".")
        save_file(tensors, path)
    except ImportError:
        log.info("safetensors not installed, falling back to pickle")
        save_pickle(directions, path.replace(".safetensors", ".pkl"))


def load_directions_safetensors(path: str, device: str = "cpu") -> Optional[Dict[int, torch.Tensor]]:
    """Load direction vectors from safetensors with zero-copy to device.

    Falls back to pickle if safetensors is not installed.
    """
    try:
        from safetensors.torch import load_file
        if not os.path.exists(path):
            return None
        tensors = load_file(path, device=device)
        return {int(k.replace("layer_", "")): v for k, v in tensors.items()}
    except ImportError:
        pkl_path = path.replace(".safetensors", ".pkl")
        return safe_load_pickle(pkl_path)
    except Exception as e:
        log.warning("Failed to load safetensors %s: %s", path, e)
        return None


def load_env_file(search_paths: Optional[List[str]] = None) -> Optional[str]:
    """Load API key from .env file. Handles both raw token and KEY=VALUE format.

    Searches common locations: .env, ../.env, ../../.env
    """
    if search_paths is None:
        search_paths = [".env", "../.env", "../../.env"]

    for env_path in search_paths:
        if os.path.exists(env_path):
            with open(env_path) as f:
                content = f.read().strip()
            # Handle KEY=VALUE format (from .env.template)
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "OPENAI_API_KEY":
                        return value.strip().strip('"').strip("'")
                elif line.startswith("sk-"):
                    return line
            # If no KEY=VALUE found, treat whole content as token
            if content.startswith("sk-") or content.startswith("hf_"):
                return content
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def config_hash(cfg: Any) -> str:
    """Compute a short hash of the config for checkpoint validation."""
    serialized = json.dumps(
        {
            "model": str(cfg.model.name),
            "steering": str(cfg.steering.method),
            "data": str(cfg.data.concept_class),
            "label_type": str(cfg.training.label_type),
            "rep_token": str(cfg.training.rep_token),
            "batch_size": int(cfg.training.batch_size),
            "seed": int(cfg.seed),
        },
        sort_keys=True,
    )
    return hashlib.md5(serialized.encode()).hexdigest()[:12]


def get_coefficients(cfg: Any) -> List[float]:
    """Get steering coefficients based on model and label type."""
    if cfg.training.label_type == "soft":
        return list(cfg.model.coefficients_soft)
    return list(cfg.model.coefficients_hard)


def get_concept_slice(concepts: List[str], cfg: Any) -> List[str]:
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


# ---------------------------------------------------------------------------
# Core imports helper (shared across pipeline stages)
# ---------------------------------------------------------------------------

@contextmanager
def core_imports_and_cwd(data_dir: str):
    """Context manager that sets up core/ imports and working directory.

    The original code uses relative paths like "data/general_statements/".
    This cd's to the data parent and adds core/ to sys.path.
    Restores both on exit.

    Usage:
        with core_imports_and_cwd(cfg.paths.data_dir):
            from utils import select_llm
            llm = select_llm(...)
    """
    core_dir = os.path.join(os.path.dirname(__file__), "core")
    core_dir = os.path.abspath(core_dir)
    data_dir = os.path.abspath(data_dir)
    data_parent = os.path.dirname(data_dir)
    original_cwd = os.getcwd()

    added_to_path = False
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
        added_to_path = True

    os.chdir(data_parent)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        if added_to_path and core_dir in sys.path:
            sys.path.remove(core_dir)


# ---------------------------------------------------------------------------
# Config loading (replaces Hydra for HPC compatibility)
# ---------------------------------------------------------------------------

def load_config(
    model: str = "llama_3_1_8b",
    steering: str = "rfm",
    data: str = "fears",
    experiment: str = "full",
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load and merge config YAMLs without Hydra process wrapper.

    This replicates Hydra's defaults merging without the process management
    that causes CUDA initialization conflicts on HPC compute nodes.

    Args:
        model: Model config name (filename without .yaml in conf/model/)
        steering: Steering method config name
        data: Concept class config name
        experiment: Experiment preset name (full, smoke_test, timing)
        overrides: Dict of dot-path overrides (e.g. {"training.batch_size": 32})

    Returns:
        OmegaConf DictConfig with all defaults merged and resolved

    Raises:
        FileNotFoundError: If any config file is missing
    """
    from omegaconf import OmegaConf

    conf_dir = os.path.join(os.path.dirname(__file__), "conf")

    # Validate config files exist
    required = {
        "base": os.path.join(conf_dir, "config.yaml"),
        "model": os.path.join(conf_dir, "model", f"{model}.yaml"),
        "steering": os.path.join(conf_dir, "steering", f"{steering}.yaml"),
        "data": os.path.join(conf_dir, "data", f"{data}.yaml"),
        "experiment": os.path.join(conf_dir, "experiment", f"{experiment}.yaml"),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path} ({name})")

    # Load and merge
    base = OmegaConf.load(required["base"])
    model_cfg = OmegaConf.load(required["model"])
    steering_cfg = OmegaConf.load(required["steering"])
    data_cfg = OmegaConf.load(required["data"])
    exp_cfg = OmegaConf.load(required["experiment"])

    cfg = OmegaConf.merge(
        base,
        {"model": model_cfg},
        {"steering": steering_cfg},
        {"data": data_cfg},
        exp_cfg,  # uses @package _global_
    )

    # Apply dot-path overrides
    if overrides:
        override_cfg = OmegaConf.create({})
        for key, val in overrides.items():
            OmegaConf.update(override_cfg, key, val, merge=True)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Remove Hydra-specific keys
    for key in ("hydra", "defaults"):
        if key in cfg:
            with open(os.devnull, "w"):  # suppress OmegaConf warnings
                try:
                    del cfg[key]
                except Exception:
                    pass

    OmegaConf.resolve(cfg)
    return cfg
