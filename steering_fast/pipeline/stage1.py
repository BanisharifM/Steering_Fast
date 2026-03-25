"""Stage 1: Train steering directions per concept.

Uses the ORIGINAL compute_save_directions() which internally calls
NeuralController -> control_toolkits -> direction_utils -> rfm.py.
This guarantees identical direction vectors including normalization,
RFM training, and sign correction. Adds checkpointing and timing.
"""
import logging
import os
import sys

import torch

from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, ensure_dir, read_concept_list, set_seed

log = logging.getLogger(__name__)


def _setup_core_imports():
    """Add core/ to sys.path so original modules can import each other."""
    core_dir = os.path.join(os.path.dirname(__file__), "..", "core")
    core_dir = os.path.abspath(core_dir)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)


def run_stage1(cfg, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Train steering vectors using original code with checkpointing."""
    _setup_core_imports()
    set_seed(cfg.seed)

    data_dir = cfg.paths.data_dir
    use_soft_labels = cfg.training.label_type == "soft"

    # Import original modules (from core/)
    from utils import select_llm, compute_save_directions, get_tokenidx_per_layer_per_concept
    from datasets import get_dataset_fn
    import utils as orig_utils

    # Patch DATA_DIR
    orig_utils.DATA_DIR = data_dir

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    if cfg.smoke_test.enabled:
        concepts = concepts[: cfg.smoke_test.n_concepts]
    log.info("Stage 1: %d concepts, method=%s, labels=%s",
             len(concepts), cfg.steering.method, cfg.training.label_type)

    # Checkpoint
    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, "stage1", config_hash(cfg))
    completed, _ = ckpt.load()

    # Load model ONCE
    llm = select_llm(cfg.model.name)

    dataset_fn = get_dataset_fn(cfg.data.concept_class, paired_samples=False)
    rep_token = cfg.training.rep_token
    head_agg = cfg.training.head_agg
    method = cfg.steering.method
    attn_dir = os.path.join(data_dir, "attention_to_prompt")

    for concept in concepts:
        # Check if direction already exists
        suffix = "_softlabels" if use_soft_labels else ""
        vec_path = os.path.join(
            data_dir, "directions",
            f"{method}_{concept}_tokenidx_{rep_token}_block{suffix}_{cfg.model.name}.pkl",
        )

        if concept in completed or os.path.exists(vec_path):
            log.info("Skipping %s (exists)", concept)
            completed.add(concept)
            continue

        with timer.time_concept("stage1", concept):
            layer_to_token = None
            if rep_token == "max_attn_per_layer":
                layer_to_token = get_tokenidx_per_layer_per_concept(
                    concept, cfg.model.name, head_agg=head_agg, root_dir=attn_dir
                )

            data = dataset_fn(llm, concept, datasize="single")

            # ORIGINAL function handles: normalization, RFM training,
            # sign correction, all 5 methods, everything
            compute_save_directions(
                llm, data, use_soft_labels, concept,
                rep_token=rep_token,
                hidden_state="block",
                layer_to_token=layer_to_token,
                concat_layers=[],
                control_method=method,
                head_agg=head_agg,
            )

            del data
            torch.cuda.empty_cache()
            log.info("Saved direction for %s", concept)

        completed.add(concept)
        ckpt.save(completed, {})
        tracker.log_concept(concept, "stage1", {"saved": True})

    ckpt.cleanup()
    log.info("Stage 1 complete: %d concepts", len(concepts))
