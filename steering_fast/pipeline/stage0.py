"""Stage 0: Extract attention-to-prefix for all concepts.

Uses the ORIGINAL direction_utils.get_attns_lastNtoks() to guarantee
identical results. Adds checkpointing and timing on top.
"""
import logging
import os
import sys

import numpy as np

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


def run_stage0(cfg, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Extract attention-to-prefix for all concepts using original code."""
    _setup_core_imports()
    set_seed(cfg.seed)

    data_dir = cfg.paths.data_dir
    output_dir = os.path.join(data_dir, "attention_to_prompt")
    ensure_dir(output_dir)

    # Import original modules (from core/)
    from utils import select_llm, read_file
    from datasets import get_dataset_fn
    import direction_utils

    # Patch DATA_DIR in original utils to use our configured path
    import utils as orig_utils
    orig_utils.DATA_DIR = data_dir

    # Load concept list
    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    if cfg.smoke_test.enabled:
        concepts = concepts[: cfg.smoke_test.n_concepts]
    log.info("Stage 0: %d concepts", len(concepts))

    # Checkpoint
    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, "stage0", config_hash(cfg))
    completed, _ = ckpt.load()

    # Load model ONCE (same as original)
    llm = select_llm(model_name=cfg.model.name, attn_implementation="eager")
    NUM_COMMON_TOKS = llm.n_added_tokens
    model = llm.language_model
    n_layers = model.config.num_hidden_layers

    paired_dataset_fn = get_dataset_fn(cfg.data.concept_class, paired_samples=True)

    for concept in concepts:
        head_agg = cfg.training.head_agg
        attn_outpath = os.path.join(
            output_dir,
            f"attentions_{head_agg}head_{cfg.model.name}_{concept}_paired_statements.npy",
        )

        if concept in completed or os.path.exists(attn_outpath):
            log.info("Skipping %s (exists)", concept)
            completed.add(concept)
            continue

        with timer.time_concept("stage0", concept):
            # Use ORIGINAL dataset function (guarantees same data)
            paired_data = paired_dataset_fn(llm, concept)
            pairs = np.array(paired_data["inputs"])
            pairs = pairs[np.arange(0, len(pairs), 2)]  # same halving as original

            pos_data = [pos for pos, neg in pairs]

            # Use ORIGINAL attention extraction (guarantees same results)
            attns = np.zeros((len(pos_data), n_layers, NUM_COMMON_TOKS))
            layer_to_attns = direction_utils.get_attns_lastNtoks(
                pos_data, llm, model, llm.tokenizer, NUM_COMMON_TOKS, head_agg
            )
            for layer in range(n_layers):
                attns[:, layer, :] = layer_to_attns[layer]

            np.save(attn_outpath, attns)
            log.info("Saved attention for %s", concept)

        completed.add(concept)
        ckpt.save(completed, {})
        tracker.log_concept(concept, "stage0", {"saved": True})

    ckpt.cleanup()
    log.info("Stage 0 complete: %d concepts", len(concepts))
