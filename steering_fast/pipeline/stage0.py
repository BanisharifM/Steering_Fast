"""Stage 0: Extract attention-to-prefix for all concepts.

Optimizations over original:
- Batched forward passes (configurable batch_size)
- Vectorized attention aggregation (no Python head loop)
- Pre-allocated output arrays
- Checkpoint/resume per concept
"""
import logging
import os

import numpy as np

from ..data.statements import StatementCache
from ..extraction.attention import extract_attention_batched, get_prefix_indices
from ..model_loader import load_model
from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, ensure_dir, read_concept_list, set_seed

log = logging.getLogger(__name__)


def run_stage0(cfg, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Extract attention-to-prefix for all concepts."""
    set_seed(cfg.seed)

    data_dir = cfg.paths.data_dir
    output_dir = os.path.join(data_dir, "attention_to_prompt")
    ensure_dir(output_dir)

    # Load concept list
    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    if cfg.smoke_test.enabled:
        concepts = concepts[: cfg.smoke_test.n_concepts]
    log.info("Stage 0: %d concepts to process", len(concepts))

    # Checkpoint
    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, "stage0", config_hash(cfg))
    completed, _ = ckpt.load()

    # Load model (eager attention required for output_attentions=True)
    llm = load_model(cfg.model.name, cfg.model.hf_id, cfg.paths.cache_dir, attn_implementation="eager")
    stmt_cache = StatementCache(data_dir, cfg.training.datasize)

    for concept in concepts:
        outpath = os.path.join(output_dir, f"attentions_{cfg.training.head_agg}head_{cfg.model.name}_{concept}_paired_statements.npy")

        if concept in completed or os.path.exists(outpath):
            log.info("Skipping %s (already done)", concept)
            completed.add(concept)
            continue

        with timer.time_concept("stage0", concept):
            # Get paired dataset
            dataset = stmt_cache.get_paired_dataset(
                concept, cfg.data.positive_template, cfg.data.negative_template, llm.tokenizer,
            )
            pairs = dataset["inputs"]

            # Take every other pair (same as original: efficiency halving)
            pos_prompts = [p[0] for p in pairs[::2]]

            # Extract attention (batched + vectorized)
            layer_attns = extract_attention_batched(
                pos_prompts, llm.language_model, llm.tokenizer,
                llm.n_added_tokens, cfg.training.head_agg, cfg.training.batch_size,
            )

            # Stack into array: (N, n_layers, n_common_toks)
            n_layers = llm.language_model.config.num_hidden_layers
            attns = np.zeros((len(pos_prompts), n_layers, llm.n_added_tokens))
            for layer in range(n_layers):
                attns[:, layer, :] = layer_attns[layer]

            np.save(outpath, attns)
            log.info("Saved attention for %s: %s", concept, outpath)

        completed.add(concept)
        ckpt.save(completed, {})
        tracker.log_concept(concept, "stage0", {"saved": True})

    ckpt.cleanup()
    log.info("Stage 0 complete: %d concepts", len(concepts))
