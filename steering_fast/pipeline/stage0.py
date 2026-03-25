"""Stage 0: Extract attention-to-prefix for all concepts.

Uses the ORIGINAL direction_utils.get_attns_lastNtoks() to guarantee
identical results. Adds checkpointing, timing, and proper cleanup.
"""
import logging
import os

import numpy as np
import torch

from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, core_imports_and_cwd, ensure_dir, get_concept_slice, read_concept_list, set_seed

log = logging.getLogger(__name__)


def run_stage0(cfg: object, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Extract attention-to-prefix for all concepts using original code."""
    set_seed(cfg.seed)

    data_dir = os.path.abspath(cfg.paths.data_dir)
    output_dir = os.path.join(data_dir, "attention_to_prompt")
    ensure_dir(output_dir)

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    all_concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    concepts = get_concept_slice(all_concepts, cfg)
    log.info("Stage 0: %d/%d concepts", len(concepts), len(all_concepts))

    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, "stage0", config_hash(cfg))
    completed, _ = ckpt.load()

    with core_imports_and_cwd(data_dir):
        from utils import select_llm
        from datasets import get_dataset_fn
        import direction_utils
        import utils as orig_utils
        orig_utils.DATA_DIR = data_dir

        llm = select_llm(model_name=cfg.model.name, attn_implementation="eager")
        n_common_toks = llm.n_added_tokens
        model = llm.language_model
        n_layers = model.config.num_hidden_layers
        head_agg = cfg.training.head_agg

        paired_dataset_fn = get_dataset_fn(cfg.data.concept_class, paired_samples=True)

        for concept in concepts:
            attn_outpath = os.path.join(
                output_dir,
                f"attentions_{head_agg}head_{cfg.model.name}_{concept}_paired_statements.npy",
            )

            if concept in completed or os.path.exists(attn_outpath):
                log.info("Skipping %s (exists)", concept)
                completed.add(concept)
                continue

            with timer.time_concept("stage0", concept):
                paired_data = paired_dataset_fn(llm, concept)
                pairs = np.array(paired_data["inputs"])
                pairs = pairs[np.arange(0, len(pairs), 2)]
                pos_data = [pos for pos, neg in pairs]

                attns = np.zeros((len(pos_data), n_layers, n_common_toks))
                layer_to_attns = direction_utils.get_attns_lastNtoks(
                    pos_data, llm, model, llm.tokenizer, n_common_toks, head_agg
                )
                for layer in range(n_layers):
                    attns[:, layer, :] = layer_to_attns[layer]

                np.save(attn_outpath, attns)
                log.info("Saved attention for %s", concept)

            completed.add(concept)
            ckpt.save(completed, {})
            tracker.log_concept(concept, "stage0", {"saved": True})

        # Cleanup GPU memory
        del llm, model
        torch.cuda.empty_cache()

    ckpt.cleanup()
    log.info("Stage 0 complete: %d concepts", len(concepts))
