"""Stage 2: Generate steered outputs with varying coefficients.

Uses the ORIGINAL utils.generate() which internally uses NeuralController
and generation_utils.hook_model(). Adds checkpointing and timing.
"""
import gc
import logging
import os
import pickle
import sys

import numpy as np
import torch
import yaml

from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, ensure_dir, get_coefficients, read_concept_list, set_seed

log = logging.getLogger(__name__)


def _setup_core_imports():
    core_dir = os.path.join(os.path.dirname(__file__), "..", "core")
    core_dir = os.path.abspath(core_dir)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)


def run_stage2(cfg, version: int, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Generate steered outputs using original code with checkpointing."""
    _setup_core_imports()
    set_seed(cfg.seed)

    data_dir = cfg.paths.data_dir
    use_soft_labels = cfg.training.label_type == "soft"
    coefs = get_coefficients(cfg)

    # Import original modules
    from utils import select_llm, read_file, generate
    import utils as orig_utils
    orig_utils.DATA_DIR = data_dir

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    if cfg.smoke_test.enabled:
        concepts = concepts[: cfg.smoke_test.n_concepts]

    # Load test prompt (test_prompts.yaml is in data_dir parent or data_dir)
    prompts_path = os.path.join(data_dir, cfg.data.test_prompts_file)
    if not os.path.exists(prompts_path):
        # Try parent directory (original location)
        prompts_path = os.path.join(os.path.dirname(data_dir), cfg.data.test_prompts_file)
    with open(prompts_path) as f:
        test_prompts = yaml.safe_load(f)
    prompt_text = test_prompts[cfg.data.concept_class][version]

    # Output filename (matches original format exactly)
    suffix = "softlabels_" if use_soft_labels else ""
    output_dir = os.path.join(data_dir, "cached_outputs")
    ensure_dir(output_dir)
    outpath = os.path.join(
        output_dir,
        f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}"
        f"_block_{suffix}steered_500_concepts_{cfg.model.name}_{version}.pkl",
    )

    log.info("Stage 2: %d concepts, version=%d, %d coefficients", len(concepts), version, len(coefs))

    # Checkpoint
    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, f"stage2_v{version}", config_hash(cfg))
    completed, all_outputs = ckpt.load()

    # Load model ONCE
    llm = select_llm(cfg.model.name)
    layers_to_control = list(range(1, llm.language_model.config.num_hidden_layers))

    max_tokens = 200 if cfg.data.concept_class == "jailbreaking" else cfg.generation.max_tokens

    for concept in concepts:
        if concept in completed:
            continue

        with timer.time_concept("stage2", concept):
            # Use ORIGINAL generate function (handles hooks, directions, everything)
            outputs = generate(
                concept, llm, prompt_text,
                use_soft_labels=use_soft_labels,
                coefs=coefs,
                rep_token=cfg.training.rep_token,
                control_method=cfg.steering.method,
                max_tokens=max_tokens,
                gen_orig=True,
                hidden_state="block",
                layers_to_control=layers_to_control,
                start_from_token=0,
                head_agg=cfg.training.head_agg,
            )

            all_outputs[concept] = outputs
            log.info("Generated %d outputs for %s", len(outputs), concept)

        completed.add(concept)
        ckpt.save(completed, all_outputs)
        tracker.log_concept(concept, "stage2", {"n_coefs": len(coefs)})

    # Save final output
    with open(outpath, "wb") as f:
        pickle.dump(all_outputs, f)

    ckpt.cleanup()
    log.info("Stage 2 v%d complete: %d concepts -> %s", version, len(all_outputs), outpath)

    del llm
    torch.cuda.empty_cache()
    gc.collect()
