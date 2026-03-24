"""Stage 2: Generate steered outputs with varying coefficients.

Optimizations over original:
- Flash Attention 2 (no attention weights needed for generation)
- Checkpoint/resume per concept
"""
import gc
import logging
import os
import pickle

import numpy as np
import torch
import yaml

from ..generation.hooks import generate_steered
from ..model_loader import load_model
from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, ensure_dir, get_coefficients, read_concept_list, safe_load_pickle, save_pickle, set_seed

log = logging.getLogger(__name__)


def run_stage2(cfg, version: int, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Generate steered outputs for all concepts at a specific version."""
    set_seed(cfg.seed)

    data_dir = cfg.paths.data_dir
    output_dir = os.path.join(data_dir, "cached_outputs")
    directions_dir = os.path.join(data_dir, "directions")
    ensure_dir(output_dir)

    use_soft_labels = cfg.training.label_type == "soft"
    coefs = get_coefficients(cfg)

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    if cfg.smoke_test.enabled:
        concepts = concepts[: cfg.smoke_test.n_concepts]

    # Load test prompt
    prompts_file = os.path.join(data_dir, cfg.data.test_prompts_file)
    with open(prompts_file) as f:
        test_prompts = yaml.safe_load(f)
    prompt_text = test_prompts[cfg.data.concept_class][version]

    # Output filename (matches original format)
    suffix = "softlabels_" if use_soft_labels else ""
    outpath = os.path.join(
        output_dir,
        f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}_block_{suffix}steered_500_concepts_{cfg.model.name}_{version}.pkl",
    )

    log.info("Stage 2: %d concepts, version=%d, %d coefficients", len(concepts), version, len(coefs))

    # Checkpoint
    ckpt_name = f"stage2_v{version}"
    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, ckpt_name, config_hash(cfg))
    completed, all_outputs = ckpt.load()

    # Load model with flash attention (no attention weights needed for generation)
    llm = load_model(cfg.model.name, cfg.model.hf_id, cfg.paths.cache_dir, attn_implementation="flash_attention_2")
    layers_to_control = list(range(1, llm.language_model.config.num_hidden_layers))

    for concept in concepts:
        if concept in completed:
            continue

        with timer.time_concept("stage2", concept):
            # Load directions for this concept
            dir_suffix = "_softlabels" if use_soft_labels else ""
            dir_path = os.path.join(
                directions_dir,
                f"{cfg.steering.method}_{concept}_tokenidx_{cfg.training.rep_token}_block{dir_suffix}_{cfg.model.name}.pkl",
            )
            if not os.path.exists(dir_path):
                log.warning("No directions for %s, skipping", concept)
                continue

            with open(dir_path, "rb") as f:
                directions = pickle.load(f)

            # Generate original (unsteered) output
            original = generate_steered(
                llm.language_model, llm.tokenizer, prompt_text.format(concept=concept) if "{concept}" in prompt_text else prompt_text,
                directions, [], 0.0,
                max_new_tokens=cfg.generation.max_tokens,
                do_sample=cfg.generation.do_sample,
            )

            # Generate steered outputs at each coefficient
            outputs = []
            for coef in coefs:
                steered = generate_steered(
                    llm.language_model, llm.tokenizer, prompt_text,
                    directions, layers_to_control, coef,
                    max_new_tokens=cfg.generation.max_tokens,
                    do_sample=cfg.generation.do_sample,
                )
                outputs.append((coef, steered))
                log.debug("  coef=%.2f: %s...", coef, steered[:80])

            all_outputs[concept] = outputs

        completed.add(concept)
        ckpt.save(completed, all_outputs)
        tracker.log_concept(concept, "stage2", {"n_coefs": len(coefs)})

    # Save final output
    save_pickle(all_outputs, outpath)
    ckpt.cleanup()
    log.info("Stage 2 v%d complete: %d concepts, saved to %s", version, len(all_outputs), outpath)

    # Free GPU
    del llm
    torch.cuda.empty_cache()
    gc.collect()
