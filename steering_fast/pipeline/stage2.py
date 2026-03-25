"""Stage 2: Generate steered outputs with varying coefficients.

Uses the ORIGINAL utils.generate() which internally uses NeuralController
and generation_utils.hook_model(). Adds checkpointing and timing.
"""
import gc
import logging
import os
import pickle

import numpy as np
import torch
import yaml

from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, core_imports_and_cwd, ensure_dir, get_coefficients, get_concept_slice, read_concept_list, set_seed

log = logging.getLogger(__name__)


def run_stage2(cfg: object, version: int, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Generate steered outputs using original code with checkpointing."""
    set_seed(cfg.seed)

    data_dir = os.path.abspath(cfg.paths.data_dir)
    use_soft_labels = cfg.training.label_type == "soft"
    coefs = get_coefficients(cfg)

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    all_concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    concepts = get_concept_slice(all_concepts, cfg)

    # Load test prompt
    prompts_path = os.path.join(data_dir, cfg.data.test_prompts_file)
    if not os.path.exists(prompts_path):
        prompts_path = os.path.join(os.path.dirname(data_dir), cfg.data.test_prompts_file)
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"test_prompts.yaml not found in {data_dir} or parent")
    with open(prompts_path) as f:
        test_prompts = yaml.safe_load(f)
    prompt_text = test_prompts[cfg.data.concept_class][version]

    # Output filename (matches original format)
    suffix = "softlabels_" if use_soft_labels else ""
    output_dir = os.path.join(data_dir, "cached_outputs")
    ensure_dir(output_dir)
    outpath = os.path.join(
        output_dir,
        f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}"
        f"_block_{suffix}steered_500_concepts_{cfg.model.name}_{version}.pkl",
    )

    max_tokens = 200 if cfg.data.concept_class == "jailbreaking" else int(cfg.generation.max_tokens)
    log.info("Stage 2: %d concepts, version=%d, %d coefficients, max_tokens=%d",
             len(concepts), version, len(coefs), max_tokens)

    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, f"stage2_v{version}", config_hash(cfg))
    completed, all_outputs = ckpt.load()

    with core_imports_and_cwd(data_dir):
        from utils import select_llm, generate
        import utils as orig_utils
        orig_utils.DATA_DIR = data_dir

        llm = select_llm(cfg.model.name)
        layers_to_control = list(range(1, llm.language_model.config.num_hidden_layers))

        for concept in concepts:
            if concept in completed:
                continue

            with timer.time_concept("stage2", concept):
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

        del llm
        torch.cuda.empty_cache()
        gc.collect()

    # Save final output
    with open(outpath, "wb") as f:
        pickle.dump(all_outputs, f, protocol=5)

    ckpt.cleanup()
    log.info("Stage 2 v%d complete: %d concepts -> %s", version, len(all_outputs), outpath)
