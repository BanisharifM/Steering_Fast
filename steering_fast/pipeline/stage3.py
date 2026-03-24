"""Stage 3: Evaluate steered outputs with GPT-4o.

Fixes over original:
- Rate limiting with exponential backoff (handles 429 AND transient errors)
- Checkpoint/resume per concept (no progress lost on crash)
- Multi-digit score parsing (regex instead of single-char indexing)
"""
import logging
import os

import pandas as pd

from ..evaluation.openai_eval import OpenAIEvaluator, load_eval_prompt, parse_model_response
from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, ensure_dir, read_concept_list, safe_load_pickle, set_seed

log = logging.getLogger(__name__)


def run_stage3(cfg, version: int, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Evaluate steered outputs for all concepts at a specific version."""
    set_seed(cfg.seed)

    data_dir = cfg.paths.data_dir
    csv_dir = os.path.join(data_dir, "csvs")
    outputs_dir = os.path.join(data_dir, "cached_outputs")
    ensure_dir(csv_dir)

    use_soft_labels = cfg.training.label_type == "soft"

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    if cfg.smoke_test.enabled:
        concepts = concepts[: cfg.smoke_test.n_concepts]

    # Load steered outputs
    suffix = "softlabels_" if use_soft_labels else ""
    pkl_path = os.path.join(
        outputs_dir,
        f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}_block_{suffix}steered_500_concepts_{cfg.model.name}_{version}.pkl",
    )

    results = safe_load_pickle(pkl_path)
    if results is None:
        log.error("No steered outputs found at %s", pkl_path)
        return

    # Output CSV path (matches original format)
    csv_suffix = "_softlabels" if use_soft_labels else ""
    csv_path = os.path.join(
        csv_dir,
        f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}_block{csv_suffix}_gpt4o_outputs_500_concepts_{cfg.model.name}_{version}.csv",
    )

    # Check if already complete
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        if len(existing) == len(concepts):
            log.info("Stage 3 v%d already complete: %s", version, csv_path)
            return

    log.info("Stage 3: %d concepts, version=%d", len(concepts), version)

    # Checkpoint
    ckpt_name = f"stage3_v{version}"
    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, ckpt_name, config_hash(cfg))
    completed, eval_results = ckpt.load()

    # Init evaluator with rate limiting
    evaluator = OpenAIEvaluator(
        model=cfg.evaluation.openai_model,
        delay=cfg.evaluation.rate_limit_delay,
        max_retries=cfg.evaluation.max_retries,
        temperature=cfg.evaluation.temperature,
        max_tokens=cfg.evaluation.max_tokens,
    )

    # Load evaluation prompt template
    prompt_template = load_eval_prompt(data_dir, cfg.data.concept_class, version)

    for concept in concepts:
        if concept not in results:
            log.warning("No outputs for concept %s, skipping", concept)
            continue

        if concept in completed:
            continue

        with timer.time_concept("stage3", concept):
            responses = results[concept]
            best_score = 0
            best_coef = 0.0

            for response in responses:
                coef = response[0]
                parsed = parse_model_response(response[1], cfg.model.name)
                if not parsed:
                    parsed = "None"

                prompt = prompt_template.format(personality=concept, parsed_response=parsed)
                score, gpt_output = evaluator.score_response(prompt)

                if score > best_score:
                    best_score = score
                    best_coef = coef

                log.debug("  %s coef=%.2f score=%d", concept, coef, score)

            eval_results[concept] = (concept, best_score, best_coef)
            log.info("%s: best_score=%d, best_coef=%.2f", concept, best_score, best_coef)

        completed.add(concept)
        ckpt.save(completed, eval_results)
        tracker.log_concept(concept, "stage3", {"best_score": best_score, "best_coef": best_coef})

    # Save CSV
    rows = [eval_results[c] for c in concepts if c in eval_results]
    df = pd.DataFrame(rows, columns=["mood", "best_score", "best_coef"])
    df.to_csv(csv_path, index=False)

    ckpt.cleanup()
    log.info("Stage 3 v%d complete: %.1f%% success, saved to %s",
             version, df.best_score.mean() * 100, csv_path)
