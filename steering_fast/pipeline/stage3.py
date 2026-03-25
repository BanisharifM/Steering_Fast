"""Stage 3: Evaluate steered outputs with GPT-4o.

Uses the ORIGINAL prompt templates and response parsing, but adds:
- Rate limiting with exponential backoff on ALL errors
- Checkpoint/resume per concept (no progress lost on crash)
- Multi-digit score parsing fix
"""
import logging
import os

import pandas as pd

from ..evaluation.openai_eval import OpenAIEvaluator
from ..tracking.checkpoint import CheckpointManager
from ..tracking.timer import PipelineTimer
from ..tracking.wandb_tracker import WandbTracker
from ..utils import config_hash, core_imports_and_cwd, ensure_dir, get_concept_slice, read_concept_list, safe_load_pickle, set_seed

log = logging.getLogger(__name__)


def run_stage3(cfg: object, version: int, timer: PipelineTimer, tracker: WandbTracker) -> None:
    """Evaluate steered outputs with rate-limited GPT-4o and checkpointing."""
    set_seed(cfg.seed)

    data_dir = os.path.abspath(cfg.paths.data_dir)
    use_soft_labels = cfg.training.label_type == "soft"

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    all_concepts = read_concept_list(concept_file, lowercase=cfg.data.lowercase)
    concepts = get_concept_slice(all_concepts, cfg)

    # Load steered outputs
    suffix = "softlabels_" if use_soft_labels else ""
    pkl_path = os.path.join(
        data_dir, "cached_outputs",
        f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}"
        f"_block_{suffix}steered_500_concepts_{cfg.model.name}_{version}.pkl",
    )
    results = safe_load_pickle(pkl_path)
    if results is None:
        log.error("No steered outputs at %s", pkl_path)
        return

    # Output CSV
    csv_suffix = "_softlabels" if use_soft_labels else ""
    csv_dir = os.path.join(data_dir, "csvs")
    ensure_dir(csv_dir)
    csv_path = os.path.join(
        csv_dir,
        f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}"
        f"_block{csv_suffix}_gpt4o_outputs_500_concepts_{cfg.model.name}_{version}.csv",
    )

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        if len(existing) >= len(concepts):
            log.info("Stage 3 v%d already complete", version)
            return

    log.info("Stage 3: %d concepts, version=%d", len(concepts), version)

    ckpt = CheckpointManager(cfg.paths.checkpoint_dir, f"stage3_v{version}", config_hash(cfg))
    completed, eval_results = ckpt.load()

    evaluator = OpenAIEvaluator(
        model=cfg.evaluation.openai_model,
        delay=cfg.evaluation.rate_limit_delay,
        max_retries=cfg.evaluation.max_retries,
        temperature=cfg.evaluation.temperature,
        max_tokens=cfg.evaluation.max_tokens,
    )

    with core_imports_and_cwd(data_dir):
        from utils import parse_personality_responses, load_prompt
        import utils as orig_utils
        orig_utils.DATA_DIR = data_dir

        for concept in concepts:
            if concept not in results:
                log.warning("No outputs for %s, skipping", concept)
                continue
            if concept in completed:
                continue

            with timer.time_concept("stage3", concept):
                responses = results[concept]
                best_score = 0
                best_coef = 0.0

                for response in responses:
                    parsed = parse_personality_responses(response, cfg.model.name)
                    if not parsed:
                        parsed = "None"

                    prompt_template = load_prompt(cfg.data.concept_class, version)
                    prompt = prompt_template.format(personality=concept, parsed_response=parsed)

                    score, _ = evaluator.score_response(prompt)
                    if score > best_score:
                        best_score = score
                        best_coef = response[0]

                eval_results[concept] = (concept, best_score, best_coef)
                log.info("%s: score=%d, coef=%.2f", concept, best_score, best_coef)

            completed.add(concept)
            ckpt.save(completed, eval_results)
            tracker.log_concept(concept, "stage3", {"best_score": best_score, "best_coef": best_coef})

    # Save CSV
    rows = [eval_results[c] for c in concepts if c in eval_results]
    df = pd.DataFrame(rows, columns=["concept", "best_score", "best_coef"])
    df.to_csv(csv_path, index=False)

    ckpt.cleanup()
    log.info("Stage 3 v%d: %.1f%% success -> %s", version, df.best_score.mean() * 100, csv_path)
