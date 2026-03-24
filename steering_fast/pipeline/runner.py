"""Main pipeline runner. Hydra entry point.

Usage:
    # Full run
    steering-fast model=llama_3_1_8b data=fears steering=rfm

    # Smoke test (3 concepts, 1 version)
    steering-fast experiment=smoke_test

    # Specific stages only
    steering-fast stages=[0,1]

    # Override batch size
    steering-fast training.batch_size=32
"""
import logging
import os
import sys
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def run_pipeline(cfg: DictConfig, stages: Optional[List[int]] = None) -> None:
    """Run the steering pipeline (all stages or specific ones)."""
    from ..tracking.timer import PipelineTimer
    from ..tracking.wandb_tracker import WandbTracker
    from ..utils import set_seed

    set_seed(cfg.seed)

    # Resolve stages to run
    if stages is None:
        stages = [0, 1, 2, 3, 4]

    timer = PipelineTimer(enabled=cfg.timing.enabled)
    tracker = WandbTracker(cfg)

    log.info("=" * 60)
    log.info("Steering Pipeline: %s / %s / %s", cfg.model.name, cfg.data.concept_class, cfg.steering.method)
    log.info("Stages: %s", stages)
    log.info("Label type: %s, rep_token: %s, batch_size: %d", cfg.training.label_type, cfg.training.rep_token, cfg.training.batch_size)
    if cfg.smoke_test.enabled:
        log.info("SMOKE TEST MODE: %d concepts", cfg.smoke_test.n_concepts)
    log.info("=" * 60)

    if 0 in stages:
        from .stage0 import run_stage0
        log.info("--- Stage 0: Attention Extraction ---")
        with timer.time_stage("stage0_total"):
            run_stage0(cfg, timer, tracker)

    if 1 in stages:
        from .stage1 import run_stage1
        log.info("--- Stage 1: Direction Training ---")
        with timer.time_stage("stage1_total"):
            run_stage1(cfg, timer, tracker)

    if 2 in stages:
        from .stage2 import run_stage2
        log.info("--- Stage 2: Steered Generation ---")
        for version in cfg.generation.versions:
            log.info("  Version %d", version)
            with timer.time_stage(f"stage2_v{version}_total"):
                run_stage2(cfg, version, timer, tracker)

    if 3 in stages:
        from .stage3 import run_stage3
        log.info("--- Stage 3: GPT-4o Evaluation ---")
        for version in cfg.generation.versions:
            log.info("  Version %d", version)
            with timer.time_stage(f"stage3_v{version}_total"):
                run_stage3(cfg, version, timer, tracker)

    if 4 in stages:
        from .stage4 import run_stage4
        log.info("--- Stage 4: Score Aggregation ---")
        run_stage4(cfg)

    # Print timing summary
    if cfg.timing.enabled:
        log.info("=" * 60)
        log.info("TIMING SUMMARY")
        summary = timer.summary()
        for stage, total in summary.items():
            log.info("  %s: %.1fs", stage, total)
        log.info("=" * 60)

        # Save timing CSV
        timing_path = os.path.join(cfg.paths.output_dir, "timing.csv")
        os.makedirs(os.path.dirname(timing_path), exist_ok=True)
        timer.to_csv(timing_path)

    tracker.finish()


@hydra.main(config_path="pkg://steering_fast.conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    stages = list(cfg.stages) if "stages" in cfg else None
    run_pipeline(cfg, stages)


if __name__ == "__main__":
    main()
