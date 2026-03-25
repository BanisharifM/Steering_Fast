"""Main pipeline runner.

Two ways to use:
1. Direct: load_config() + run_pipeline()  (recommended for SLURM/HPC)
2. Hydra CLI: steering-fast model=llama_3_1_8b  (for local use)
"""
import logging
import os
from typing import List, Optional

log = logging.getLogger(__name__)


def run_pipeline(cfg, stages: Optional[List[int]] = None) -> None:
    """Run the steering pipeline."""
    from ..tracking.timer import PipelineTimer
    from ..tracking.wandb_tracker import WandbTracker
    from ..utils import set_seed

    set_seed(cfg.seed)
    if stages is None:
        stages = list(cfg.stages)

    timer = PipelineTimer(enabled=cfg.timing.enabled)
    tracker = WandbTracker(cfg)

    log.info("=" * 60)
    log.info("Pipeline: %s / %s / %s", cfg.model.name, cfg.data.concept_class, cfg.steering.method)
    log.info("Stages: %s, labels: %s, batch: %d", stages, cfg.training.label_type, cfg.training.batch_size)
    if cfg.smoke_test.enabled:
        log.info("SMOKE TEST: %d concepts", cfg.smoke_test.n_concepts)
    log.info("=" * 60)

    if 0 in stages:
        from .stage0 import run_stage0
        log.info("--- Stage 0: Attention Extraction ---")
        with timer.time_stage("stage0"):
            run_stage0(cfg, timer, tracker)

    if 1 in stages:
        from .stage1 import run_stage1
        log.info("--- Stage 1: Direction Training ---")
        with timer.time_stage("stage1"):
            run_stage1(cfg, timer, tracker)

    if 2 in stages:
        from .stage2 import run_stage2
        log.info("--- Stage 2: Steered Generation ---")
        for version in cfg.generation.versions:
            with timer.time_stage(f"stage2_v{version}"):
                run_stage2(cfg, version, timer, tracker)

    if 3 in stages:
        from .stage3 import run_stage3
        log.info("--- Stage 3: GPT-4o Evaluation ---")
        for version in cfg.generation.versions:
            with timer.time_stage(f"stage3_v{version}"):
                run_stage3(cfg, version, timer, tracker)

    if 4 in stages:
        from .stage4 import run_stage4
        log.info("--- Stage 4: Score Aggregation ---")
        run_stage4(cfg)

    if cfg.timing.enabled:
        log.info("=" * 60)
        log.info("TIMING SUMMARY")
        for stage, total in timer.summary().items():
            log.info("  %s: %.1fs", stage, total)
        log.info("=" * 60)
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        timer.to_csv(os.path.join(cfg.paths.output_dir, "timing.csv"))

    tracker.finish()


def main():
    """Hydra CLI entry point."""
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="pkg://steering_fast.conf", config_name="config", version_base=None)
    def _hydra_main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO)
        stages = list(cfg.stages) if "stages" in cfg else None
        run_pipeline(cfg, stages)

    _hydra_main()


if __name__ == "__main__":
    main()
