"""Optional Weights & Biases experiment tracking. No-ops when disabled."""
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class WandbTracker:
    """W&B integration that gracefully no-ops when disabled.

    Usage:
        tracker = WandbTracker(cfg)
        tracker.log_concept("bacteria", "stage1", {"r_score": 0.98, "time": 14.2})
        tracker.log_stage_summary("stage1", {"mean_r": 0.95, "total_time": 1200})
        tracker.finish()
    """

    def __init__(self, cfg):
        self.enabled = cfg.wandb.enabled
        self._run = None

        if not self.enabled:
            return

        try:
            import wandb
            from omegaconf import OmegaConf

            self._run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity if cfg.wandb.entity else None,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=list(cfg.wandb.tags) if cfg.wandb.tags else [],
                name=f"{cfg.model.name}_{cfg.data.concept_class}_{cfg.steering.method}",
            )
            log.info("W&B initialized: %s", self._run.url)
        except ImportError:
            log.warning("wandb not installed, disabling tracking")
            self.enabled = False
        except Exception as e:
            log.warning("W&B init failed: %s, disabling tracking", e)
            self.enabled = False

    def log_concept(self, concept: str, stage: str, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        import wandb
        wandb.log({f"{stage}/{k}": v for k, v in metrics.items()})

    def log_stage_summary(self, stage: str, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        import wandb
        wandb.log({f"{stage}_summary/{k}": v for k, v in metrics.items()})

    def log_timing(self, stage: str, concept: str, elapsed: float) -> None:
        if not self.enabled:
            return
        import wandb
        wandb.log({f"timing/{stage}": elapsed})

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            import wandb
            wandb.finish()
