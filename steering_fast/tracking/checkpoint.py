"""Checkpoint/resume logic for all pipeline stages.

Saves progress after each concept so a crashed job can resume without recomputation.
Uses a JSON manifest (lightweight, human-readable) plus pickle data.
"""
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

log = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint state for a single pipeline stage.

    Usage:
        ckpt = CheckpointManager("checkpoints/", "stage1", "abc123")
        completed, results = ckpt.load()
        for concept in concepts:
            if concept in completed:
                continue
            # ... compute ...
            results[concept] = output
            completed.add(concept)
            ckpt.save(completed, results)
    """

    def __init__(self, checkpoint_dir: str, stage: str, cfg_hash: str):
        self.dir = Path(checkpoint_dir)
        self.stage = stage
        self.cfg_hash = cfg_hash
        self.meta_path = self.dir / f"{stage}_{cfg_hash}.json"
        self.data_path = self.dir / f"{stage}_{cfg_hash}.pkl"

    def load(self) -> tuple[Set[str], Dict[str, Any]]:
        """Load existing checkpoint if it matches current config.

        Returns (completed_concepts, partial_results). Both empty if no checkpoint.
        """
        if not self.meta_path.exists():
            return set(), {}

        try:
            meta = json.loads(self.meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt checkpoint metadata at %s, starting fresh", self.meta_path)
            return set(), {}

        if meta.get("config_hash") != self.cfg_hash:
            log.info("Config changed (old=%s, new=%s), starting fresh", meta.get("config_hash"), self.cfg_hash)
            return set(), {}

        try:
            data = pickle.loads(self.data_path.read_bytes())
        except (pickle.UnpicklingError, EOFError, OSError):
            log.warning("Corrupt checkpoint data at %s, starting fresh", self.data_path)
            return set(), {}

        completed = set(meta.get("completed_concepts", []))
        log.info("Resumed from checkpoint: %d concepts completed for %s", len(completed), self.stage)
        return completed, data

    def save(self, completed: Set[str], results: Dict[str, Any]) -> None:
        """Save checkpoint after processing a concept."""
        self.dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "stage": self.stage,
            "config_hash": self.cfg_hash,
            "completed_concepts": sorted(completed),
            "n_completed": len(completed),
            "timestamp": time.time(),
        }

        # Write data first (larger), then metadata (atomic signal)
        self.data_path.write_bytes(pickle.dumps(results))
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def cleanup(self) -> None:
        """Remove checkpoint files after stage completes successfully."""
        for p in (self.meta_path, self.data_path):
            if p.exists():
                p.unlink()
        log.info("Cleaned up checkpoint for %s", self.stage)
