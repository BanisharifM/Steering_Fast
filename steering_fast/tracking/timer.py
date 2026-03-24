"""Per-concept timing utilities for performance comparison."""
import csv
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    stage: str
    concept: str
    elapsed_seconds: float


class PipelineTimer:
    """Tracks per-concept and per-stage timing.

    Usage:
        timer = PipelineTimer(enabled=True)
        with timer.time_concept("stage0", "bacteria"):
            # ... do work ...
        timer.summary()
        timer.to_csv("timing_results.csv")
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: List[TimingRecord] = []

    @contextmanager
    def time_concept(self, stage: str, concept: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.records.append(TimingRecord(stage, concept, elapsed))
        log.info("[%s] %s: %.1fs", stage, concept, elapsed)

    def time_stage(self, stage: str):
        """Context manager for timing an entire stage."""
        return self.time_concept(stage, "__total__")

    def summary(self) -> Dict[str, float]:
        """Aggregate timing by stage."""
        by_stage: Dict[str, List[float]] = defaultdict(list)
        for r in self.records:
            if r.concept != "__total__":
                by_stage[r.stage].append(r.elapsed_seconds)

        result = {}
        for stage, times in sorted(by_stage.items()):
            total = sum(times)
            avg = total / len(times) if times else 0
            result[stage] = total
            log.info(
                "  %s: %.1fs total, %.1fs/concept (%d concepts)",
                stage, total, avg, len(times),
            )
        return result

    def to_csv(self, path: str) -> None:
        """Export per-concept timings to CSV for comparison with original."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "concept", "elapsed_seconds"])
            for r in self.records:
                writer.writerow([r.stage, r.concept, f"{r.elapsed_seconds:.3f}"])
        log.info("Timing results saved to %s", path)
