"""Stage 4: Aggregate scores across versions into summary CSV.

Fixes over original:
- Model type from config (not hardcoded to llama_3.1_70b)
- Handles missing versions gracefully
"""
import logging
import os

import numpy as np
import pandas as pd

from ..utils import read_concept_list

log = logging.getLogger(__name__)


def run_stage4(cfg) -> pd.DataFrame:
    """Aggregate scores from all versions."""
    data_dir = cfg.paths.data_dir
    csv_dir = os.path.join(data_dir, "csvs")

    use_soft_labels = cfg.training.label_type == "soft"
    suffix = "_softlabels" if use_soft_labels else ""

    concept_file = os.path.join(data_dir, cfg.data.concept_file)
    n_concepts = len(read_concept_list(concept_file, lowercase=cfg.data.lowercase))

    versions = list(cfg.generation.versions)
    scores = []
    all_found = True

    for version in versions:
        csv_path = os.path.join(
            csv_dir,
            f"{cfg.steering.method}_{cfg.data.concept_class}_tokenidx{cfg.training.rep_token}"
            f"_block{suffix}_gpt4o_outputs_500_concepts_{cfg.model.name}_{version}.csv",
        )
        try:
            df = pd.read_csv(csv_path)
            scores.extend(df["best_score"].astype(float).values)
        except FileNotFoundError:
            log.warning("Missing CSV: %s", csv_path)
            scores.extend([float("nan")] * n_concepts)
            all_found = False

    mean_score = np.nanmean(scores) if scores else 0.0
    log.info("Stage 4: %s/%s/%s -> %.1f%% (%d versions, %d concepts)",
             cfg.steering.method, cfg.data.concept_class, cfg.training.rep_token,
             mean_score * 100, len(versions), n_concepts)

    summary_path = os.path.join(csv_dir, f"{cfg.model.name}_{cfg.data.concept_class}_{cfg.steering.method}_summary.csv")
    summary = pd.DataFrame([{
        "model": cfg.model.name,
        "concept_class": cfg.data.concept_class,
        "method": cfg.steering.method,
        "label_type": cfg.training.label_type,
        "rep_token": cfg.training.rep_token,
        "mean_score": mean_score,
        "n_versions": len(versions),
        "all_versions_complete": all_found,
    }])
    summary.to_csv(summary_path, index=False)
    log.info("Summary -> %s", summary_path)
    return summary
