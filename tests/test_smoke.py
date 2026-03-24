"""Smoke test: run 3 concepts and compare with original pipeline outputs.

Usage: pytest tests/test_smoke.py -v
Requires: Original pipeline outputs in ../attention_guided_steering/data/
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

ORIGINAL_DATA = Path(__file__).parent.parent.parent / "attention_guided_steering" / "data"
FAST_DATA = ORIGINAL_DATA  # Same data dir by default


def get_test_concepts(concept_class: str = "fears", n: int = 3):
    """Get first N concepts for smoke testing."""
    concept_file = ORIGINAL_DATA / "concepts" / f"{concept_class}.txt"
    if not concept_file.exists():
        pytest.skip(f"No concept file: {concept_file}")
    with open(concept_file) as f:
        concepts = sorted(set(line.strip().lower() for line in f if line.strip()))
    return concepts[:n]


class TestAttentionEquivalence:
    """Compare attention arrays from stage 0."""

    @pytest.mark.parametrize("concept", get_test_concepts())
    def test_attention_file_exists(self, concept):
        path = ORIGINAL_DATA / "attention_to_prompt" / f"attentions_meanhead_llama_3.1_8b_{concept}_paired_statements.npy"
        assert path.exists(), f"Missing: {path}"

    @pytest.mark.parametrize("concept", get_test_concepts())
    def test_attention_shape(self, concept):
        path = ORIGINAL_DATA / "attention_to_prompt" / f"attentions_meanhead_llama_3.1_8b_{concept}_paired_statements.npy"
        if not path.exists():
            pytest.skip("No attention file")
        data = np.load(path)
        assert data.ndim == 3, f"Expected 3D array, got {data.ndim}D"
        assert data.shape[1] == 32, f"Expected 32 layers, got {data.shape[1]}"


class TestDirectionEquivalence:
    """Compare direction vectors from stage 1."""

    @pytest.mark.parametrize("concept", get_test_concepts())
    def test_direction_file_exists(self, concept):
        path = ORIGINAL_DATA / "directions" / f"rfm_{concept}_tokenidx_max_attn_per_layer_block_softlabels_llama_3.1_8b.pkl"
        assert path.exists(), f"Missing: {path}"

    @pytest.mark.parametrize("concept", get_test_concepts())
    def test_direction_layers(self, concept):
        path = ORIGINAL_DATA / "directions" / f"rfm_{concept}_tokenidx_max_attn_per_layer_block_softlabels_llama_3.1_8b.pkl"
        if not path.exists():
            pytest.skip("No direction file")
        with open(path, "rb") as f:
            directions = pickle.load(f)
        assert isinstance(directions, dict), "Expected dict"
        assert len(directions) >= 30, f"Expected ~31 layers, got {len(directions)}"
        for layer_idx, vec in directions.items():
            assert vec.shape[-1] == 4096, f"Layer {layer_idx}: expected dim 4096, got {vec.shape[-1]}"


class TestScoringEquivalence:
    """Compare evaluation CSVs from stage 3."""

    def test_fears_v1_exists(self):
        path = ORIGINAL_DATA / "csvs" / "rfm_fears_tokenidxmax_attn_per_layer_block_softlabels_gpt4o_outputs_500_concepts_llama_3.1_8b_1.csv"
        if not path.exists():
            pytest.skip("No evaluation CSV yet")
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) > 0, "Empty CSV"
        success_rate = df.best_score.mean()
        assert success_rate > 0.5, f"Success rate too low: {success_rate:.1%}"
