"""Tests for make_validation / pruning."""
import os
import json
import tempfile
import pytest

from algorithmeai import Snake

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


def _train_model(n_layers=5):
    return Snake(SAMPLE_CSV, target_index=3, n_layers=n_layers, bucket=5, vocal=False)


def _val_data(s):
    """Build validation data from the population (include target key)."""
    return [dict(dp) for dp in s.population]


class TestMakeValidation:
    def test_pruning_reduces_layers(self):
        s = _train_model(n_layers=5)
        assert len(s.layers) == 5
        s.make_validation(_val_data(s), pruning_coef=0.5)
        # 5 * 0.5 = 2.5 -> int(2.5) = 2, max(1, 2) = 2
        assert len(s.layers) in (2, 3)
        assert s.n_layers == len(s.layers)

    def test_pruning_coef_1_keeps_all(self):
        s = _train_model(n_layers=5)
        s.make_validation(_val_data(s), pruning_coef=1.0)
        assert len(s.layers) == 5
        assert s.n_layers == 5

    def test_pruning_coef_small_keeps_at_least_one(self):
        s = _train_model(n_layers=5)
        s.make_validation(_val_data(s), pruning_coef=0.01)
        # max(1, int(5 * 0.01)) = max(1, 0) = 1
        assert len(s.layers) == 1
        assert s.n_layers == 1

    def test_pruning_updates_n_layers(self):
        s = _train_model(n_layers=4)
        s.make_validation(_val_data(s), pruning_coef=0.5)
        assert s.n_layers == len(s.layers)

    def test_validation_skips_missing_target(self):
        """Dicts without target key should be silently skipped."""
        s = _train_model(n_layers=3)
        # Mix valid and invalid entries
        val = _val_data(s) + [{"color": "red", "size": 10.0, "shape": "circle"}]
        # Should not raise
        s.make_validation(val, pruning_coef=0.5)
        assert s.n_layers >= 1

    def test_save_after_pruning(self):
        """JSON roundtrip works after pruning."""
        s = _train_model(n_layers=4)
        s.make_validation(_val_data(s), pruning_coef=0.5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            s2 = Snake(path)
            assert len(s2.layers) == len(s.layers)
            assert s2.n_layers == s.n_layers
            X = {"color": "red", "size": 10.0, "shape": "circle"}
            pred = s2.get_prediction(X)
            assert pred in ["A", "B", "C"]
        finally:
            os.unlink(path)

    def test_pruning_accuracy_reasonable(self):
        """Prediction still works after pruning."""
        s = _train_model(n_layers=5)
        val = _val_data(s)
        s.make_validation(val, pruning_coef=0.5)
        correct = 0
        for dp in s.population:
            features = {k: v for k, v in dp.items() if k != s.target}
            pred = s.get_prediction(features)
            if pred == dp[s.target]:
                correct += 1
        # After pruning, should still get at least some right
        assert correct > 0
