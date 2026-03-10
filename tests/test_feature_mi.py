"""Tests for MI-weighted feature selection and lookahead literal selection."""
import json
import os
import random
import tempfile

import pytest
from algorithmeai import Snake


SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")


# ---- Test data ----

# Data where feature 'signal' is perfectly predictive, 'noise' is random
def _make_signal_noise_data(n=100, n_noise=10):
    data = []
    for i in range(n):
        label = "A" if i % 2 == 0 else "B"
        row = {"label": label, "signal": f"sig_{label}_{i % 5}"}
        for j in range(n_noise):
            row[f"noise_{j}"] = f"val_{random.randint(0, 100)}"
        data.append(row)
    return data


NUMERIC_SIGNAL_DATA = [
    {"label": "pos", "signal": 10.0, "noise1": 0.5, "noise2": 0.3},
    {"label": "pos", "signal": 12.0, "noise1": 0.8, "noise2": 0.1},
    {"label": "pos", "signal": 11.0, "noise1": 0.2, "noise2": 0.9},
    {"label": "pos", "signal": 13.0, "noise1": 0.6, "noise2": 0.7},
    {"label": "neg", "signal": 1.0, "noise1": 0.4, "noise2": 0.5},
    {"label": "neg", "signal": 2.0, "noise1": 0.7, "noise2": 0.2},
    {"label": "neg", "signal": 0.5, "noise1": 0.1, "noise2": 0.8},
    {"label": "neg", "signal": 1.5, "noise1": 0.9, "noise2": 0.6},
]


class TestFeatureMI:
    """Tests for _precompute_feature_mi."""

    def test_mi_precomputed_on_init(self):
        """MI weights are computed during training."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        assert hasattr(s, '_feature_mi')
        assert len(s._feature_mi) > 0

    def test_mi_signal_higher_than_noise(self):
        """Signal feature should have higher MI than noise features."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        # 'signal' is header index 1 (first feature after target)
        signal_idx = s.header.index("signal")
        noise1_idx = s.header.index("noise1")
        noise2_idx = s.header.index("noise2")
        assert s._feature_mi[signal_idx] > s._feature_mi[noise1_idx]
        assert s._feature_mi[signal_idx] > s._feature_mi[noise2_idx]

    def test_mi_text_features(self):
        """MI works for text features too."""
        random.seed(42)
        data = _make_signal_noise_data(50, 3)
        s = Snake(data, n_layers=1, bucket=250)
        signal_idx = s.header.index("signal")
        # Signal should have highest MI
        max_mi_idx = max(s._feature_mi, key=s._feature_mi.get)
        assert max_mi_idx == signal_idx

    def test_mi_all_same_values(self):
        """Feature with identical values across all rows should have MI=0."""
        data = [
            {"label": "A", "same": "x", "diff": "a"},
            {"label": "A", "same": "x", "diff": "a"},
            {"label": "B", "same": "x", "diff": "b"},
            {"label": "B", "same": "x", "diff": "b"},
        ]
        s = Snake(data, n_layers=1, bucket=250)
        same_idx = s.header.index("same")
        diff_idx = s.header.index("diff")
        assert s._feature_mi[same_idx] == 0.0
        assert s._feature_mi[diff_idx] > 0.0

    def test_mi_with_many_text_values(self):
        """MI capping at 200 unique values works without error."""
        data = [{"label": str(i % 3), "feat": f"val_{i}"} for i in range(300)]
        s = Snake(data, n_layers=1, bucket=250)
        assert len(s._feature_mi) > 0

    def test_mi_nonnegative(self):
        """All MI values should be >= 0."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        for mi in s._feature_mi.values():
            assert mi >= 0.0

    def test_mi_small_population(self):
        """MI computation handles very small datasets."""
        data = [
            {"label": "A", "x": 1.0},
            {"label": "B", "x": 2.0},
        ]
        s = Snake(data, n_layers=1, bucket=250)
        assert len(s._feature_mi) == 1


class TestWeightedFeatureChoice:
    """Tests for _weighted_feature_choice."""

    def test_weighted_choice_with_empty_mi(self):
        """Falls back to uniform when MI is empty."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        s._feature_mi = {}
        # Should not crash
        candidates = [1, 2, 3]
        result = s._weighted_feature_choice(candidates)
        assert result in candidates

    def test_weighted_choice_single_candidate(self):
        """Single candidate always returned."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        assert s._weighted_feature_choice([2]) == 2

    def test_weighted_choice_favors_high_mi(self):
        """With large MI differences, weighted choice strongly favors high-MI feature."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        # Set extreme MI weights
        s._feature_mi = {1: 100.0, 2: 0.001, 3: 0.001}
        counts = {1: 0, 2: 0, 3: 0}
        for _ in range(1000):
            idx = s._weighted_feature_choice([1, 2, 3])
            counts[idx] += 1
        # Feature 1 should get the vast majority
        assert counts[1] > 900


class TestLookahead:
    """Tests for _oppose_lookahead and the lookahead parameter."""

    def test_lookahead_default(self):
        """Default lookahead is 5."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        assert s.lookahead == 5

    def test_lookahead_1_equivalent_to_no_lookahead(self):
        """lookahead=1 should produce a valid model."""
        random.seed(42)
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=2, bucket=250, lookahead=1)
        pred = s.get_prediction({"signal": 11.0, "noise1": 0.5, "noise2": 0.5})
        assert pred in ("pos", "neg")

    def test_lookahead_custom_value(self):
        """Custom lookahead value is stored."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250, lookahead=10)
        assert s.lookahead == 10

    def test_lookahead_persisted_in_json(self):
        """Lookahead is saved/loaded via JSON."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250, lookahead=7)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            s2 = Snake(path)
            assert s2.lookahead == 7
        finally:
            os.unlink(path)

    def test_lookahead_backwards_compat(self):
        """Old models without lookahead in config default to 5."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            # Remove lookahead from config
            with open(path) as f:
                data = json.load(f)
            del data["config"]["lookahead"]
            with open(path, "w") as f:
                json.dump(data, f)
            s2 = Snake(path)
            assert s2.lookahead == 5
        finally:
            os.unlink(path)

    def test_oppose_lookahead_returns_literal(self):
        """_oppose_lookahead returns a valid literal or None."""
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250)
        Ts = [s.population[i] for i in range(len(s.population)) if s.targets[i] != "pos"]
        F = s.population[0]
        lit = s._oppose_lookahead(Ts, F)
        if lit is not None:
            assert isinstance(lit, list)
            assert len(lit) == 4  # [index, value, negat, type]


class TestMIWithProfiles:
    """MI weights work with all oppose profiles."""

    def test_mi_with_scientific(self):
        s = Snake(NUMERIC_SIGNAL_DATA, n_layers=1, bucket=250, oppose_profile="scientific")
        assert len(s._feature_mi) > 0

    def test_mi_with_balanced(self):
        data = [
            {"label": "A", "name": "hello", "val": 1.0},
            {"label": "A", "name": "hi", "val": 2.0},
            {"label": "B", "name": "bye", "val": 10.0},
            {"label": "B", "name": "later", "val": 11.0},
        ]
        s = Snake(data, n_layers=1, bucket=250, oppose_profile="balanced")
        assert len(s._feature_mi) > 0

    def test_mi_with_csv(self):
        """MI works in CSV flow too."""
        s = Snake(SAMPLE_CSV, n_layers=1, bucket=250)
        assert len(s._feature_mi) > 0


class TestMIWithWorkers:
    """MI weights are passed to parallel workers."""

    def test_parallel_training_with_mi(self):
        """Parallel training should work with MI weights."""
        random.seed(42)
        data = _make_signal_noise_data(30, 3)
        s = Snake(data, n_layers=3, bucket=250, workers=2)
        assert len(s.layers) == 3
        pred = s.get_prediction({"signal": "sig_A_0", "noise_0": "x", "noise_1": "y", "noise_2": "z"})
        assert pred in ("A", "B")
