"""Tests for Snake: 5 input modes, CSV compat, save/load."""
import os
import json
import tempfile
import pytest

from algorithmeai import Snake, floatconversion, __version__

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


def test_version():
    assert __version__ == "4.3.3"


def test_floatconversion():
    assert floatconversion("3.14") == 3.14
    assert floatconversion("bad") == 0.0
    assert floatconversion("0") == 0.0


# ---- Mode 1: CSV ----

class TestCSVMode:
    def test_csv_train(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=3, bucket=5, vocal=False)
        assert len(s.population) > 0
        assert s.target == "label"
        assert len(s.layers) == 3

    def test_csv_prediction(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=3, bucket=5, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        pred = s.get_prediction(X)
        prob = s.get_probability(X)
        assert pred in ["A", "B", "C"]
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_csv_save_load(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=3, bucket=5, vocal=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            s2 = Snake(path)
            assert s2.target == s.target
            assert len(s2.population) == len(s.population)
            assert len(s2.layers) == len(s.layers)
            X = {"color": "blue", "size": 20.0, "shape": "square"}
            assert s2.get_prediction(X) == s.get_prediction(X)
        finally:
            os.unlink(path)


# ---- Mode 2: list[dict] ----

class TestDictMode:
    def test_list_of_dicts(self):
        data = [
            {"fruit": "apple", "weight": 150, "color": "red"},
            {"fruit": "apple", "weight": 160, "color": "red"},
            {"fruit": "banana", "weight": 120, "color": "yellow"},
            {"fruit": "banana", "weight": 130, "color": "yellow"},
            {"fruit": "cherry", "weight": 5, "color": "red"},
            {"fruit": "cherry", "weight": 6, "color": "dark red"},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        assert s.target == "fruit"
        assert len(s.population) > 0

    def test_dict_with_named_target(self):
        data = [
            {"weight": 150, "color": "red", "fruit": "apple"},
            {"weight": 120, "color": "yellow", "fruit": "banana"},
            {"weight": 5, "color": "red", "fruit": "cherry"},
            {"weight": 6, "color": "dark red", "fruit": "cherry"},
        ]
        s = Snake(data, target_index="fruit", n_layers=2, bucket=3, vocal=False)
        assert s.target == "fruit"


# ---- Mode 3: list[str] self-classing ----

class TestSelfClassingMode:
    def test_self_classing(self):
        s = Snake(["A", "B", "C", "A", "B", "C", "A", "B"], n_layers=2, bucket=3, vocal=False)
        assert s.target == "target"
        assert len(s.population) == 3  # deduped: A, B, C

    def test_self_classing_numbers(self):
        s = Snake([1, 2, 3, 1, 2, 3], n_layers=2, bucket=3, vocal=False)
        assert len(s.population) == 3


# ---- Mode 4: list[tuple] uniform ----

class TestTupleMode:
    def test_uniform_tuples(self):
        data = [
            ("cat", 4, "small"),
            ("dog", 30, "medium"),
            ("cat", 5, "small"),
            ("dog", 25, "large"),
            ("bird", 1, "tiny"),
            ("bird", 2, "tiny"),
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        assert s.target == "target"
        assert len(s.population) > 0

    def test_variable_length_tuples(self):
        data = [
            ["X", 1, 2, 3],
            ["Y", 4, 5],
            ["X", 6, 7, 8],
            ["Y", 9],
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        assert len(s.header) == 4  # target + 3 features (padded)


# ---- Mode 5: DataFrame duck-typed ----

class TestDataFrameMode:
    def test_duck_dataframe(self):
        class FakeDF:
            def __init__(self, records):
                self._records = records
            def to_dict(self, orient):
                return self._records

        df = FakeDF([
            {"species": "cat", "weight": 4},
            {"species": "dog", "weight": 30},
            {"species": "cat", "weight": 5},
            {"species": "dog", "weight": 25},
        ])
        s = Snake(df, n_layers=2, bucket=3, vocal=False)
        assert s.target == "species"


# ---- Backwards compat: flat JSON ----

# ---- Mode 6: Complex JSON targets (dict/list) ----

class TestComplexJsonTargets:
    def test_dict_targets(self):
        """Train with dict targets, check datatype is J."""
        data = [
            {"label": {"color": "red", "size": "big"}, "feature1": "a", "feature2": 1},
            {"label": {"color": "red", "size": "big"}, "feature1": "b", "feature2": 2},
            {"label": {"color": "blue", "size": "small"}, "feature1": "c", "feature2": 3},
            {"label": {"color": "blue", "size": "small"}, "feature1": "d", "feature2": 4},
            {"label": {"color": "green", "size": "medium"}, "feature1": "e", "feature2": 5},
            {"label": {"color": "green", "size": "medium"}, "feature1": "f", "feature2": 6},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        assert s.datatypes[0] == "J"
        assert len(s.population) > 0

    def test_dict_target_prediction(self):
        """Prediction should return the original dict, not a string."""
        data = [
            {"label": {"type": "A"}, "x": "hello", "y": 10},
            {"label": {"type": "A"}, "x": "world", "y": 20},
            {"label": {"type": "B"}, "x": "foo", "y": 30},
            {"label": {"type": "B"}, "x": "bar", "y": 40},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        pred = s.get_prediction({"x": "hello", "y": 10})
        assert isinstance(pred, dict)
        assert "type" in pred

    def test_dict_target_probability_sums_to_one(self):
        """Probability values should sum to 1.0 even with dict targets."""
        data = [
            {"label": {"v": 1}, "f": "a"},
            {"label": {"v": 1}, "f": "b"},
            {"label": {"v": 2}, "f": "c"},
            {"label": {"v": 2}, "f": "d"},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        prob = s.get_probability({"f": "a"})
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_dict_target_save_load(self):
        """JSON round-trip should preserve complex targets."""
        data = [
            {"label": {"color": "red"}, "feat": "x"},
            {"label": {"color": "red"}, "feat": "y"},
            {"label": {"color": "blue"}, "feat": "z"},
            {"label": {"color": "blue"}, "feat": "w"},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            s2 = Snake(path)
            assert s2.datatypes[0] == "J"
            pred = s2.get_prediction({"feat": "x"})
            assert isinstance(pred, dict)
        finally:
            os.unlink(path)

    def test_list_targets(self):
        """Train with list targets."""
        data = [
            {"label": [1, 2], "f": "a"},
            {"label": [1, 2], "f": "b"},
            {"label": [3, 4], "f": "c"},
            {"label": [3, 4], "f": "d"},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        assert s.datatypes[0] == "J"
        pred = s.get_prediction({"f": "a"})
        assert isinstance(pred, list)


# ---- Backwards compat: flat JSON ----

# ---- get_augmented ----

class TestGetAugmented:
    def test_has_all_keys(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        Y = s.get_augmented(X)
        assert "Lookalikes" in Y
        assert "Probability" in Y
        assert "Prediction" in Y
        assert "Audit" in Y
        assert Y["Prediction"] in ["A", "B", "C"]

    def test_does_not_mutate_input(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        X_copy = dict(X)
        s.get_augmented(X)
        assert X == X_copy

    def test_probability_sums_to_one(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        X = {"color": "blue", "size": 20.0, "shape": "square"}
        Y = s.get_augmented(X)
        assert abs(sum(Y["Probability"].values()) - 1.0) < 1e-9


# ---- Excluded features (CSV only) ----

class TestExcludedFeatures:
    def test_csv_excluded_column(self):
        """Excluding column 2 (shape) should reduce feature count."""
        s_all = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        s_excl = Snake(SAMPLE_CSV, target_index=3, excluded_features_index=(2,), n_layers=1, bucket=5, vocal=False)
        assert len(s_excl.header) == len(s_all.header) - 1
        assert "shape" not in s_excl.header


# ---- Deduplication ----

class TestDeduplication:
    def test_duplicate_features_different_targets_drops(self):
        """Duplicate features with different targets should drop the second row."""
        data = [
            {"label": "A", "x": "same"},
            {"label": "B", "x": "same"},
            {"label": "C", "x": "different"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        # "same" feature appears twice with different targets -> one dropped
        assert len(s.population) == 2


# ---- CSV parsing ----

class TestMakeBlocFromLine:
    def test_simple_line(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        result = s.make_bloc_from_line("a,b,c\n")
        assert result == ["a", "b", "c"]

    def test_quoted_line(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        result = s.make_bloc_from_line('"hello, world",b,c\n')
        assert result == ["hello, world", "b", "c"]


# ---- Vocal modes ----

class TestVocalModes:
    def test_vocal_false_no_stdout(self, capsys):
        Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_vocal_true_has_stdout(self, capsys):
        Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "v4.3.3" in captured.out


# ---- Backwards compat: flat JSON ----

class TestBackwardsCompat:
    def test_load_flat_json(self):
        """Simulate a v0.1 flat JSON model."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        # Build a flat model manually
        flat = {
            "population": s.population,
            "header": s.header,
            "target": s.target,
            "targets": s.targets,
            "datatypes": s.datatypes,
            "clauses": s.layers[0][0]["clauses"] if s.layers else [],
            "lookalikes": s.layers[0][0]["lookalikes"] if s.layers else {},
            "n_layers": 2,
            "vocal": False,
            "log": ""
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(flat, f)
            path = f.name
        try:
            s2 = Snake(path)
            assert len(s2.layers) == 1
            assert s2.layers[0][0]["condition"] is None
        finally:
            os.unlink(path)
