"""Tests for error handling, type detection, extreme parameters, prediction edge cases."""
import os
import json
import tempfile
import pytest

from algorithmeai import Snake

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Snake([])

    def test_empty_dataframe_raises(self):
        class FakeDF:
            def __init__(self, records):
                self._records = records
            def to_dict(self, orient):
                return self._records
        with pytest.raises(ValueError, match="Empty DataFrame"):
            Snake(FakeDF([]))

    def test_file_not_found_csv(self):
        with pytest.raises(FileNotFoundError):
            Snake("nonexistent.csv")

    def test_file_not_found_json(self):
        with pytest.raises(FileNotFoundError):
            Snake("nonexistent.json")

    def test_malformed_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{bad json")
            path = f.name
        try:
            with pytest.raises(json.JSONDecodeError):
                Snake(path)
        finally:
            os.unlink(path)

    def test_target_index_string_not_found(self):
        data = [
            {"a": "X", "b": 1},
            {"a": "Y", "b": 2},
        ]
        with pytest.raises(ValueError):
            Snake(data, target_index="nonexistent", n_layers=1, bucket=3, vocal=False)

    def test_single_class_training(self):
        """Single class trains and predicts that class."""
        data = [
            {"label": "only", "x": "a"},
            {"label": "only", "x": "b"},
            {"label": "only", "x": "c"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        pred = s.get_prediction({"x": "a"})
        assert pred == "only"


# ---------------------------------------------------------------------------
# TestTypeDetection
# ---------------------------------------------------------------------------

class TestTypeDetection:
    def test_binary_01(self):
        data = [
            {"label": "0", "x": "a"},
            {"label": "1", "x": "b"},
            {"label": "0", "x": "c"},
            {"label": "1", "x": "d"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        assert s.datatypes[0] == "B"
        assert all(isinstance(t, int) for t in s.targets)

    def test_binary_true_false(self):
        data = [
            {"label": "True", "x": "a"},
            {"label": "False", "x": "b"},
            {"label": "True", "x": "c"},
            {"label": "False", "x": "d"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        assert s.datatypes[0] == "B"

    def test_integer_target(self):
        data = [
            {"label": "1", "x": "a"},
            {"label": "2", "x": "b"},
            {"label": "3", "x": "c"},
            {"label": "1", "x": "d"},
        ]
        # Targets are digits only, but multiple values -> integer multiclass
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        assert s.datatypes[0] == "I"

    def test_numeric_target(self):
        data = [
            {"label": "1.5", "x": "a"},
            {"label": "2.5", "x": "b"},
            {"label": "1.5", "x": "c"},
            {"label": "2.5", "x": "d"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        assert s.datatypes[0] == "N"

    def test_text_target(self):
        data = [
            {"label": "cat", "x": "a"},
            {"label": "dog", "x": "b"},
            {"label": "cat", "x": "c"},
            {"label": "dog", "x": "d"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        assert s.datatypes[0] == "T"

    def test_numeric_feature_detection(self):
        data = [
            {"label": "A", "num_feat": 1.0, "txt_feat": "hello"},
            {"label": "B", "num_feat": 2.0, "txt_feat": "world"},
            {"label": "A", "num_feat": 3.0, "txt_feat": "foo"},
            {"label": "B", "num_feat": 4.0, "txt_feat": "bar"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        # header = ["label", "num_feat", "txt_feat"]
        # datatypes[1] = num_feat, datatypes[2] = txt_feat
        assert s.datatypes[1] == "N"
        assert s.datatypes[2] == "T"


# ---------------------------------------------------------------------------
# TestExtremeParameters
# ---------------------------------------------------------------------------

class TestExtremeParameters:
    def test_single_layer(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        assert len(s.layers) == 1
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        pred = s.get_prediction(X)
        assert pred in ["A", "B", "C"]

    def test_noise_zero(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, noise=0.0, vocal=False)
        assert len(s.layers) == 2
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        prob = s.get_probability(X)
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_noise_one(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, noise=1.0, vocal=False)
        assert len(s.layers) == 2
        X = {"color": "blue", "size": 20.0, "shape": "square"}
        pred = s.get_prediction(X)
        assert pred in ["A", "B", "C"]


# ---------------------------------------------------------------------------
# TestPredictionEdgeCases
# ---------------------------------------------------------------------------

class TestPredictionEdgeCases:
    def test_empty_dict_prediction(self):
        """{} should produce a valid prediction with probability summing to 1."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        pred = s.get_prediction({})
        assert pred in ["A", "B", "C"]
        prob = s.get_probability({})
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_partial_features(self):
        """Only some features still predicts."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        pred = s.get_prediction({"color": "red"})
        assert pred in ["A", "B", "C"]

    def test_unknown_features_ignored(self):
        """Extra keys don't break prediction."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle", "extra_key": "whatever"}
        pred = s.get_prediction(X)
        assert pred in ["A", "B", "C"]
        prob = s.get_probability(X)
        assert abs(sum(prob.values()) - 1.0) < 1e-9
