"""Tests for Snake: 5 input modes, CSV compat, save/load."""
import os
import json
import tempfile
import pytest

from algorithmeai import Snake, floatconversion, __version__

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


def test_version():
    assert __version__ == "4.2.0"


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
