"""Tests for Meta error classifier."""
import os
import csv
import json
import tempfile
import pytest

from algorithmeai import Meta

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")

# 3-class multiclass data (from sample.csv structure)
MULTICLASS_DATA = [
    {"label": "A", "color": "red", "size": "10", "shape": "circle"},
    {"label": "A", "color": "red", "size": "12", "shape": "circle"},
    {"label": "A", "color": "red", "size": "11", "shape": "circle"},
    {"label": "A", "color": "red", "size": "15", "shape": "oval"},
    {"label": "A", "color": "red", "size": "9", "shape": "circle"},
    {"label": "B", "color": "blue", "size": "20", "shape": "square"},
    {"label": "B", "color": "blue", "size": "22", "shape": "square"},
    {"label": "B", "color": "blue", "size": "21", "shape": "square"},
    {"label": "B", "color": "blue", "size": "25", "shape": "rectangle"},
    {"label": "B", "color": "blue", "size": "19", "shape": "square"},
    {"label": "C", "color": "green", "size": "30", "shape": "triangle"},
    {"label": "C", "color": "green", "size": "32", "shape": "triangle"},
    {"label": "C", "color": "green", "size": "31", "shape": "triangle"},
    {"label": "C", "color": "green", "size": "35", "shape": "hexagon"},
    {"label": "C", "color": "green", "size": "29", "shape": "triangle"},
]

# 2-class binary data
BINARY_DATA = [
    {"survived": "1", "pclass": "1", "sex": "female", "age": "29"},
    {"survived": "1", "pclass": "1", "sex": "female", "age": "35"},
    {"survived": "1", "pclass": "2", "sex": "female", "age": "25"},
    {"survived": "1", "pclass": "1", "sex": "female", "age": "40"},
    {"survived": "0", "pclass": "3", "sex": "male", "age": "22"},
    {"survived": "0", "pclass": "3", "sex": "male", "age": "30"},
    {"survived": "0", "pclass": "3", "sex": "male", "age": "28"},
    {"survived": "0", "pclass": "3", "sex": "male", "age": "19"},
    {"survived": "0", "pclass": "2", "sex": "male", "age": "45"},
    {"survived": "0", "pclass": "3", "sex": "male", "age": "50"},
]

# Shared fast kwargs
FAST = dict(n_layers=1, bucket=5, n_splits=3, n_runs=1, error_layers=1, error_bucket=5)


class TestMulticlassLabels:
    def test_labels_are_rank_based(self):
        m = Meta(MULTICLASS_DATA, target_index="label", **FAST)
        valid = {"R1", "R2", "R3", "W", "NS"}
        for lbl in m.labels:
            assert lbl in valid, f"unexpected label {lbl}"

    def test_rank_capped_by_n_classes(self):
        """3-class problem should never produce R4 or R5."""
        m = Meta(MULTICLASS_DATA, target_index="label", **FAST)
        for lbl in m.labels:
            assert lbl not in ("R4", "R5"), f"rank {lbl} exceeds n_classes=3"


class TestBinaryLabels:
    def test_labels_are_confusion_matrix(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        valid = {"TP", "TN", "FP", "FN", "NS"}
        for lbl in m.labels:
            assert lbl in valid, f"unexpected label {lbl}"

    def test_is_binary_detection(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        assert m.is_binary is True

    def test_multiclass_not_binary(self):
        m = Meta(MULTICLASS_DATA, target_index="label", **FAST)
        assert m.is_binary is False


class TestCSVInput:
    def test_csv_input_works(self):
        m = Meta(SAMPLE_CSV, target_index=3, **FAST)
        assert len(m.population) == 15
        assert m.target == "label"
        assert len(m.labels) == 15


class TestToList:
    def test_adds_error_type_column(self):
        m = Meta(MULTICLASS_DATA, target_index="label", **FAST)
        rows = m.to_list()
        assert len(rows) == len(m.population)
        for row in rows:
            assert "error_type" in row


class TestToCSV:
    def test_writes_valid_csv(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            m.to_csv(path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == len(m.population)
            assert "error_type" in rows[0]
        finally:
            os.unlink(path)


class TestSaveLoad:
    def test_json_round_trip(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta.json")
            m.to_json(path)

            m2 = Meta(path)
            assert m2.labels == m.labels
            assert m2.target == m.target
            assert m2.is_binary == m.is_binary
            assert m2.positive_class == m.positive_class
            assert m2.agreement_rate == m.agreement_rate
            assert len(m2.population) == len(m.population)

            # Prediction still works after load
            X = {"pclass": "1", "sex": "female", "age": "29"}
            pred = m2.get_prediction(X)
            assert isinstance(pred, str)


class TestPrediction:
    def test_get_prediction_returns_valid_label(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        X = {"pclass": "3", "sex": "male", "age": "22"}
        pred = m.get_prediction(X)
        valid = {"TP", "TN", "FP", "FN", "NS"}
        assert pred in valid

    def test_get_probability_sums_to_one(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        X = {"pclass": "1", "sex": "female", "age": "35"}
        prob = m.get_probability(X)
        assert abs(sum(prob.values()) - 1.0) < 1e-9


class TestLabelCounts:
    def test_counts_sum_to_population(self):
        m = Meta(MULTICLASS_DATA, target_index="label", **FAST)
        assert sum(m.label_counts.values()) == len(m.population)


class TestAgreementRate:
    def test_in_range(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        assert 0.0 <= m.agreement_rate <= 1.0

    def test_single_run_no_ns(self):
        """With n_runs=1, there's no agreement filtering, so NS only from empty splits."""
        m = Meta(BINARY_DATA, target_index="survived", n_runs=1,
                 n_layers=1, bucket=5, n_splits=3, error_layers=1, error_bucket=5)
        # All samples should have been seen at least once with 3 splits at 80/20
        # so agreement_rate should be high (no NS from disagreement, only from missing)
        assert m.agreement_rate > 0.0


class TestJSONPathError:
    def test_non_meta_json_raises(self):
        """Loading a plain Snake JSON via Meta should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": "5.0.0", "not_meta": True}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="not a valid Meta JSON"):
                Meta(path)
        finally:
            os.unlink(path)


class TestSummary:
    def test_summary_returns_string(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        s = m.summary()
        assert isinstance(s, str)
        assert "binary" in s
        assert "survived" in s
        assert "Label distribution" in s


class TestRepr:
    def test_repr_format(self):
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        r = repr(m)
        assert "Meta(" in r
        assert "binary" in r

    def test_repr_multiclass(self):
        m = Meta(MULTICLASS_DATA, target_index="label", **FAST)
        r = repr(m)
        assert "multiclass" in r


class TestErrorModelNoTargetLeak:
    def test_target_not_in_error_model_features(self):
        """Error model must not use the original target as a feature (data leak)."""
        m = Meta(BINARY_DATA, target_index="survived", **FAST)
        em_features = [col for col, dt in zip(m.error_model.header, m.error_model.datatypes)
                       if col != m.error_model.target]
        assert m.target not in em_features, (
            f"Original target '{m.target}' leaked into error model features"
        )

    def test_target_not_in_error_model_multiclass(self):
        """Same leak check for multiclass."""
        m = Meta(MULTICLASS_DATA, target_index="label", **FAST)
        em_features = [col for col in m.error_model.header
                       if col != m.error_model.target]
        assert m.target not in em_features
