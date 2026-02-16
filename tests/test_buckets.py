"""Tests for bucketed layer architecture: chain build, noise, routing, audit, confidence."""
import os
import pytest

from algorithmeai import Snake
from algorithmeai.snake import build_condition, build_bucket_chain, traverse_chain

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


class TestBuildCondition:
    def _make_snake(self):
        return Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=3, vocal=False)

    def test_condition_peels(self):
        s = self._make_snake()
        matching = list(range(len(s.population)))
        condition, selected = build_condition(
            matching, s.population, s.targets, 3,
            s.oppose, s.apply_literal
        )
        # selected should be <= 2*bucket or condition was empty
        assert len(selected) <= len(matching)

    def test_empty_on_single_class(self):
        """If all matching have same target, condition should be empty."""
        s = self._make_snake()
        # Grab only class A indices
        a_indices = [i for i in range(len(s.population)) if s.targets[i] == "A"]
        condition, selected = build_condition(
            a_indices, s.population, s.targets, 2,
            s.oppose, s.apply_literal
        )
        assert condition == []


class TestBuildBucketChain:
    def test_chain_covers_all(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=3, vocal=False)
        chain = build_bucket_chain(
            s.population, s.targets, 3,
            s.oppose, s.apply_literal, noise=0.0
        )
        # Every index should appear as core member in exactly one bucket (no noise)
        all_core = set()
        for entry in chain:
            for idx in entry["members"]:
                all_core.add(idx)
        assert all_core == set(range(len(s.population)))

    def test_else_bucket_exists(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=3, vocal=False)
        chain = build_bucket_chain(
            s.population, s.targets, 3,
            s.oppose, s.apply_literal, noise=0.0
        )
        # Last bucket should be ELSE (condition=None) or the only bucket
        last = chain[-1]
        assert last["condition"] is None or len(chain) == 1

    def test_noise_adds_members(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=3, vocal=False)
        chain_no_noise = build_bucket_chain(
            s.population, s.targets, 3,
            s.oppose, s.apply_literal, noise=0.0
        )
        chain_noise = build_bucket_chain(
            s.population, s.targets, 3,
            s.oppose, s.apply_literal, noise=0.5
        )
        # With noise, non-ELSE buckets should have more members (if chain has >1 bucket)
        if len(chain_noise) > 1:
            noisy_total = sum(len(e["members"]) for e in chain_noise if e["condition"] is not None)
            clean_total = sum(len(e["members"]) for e in chain_no_noise if e["condition"] is not None)
            assert noisy_total >= clean_total


class TestTraverseChain:
    def test_routes_to_bucket(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=3, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        for chain in s.layers:
            bucket = traverse_chain(chain, X, s.apply_literal)
            assert bucket is not None
            assert "members" in bucket
            assert "clauses" in bucket
            assert "lookalikes" in bucket


class TestConfidence:
    def test_train_confidence(self):
        """On training data, confidence should be high (ideally 1.0)."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=3, bucket=5, vocal=False)
        high_conf = 0
        for dp in s.population:
            prob = s.get_probability(dp)
            conf = max(prob.values())
            pred = s.get_prediction(dp)
            if conf >= 0.5:
                high_conf += 1
        # At least 50% of training data should have >50% confidence
        assert high_conf >= len(s.population) * 0.5

    def test_probability_sums_to_one(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=3, bucket=5, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        prob = s.get_probability(X)
        assert abs(sum(prob.values()) - 1.0) < 1e-9


class TestAudit:
    def test_audit_structure(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "LAYER 0" in audit
        assert "GLOBAL SUMMARY" in audit
        assert "PREDICTION" in audit
        assert "END AUDIT" in audit

    def test_audit_shows_buckets(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        X = {"color": "blue", "size": 20.0, "shape": "square"}
        audit = s.get_audit(X)
        assert "BUCKET" in audit
        assert "members" in audit

    def test_log_contains_banner(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)
        assert "v4.3.2" in s.log
        assert "Charles Dana" in s.log


class TestFIFODedup:
    def test_dedup_by_population_index(self):
        """Same population index should not vote twice."""
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=3, bucket=5, vocal=False)
        X = s.population[0]
        lookalikes = s.get_lookalikes(X)
        seen_indices = [triple[0] for triple in lookalikes]
        assert len(seen_indices) == len(set(seen_indices))


class TestBatchPrediction:
    def test_batch_returns_list(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        Xs = [
            {"color": "red", "size": 10.0, "shape": "circle"},
            {"color": "blue", "size": 20.0, "shape": "square"},
        ]
        results = s.get_batch_prediction(Xs)
        assert len(results) == 2
        for r in results:
            assert "prediction" in r
            assert "probability" in r
            assert "confidence" in r
            assert r["prediction"] in ["A", "B", "C"]
            assert abs(sum(r["probability"].values()) - 1.0) < 1e-9
            assert 0.0 <= r["confidence"] <= 1.0

    def test_batch_matches_individual(self):
        s = Snake(SAMPLE_CSV, target_index=3, n_layers=2, bucket=5, vocal=False)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        batch_result = s.get_batch_prediction([X])[0]
        individual_pred = s.get_prediction(X)
        assert batch_result["prediction"] == individual_pred


class TestApplyLiteralTypes:
    """Tests for TPS/TSS literal types and edge cases."""

    def _make_snake(self):
        return Snake(SAMPLE_CSV, target_index=3, n_layers=1, bucket=5, vocal=False)

    def test_apply_literal_tps(self):
        """TPS literal should split on comma."""
        s = self._make_snake()
        # TPS literal: [index, threshold, negat, "TPS"]
        # header[1] is "color" in sample.csv
        X = {s.header[1]: "a,b,c"}
        # 3 parts, threshold 2.0, negat=True means >= 2.0
        literal = [1, 2.0, True, "TPS"]
        assert s.apply_literal(X, literal) is True
        # negat=False means > 2.0 (i.e. 2.0 > 3 is False)
        literal_neg = [1, 2.0, False, "TPS"]
        assert s.apply_literal(X, literal_neg) is False

    def test_apply_literal_tss(self):
        """TSS literal should split on period."""
        s = self._make_snake()
        X = {s.header[1]: "a.b.c"}
        # 3 sentences, threshold 2.0, negat=True means >= 2.0
        literal = [1, 2.0, True, "TSS"]
        assert s.apply_literal(X, literal) is True
        literal_neg = [1, 2.0, False, "TSS"]
        assert s.apply_literal(X, literal_neg) is False

    def test_apply_literal_missing_key_returns_false(self):
        """Missing key should return False, not None."""
        s = self._make_snake()
        X = {}  # empty dict
        literal = [1, "test", False, "T"]
        result = s.apply_literal(X, literal)
        assert result is False

    def test_apply_literal_unknown_datat_returns_false(self):
        """Unknown datatype should return False explicitly."""
        s = self._make_snake()
        X = {s.header[1]: "hello"}
        literal = [1, "test", False, "UNKNOWN"]
        result = s.apply_literal(X, literal)
        assert result is False
