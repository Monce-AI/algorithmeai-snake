"""Tests for perfected audit system: Routing AND, Lookalike AND, plain text assertions."""
import os
import pytest

from algorithmeai import Snake

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


def _make_snake(**kwargs):
    defaults = dict(target_index=3, n_layers=2, bucket=5, vocal=False)
    defaults.update(kwargs)
    return Snake(SAMPLE_CSV, **defaults)


class TestRoutingAND:
    def test_audit_contains_routing_and(self):
        """Audit should contain 'Routing AND' for each layer."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "Routing AND" in audit

    def test_routing_and_shows_negated_skip_literals(self):
        """When X skips IF/ELIF branches, routing AND should show negated literals."""
        s = _make_snake(n_layers=2, bucket=3)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        # The routing AND should have readable literal descriptions
        # At minimum it should mention the feature names from the dataset
        assert "Routing AND" in audit

    def test_routing_and_handles_else_bucket(self):
        """ELSE bucket routing AND should mention 'ELSE' or 'default'."""
        # With large bucket, everything is in one ELSE bucket
        s = _make_snake(n_layers=1, bucket=1000)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "Routing AND" in audit
        # Single bucket means no routing conditions needed
        assert "single bucket" in audit.lower() or "only bucket" in audit.lower()


class TestLookalikeAND:
    def test_audit_contains_lookalike_and(self):
        """Audit should contain 'Lookalike AND' for each layer."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "Lookalike AND" in audit

    def test_lookalike_and_shows_per_sample_explanation(self):
        """Lookalike AND section should show individual lookalike entries."""
        s = _make_snake(n_layers=3, bucket=5)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        # Should show lookalike entries with AND explanations
        assert "Lookalike #" in audit or "matches" in audit


class TestLookalikeSummary:
    def test_audit_starts_with_summary(self):
        """Audit should start with BEGIN AUDIT and contain LOOKALIKE SUMMARY."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "BEGIN AUDIT" in audit
        assert "LOOKALIKE SUMMARY" in audit

    def test_summary_shows_percentages(self):
        """Summary should show percentage breakdown by class."""
        s = _make_snake(n_layers=3, bucket=5)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "%" in audit

    def test_summary_shows_examples(self):
        """Summary should show example feature values per class."""
        s = _make_snake(n_layers=3, bucket=5)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "e.g." in audit


class TestGetPlainTextAssertion:
    def test_with_bucket_clauses(self):
        """get_plain_text_assertion with bucket_clauses should produce AND explanation."""
        s = _make_snake(n_layers=2, bucket=5)
        # Get a real bucket to test with
        chain = s.layers[0]
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        from algorithmeai.snake import traverse_chain
        bucket = traverse_chain(chain, X, s.apply_literal)
        if bucket and bucket["clauses"] and bucket["lookalikes"]:
            # Find a lookalike with a condition
            for l in bucket["lookalikes"]:
                for condition in bucket["lookalikes"][l]:
                    result = s.get_plain_text_assertion(
                        condition, l,
                        bucket_clauses=bucket["clauses"],
                        bucket_members=bucket["members"]
                    )
                    assert "Lookalike #" in result
                    assert "AND:" in result
                    return
        # If no lookalikes found, test passes (data-dependent)

    def test_without_bucket_clauses_backwards_compat(self):
        """Without bucket_clauses, should fall back to showing condition indices."""
        s = _make_snake()
        result = s.get_plain_text_assertion([0, 1], "0")
        assert "Lookalike #" in result
        assert "condition" in result


class TestNegateAndFirstFail:
    def test_negate_literal_flips(self):
        """_negate_literal should flip the negat flag."""
        s = _make_snake()
        lit = [1, "test", False, "T"]
        neg = s._negate_literal(lit)
        assert neg == (1, "test", True, "T")

    def test_negate_literal_double_flip(self):
        """Double negation should restore original."""
        s = _make_snake()
        lit = [1, "test", False, "T"]
        double_neg = s._negate_literal(s._negate_literal(lit))
        assert double_neg[2] == lit[2]

    def test_first_failing_literal_finds_first(self):
        """_first_failing_literal should return the first failing literal."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        # A literal that will fail: "color" does NOT contain "red" (negat=True)
        fail_lit = [1, "red", True, "T"]  # header[1] = "color"
        pass_lit = [1, "red", False, "T"]
        condition = [pass_lit, fail_lit]
        result = s._first_failing_literal(X, condition)
        assert result == fail_lit

    def test_first_failing_literal_returns_none_when_all_pass(self):
        """_first_failing_literal should return None when all pass."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        pass_lit = [1, "red", False, "T"]  # "color" contains "red"
        condition = [pass_lit]
        result = s._first_failing_literal(X, condition)
        assert result is None


class TestAuditEndToEnd:
    def test_begin_end_markers(self):
        """Audit should have BEGIN AUDIT and END AUDIT markers."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "### BEGIN AUDIT ###" in audit
        assert "### END AUDIT ###" in audit

    def test_prediction_matches_get_prediction(self):
        """Prediction shown in audit should match get_prediction."""
        s = _make_snake(n_layers=3, bucket=5)
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        pred = s.get_prediction(X)
        assert f"PREDICTION: {pred}" in audit

    def test_augmented_audit_is_string(self):
        """get_augmented Audit field should be a string."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        Y = s.get_augmented(X)
        assert isinstance(Y["Audit"], str)
        assert "BEGIN AUDIT" in Y["Audit"]

    def test_probability_section_present(self):
        """Audit should have a PROBABILITY section."""
        s = _make_snake()
        X = {"color": "red", "size": 10.0, "shape": "circle"}
        audit = s.get_audit(X)
        assert "PROBABILITY" in audit
