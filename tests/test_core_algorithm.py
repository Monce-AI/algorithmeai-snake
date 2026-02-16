"""Tests for Snake core algorithm: oppose, construct_clause, construct_sat."""
import os
import random
import pytest

from algorithmeai import Snake

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


def _make_snake(**kwargs):
    defaults = dict(target_index=3, n_layers=2, bucket=5, vocal=False)
    defaults.update(kwargs)
    return Snake(SAMPLE_CSV, **defaults)


# ---------------------------------------------------------------------------
# TestOppose
# ---------------------------------------------------------------------------

class TestOppose:
    def test_oppose_returns_valid_literal_or_none(self):
        """oppose() returns [index, value, negat, type] or None."""
        s = _make_snake()
        T = s.population[0]
        F = s.population[3]  # different class
        random.seed(42)
        for _ in range(20):
            lit = s.oppose(T, F)
            if lit is not None:
                assert isinstance(lit, list) and len(lit) == 4
                assert isinstance(lit[0], int)
                assert isinstance(lit[2], bool)
                assert isinstance(lit[3], str)

    def test_oppose_text_literal_types(self):
        """Diverse text data should produce various T/TN/TLN/TWS/TPS/TSS types."""
        data = [
            {"label": "A", "txt": "hello world foo"},
            {"label": "A", "txt": "hello world bar"},
            {"label": "B", "txt": "hi"},
            {"label": "B", "txt": "x,y,z one.two"},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        T = s.population[0]
        F = s.population[2]
        types_seen = set()
        for _ in range(200):
            lit = s.oppose(T, F)
            if lit is not None:
                types_seen.add(lit[3])
        # At least T and one structural type should appear
        assert "T" in types_seen
        assert len(types_seen) >= 2

    def test_oppose_numeric_literal(self):
        """Numeric feature difference should produce N type."""
        data = [
            {"label": "A", "val": 10.0},
            {"label": "B", "val": 50.0},
        ]
        s = Snake(data, n_layers=1, bucket=3, vocal=False)
        T = s.population[0]
        F = s.population[1]
        found_n = False
        for _ in range(50):
            lit = s.oppose(T, F)
            if lit is not None and lit[3] == "N":
                found_n = True
                break
        assert found_n

    def test_oppose_identical_features_returns_none(self):
        """Same features -> None."""
        s = _make_snake()
        T = s.population[0]
        result = s.oppose(T, T)
        assert result is None

    def test_oppose_never_uses_target_column(self):
        """literal[0] should never be 0 (the target column index)."""
        s = _make_snake()
        for _ in range(100):
            T = s.population[0]
            F = s.population[3]
            lit = s.oppose(T, F)
            if lit is not None:
                assert lit[0] != 0, "oppose() should never use the target column"


# ---------------------------------------------------------------------------
# TestConstructClause
# ---------------------------------------------------------------------------

class TestConstructClause:
    def test_clause_true_on_all_ts(self):
        """clause (OR of literals) should be True on every T in Ts."""
        random.seed(42)
        s = _make_snake()
        F = s.population[3]  # class B
        Ts = [dp for dp in s.population if dp[s.target] != F[s.target]]
        clause = s.construct_clause(F, Ts)
        for T in Ts:
            assert s.apply_clause(T, clause), f"Clause not True on T: {T}"

    def test_clause_is_minimal(self):
        """Removing any single literal should cause at least one T to fail."""
        random.seed(42)
        s = _make_snake()
        F = s.population[3]
        Ts = [dp for dp in s.population if dp[s.target] != F[s.target]]
        clause = s.construct_clause(F, Ts)
        if len(clause) <= 1:
            return  # single-literal clause is trivially minimal
        for i in range(len(clause)):
            sub_clause = [clause[j] for j in range(len(clause)) if j != i]
            # At least one T should fail with this sub-clause
            some_fail = any(not s.apply_clause(T, sub_clause) for T in Ts)
            assert some_fail, f"Removing literal {i} didn't break coverage — clause not minimal"

    def test_clause_with_single_pair(self):
        """F vs [T] should produce a clause."""
        random.seed(42)
        s = _make_snake()
        F = s.population[3]
        T = s.population[0]
        clause = s.construct_clause(F, [T])
        assert isinstance(clause, list)
        assert len(clause) >= 1
        assert s.apply_clause(T, clause)


# ---------------------------------------------------------------------------
# TestConstructSat
# ---------------------------------------------------------------------------

class TestConstructSat:
    def test_sat_returns_clause_consequence_pairs(self):
        """Each entry in SAT is [clause, consequence_indices]."""
        random.seed(42)
        s = _make_snake()
        target_val = s.targets[0]
        sat = s.construct_sat(target_val)
        assert isinstance(sat, list)
        for entry in sat:
            assert isinstance(entry, list) and len(entry) == 2
            clause, consequence = entry
            assert isinstance(clause, list)
            assert isinstance(consequence, list)

    def test_sat_clauses_cover_all_positives(self):
        """Every positive sample is in exactly one consequence group."""
        random.seed(42)
        s = _make_snake()
        target_val = s.targets[0]
        sat = s.construct_sat(target_val)
        positive_indices = {i for i in range(len(s.population)) if s.targets[i] == target_val}
        covered = set()
        for clause, consequence in sat:
            for idx in consequence:
                assert idx not in covered, f"Index {idx} appears in multiple consequence groups"
                covered.add(idx)
        assert covered == positive_indices

    def test_sat_clauses_separate_f_from_ts(self):
        """Each clause should be true on non-target samples (Ts) for at least some of them."""
        random.seed(42)
        s = _make_snake()
        target_val = s.targets[0]
        sat = s.construct_sat(target_val)
        Ts = [s.population[i] for i in range(len(s.population)) if s.targets[i] != target_val]
        for clause, _ in sat:
            # The clause should be True on all Ts (by construction)
            for T in Ts:
                assert s.apply_clause(T, clause), "Clause doesn't separate Ts"
