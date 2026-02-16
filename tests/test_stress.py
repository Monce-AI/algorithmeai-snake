"""Stress tests for Snake: oppose() never-None guarantee, construct_sat termination, adversarial inputs."""
import os
import random
import time
import pytest

from algorithmeai import Snake

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


def _make_snake(**kwargs):
    defaults = dict(target_index=3, n_layers=2, bucket=5, vocal=False)
    defaults.update(kwargs)
    return Snake(SAMPLE_CSV, **defaults)


# ---------------------------------------------------------------------------
# TestOpposeNeverNone — after dedup, oppose() between samples of different
# classes must ALWAYS return a literal, never None (now exit).
# ---------------------------------------------------------------------------

class TestOpposeNeverNone:

    def test_brute_force_all_cross_class_pairs(self):
        """Every cross-class pair from sample.csv, 50 random trials each."""
        s = _make_snake()
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(50):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None and len(lit) == 4, (
                        f"oppose() failed for pair {i} vs {j}"
                    )

    def test_text_heavy_data(self):
        """All-text features, multiple classes."""
        data = [
            {"label": "A", "f1": "hello world", "f2": "foo bar"},
            {"label": "A", "f1": "hello earth", "f2": "foo baz"},
            {"label": "B", "f1": "goodbye moon", "f2": "qux quux"},
            {"label": "B", "f1": "goodbye sun", "f2": "qux corge"},
            {"label": "C", "f1": "salut monde", "f2": "abc def"},
            {"label": "C", "f1": "salut terre", "f2": "abc ghi"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(30):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_numeric_heavy_data(self):
        """All-numeric features."""
        data = [
            {"label": "A", "x": 1.0, "y": 10.0, "z": 100.0},
            {"label": "A", "x": 2.0, "y": 20.0, "z": 200.0},
            {"label": "B", "x": 50.0, "y": 60.0, "z": 70.0},
            {"label": "B", "x": 55.0, "y": 65.0, "z": 75.0},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(30):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_mixed_text_numeric(self):
        """Mixed text + numeric features, 3 classes."""
        data = [
            {"label": "X", "name": "alpha", "val": 1.5},
            {"label": "X", "name": "beta", "val": 2.5},
            {"label": "Y", "name": "gamma", "val": 10.5},
            {"label": "Y", "name": "delta", "val": 20.5},
            {"label": "Z", "name": "epsilon", "val": 100.0},
            {"label": "Z", "name": "zeta", "val": 200.0},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(30):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_substring_edge_cases(self):
        """Values that are substrings of each other — tricky for the text path last-resort."""
        data = [
            {"label": "A", "txt": "ab"},
            {"label": "B", "txt": "abc"},
            {"label": "C", "txt": "abcd"},
            {"label": "D", "txt": "abcde"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(50):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_single_char_diffs(self):
        """Strings differing by only one character."""
        data = [
            {"label": "A", "txt": "cat"},
            {"label": "B", "txt": "bat"},
            {"label": "C", "txt": "hat"},
            {"label": "D", "txt": "rat"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(50):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_glass_industry_denominations(self):
        """Realistic glass industry denomination patterns."""
        data = [
            {"article": "60442", "denomination": "44.2"},
            {"article": "60442L", "denomination": "44.2 LowE"},
            {"article": "1004", "denomination": "Float 4mm"},
            {"article": "1004C", "denomination": "Float 4mm clair"},
            {"article": "2006", "denomination": "Parsol bronze 6mm"},
            {"article": "3008", "denomination": "Planitherm 8mm"},
            {"article": "4010", "denomination": "Stadip 10mm"},
            {"article": "5012", "denomination": "SATINOVO 12mm"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(30):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_empty_vs_nonempty_strings(self):
        """Empty string vs non-empty — should always find a discriminator."""
        data = [
            {"label": "A", "txt": ""},
            {"label": "B", "txt": "something"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for _ in range(100):
            lit = s.oppose(s.population[0], s.population[1])
            assert lit is not None
            lit = s.oppose(s.population[1], s.population[0])
            assert lit is not None

    def test_long_strings(self):
        """Very long strings — performance and correctness."""
        data = [
            {"label": "A", "txt": "a" * 1000},
            {"label": "B", "txt": "b" * 1000},
            {"label": "C", "txt": "c" * 500 + "d" * 500},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(20):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_special_separators(self):
        """Strings with /, :, -, comma — the separator split path in oppose()."""
        data = [
            {"label": "A", "txt": "one/two/three"},
            {"label": "B", "txt": "alpha:beta:gamma"},
            {"label": "C", "txt": "x-y-z"},
            {"label": "D", "txt": "p,q,r,s"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for i in range(len(s.population)):
            for j in range(len(s.population)):
                if s.targets[i] == s.targets[j]:
                    continue
                for _ in range(30):
                    lit = s.oppose(s.population[i], s.population[j])
                    assert lit is not None

    def test_randomized_dataset(self):
        """30 random samples, 3 classes — oppose never None across random pairs."""
        random.seed(42)
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        data = [
            {"label": f"c{i % 3}",
             "f1": f"{random.choice(words)} {random.choice(words)}",
             "f2": random.uniform(0, 100)}
            for i in range(30)
        ]
        s = Snake(data, n_layers=1, bucket=15, vocal=False)
        for _ in range(200):
            i = random.randint(0, len(s.population) - 1)
            j = random.randint(0, len(s.population) - 1)
            if s.targets[i] == s.targets[j]:
                continue
            lit = s.oppose(s.population[i], s.population[j])
            assert lit is not None


# ---------------------------------------------------------------------------
# TestOpposeExitCanaries — verify exit() fires on impossible states
# ---------------------------------------------------------------------------

class TestOpposeExitCanaries:

    def test_identical_samples_same_object(self):
        """Same object as both T and F triggers exit."""
        s = _make_snake()
        T = s.population[0]
        with pytest.raises(SystemExit):
            s.oppose(T, T)

    def test_identical_features_different_objects(self):
        """Two distinct dicts with identical feature values triggers exit."""
        s = _make_snake()
        T = s.population[0].copy()
        F = s.population[0].copy()
        with pytest.raises(SystemExit):
            s.oppose(T, F)

    def test_nan_numeric_exits(self):
        """NaN in a numeric feature triggers exit (caught at candidates filter)."""
        data = [
            {"label": "A", "val": 1.0},
            {"label": "B", "val": 2.0},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        # NaN is filtered out of candidates by the NaN guard, leaving empty candidates -> exit
        T = {"val": float('nan')}
        F = {"val": 1.0}
        with pytest.raises(SystemExit):
            s.oppose(T, F)


# ---------------------------------------------------------------------------
# TestConstructSatTermination — construct_sat must always terminate
# ---------------------------------------------------------------------------

class TestConstructSatTermination:

    def test_basic_sample_csv(self):
        """Terminates on sample.csv for every target value."""
        s = _make_snake()
        for target_val in s._unique_targets():
            sat = s.construct_sat(target_val)
            assert isinstance(sat, list)

    def test_two_classes(self):
        """Minimal 2-class problem terminates."""
        data = [
            {"label": "A", "x": "foo"},
            {"label": "B", "x": "bar"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        sat_a = s.construct_sat("A")
        sat_b = s.construct_sat("B")
        assert isinstance(sat_a, list)
        assert isinstance(sat_b, list)

    def test_ten_classes_sparse(self):
        """10 classes, sparse features."""
        data = [{"label": chr(65 + i), "x": f"word_{i}", "y": float(i)} for i in range(10)]
        s = Snake(data, n_layers=1, bucket=15, vocal=False)
        for target_val in s._unique_targets():
            sat = s.construct_sat(target_val)
            assert isinstance(sat, list)

    def test_near_identical_features(self):
        """Samples sharing all but one feature — exercises empty clause path."""
        data = [
            {"label": "A", "x": "same", "y": "diff_a"},
            {"label": "B", "x": "same", "y": "diff_b"},
            {"label": "C", "x": "same", "y": "diff_c"},
        ]
        s = Snake(data, n_layers=1, bucket=5, vocal=False)
        for target_val in s._unique_targets():
            sat = s.construct_sat(target_val)
            assert isinstance(sat, list)

    def test_construct_sat_with_timeout(self):
        """20 samples, 3 classes — must complete within 10s."""
        random.seed(42)
        data = [
            {"label": random.choice(["A", "B", "C"]),
             "f1": random.choice(["alpha", "beta", "gamma"]),
             "f2": random.uniform(0, 100)}
            for _ in range(20)
        ]
        s = Snake(data, n_layers=1, bucket=10, vocal=False)
        start = time.time()
        for target_val in s._unique_targets():
            sat = s.construct_sat(target_val)
            assert isinstance(sat, list)
        elapsed = time.time() - start
        assert elapsed < 10, f"construct_sat took {elapsed:.1f}s"

    def test_clauses_cover_all_positives(self):
        """Every positive sample must end up in exactly one consequence group."""
        random.seed(42)
        s = _make_snake()
        for target_val in s._unique_targets():
            sat = s.construct_sat(target_val)
            positive_indices = {i for i in range(len(s.population)) if s.targets[i] == target_val}
            covered = set()
            for clause, consequence in sat:
                for idx in consequence:
                    covered.add(idx)
            assert covered == positive_indices

    def test_single_sample_per_class(self):
        """Exactly 1 sample per class — construct_sat gets 1 F and N-1 Ts."""
        data = [
            {"label": "A", "x": "alpha"},
            {"label": "B", "x": "beta"},
            {"label": "C", "x": "gamma"},
            {"label": "D", "x": "delta"},
            {"label": "E", "x": "epsilon"},
        ]
        s = Snake(data, n_layers=1, bucket=10, vocal=False)
        for target_val in s._unique_targets():
            sat = s.construct_sat(target_val)
            assert isinstance(sat, list)
            assert len(sat) >= 1


# ---------------------------------------------------------------------------
# TestFullTrainingStress — end-to-end training under adversarial conditions
# ---------------------------------------------------------------------------

class TestFullTrainingStress:

    def test_high_cardinality(self):
        """10 classes, 20 samples — trains and predicts."""
        data = [
            {"label": f"class_{i % 10}", "f1": f"word_{i}", "f2": float(i * 3.14)}
            for i in range(20)
        ]
        s = Snake(data, n_layers=1, bucket=10, vocal=False)
        pred = s.get_prediction({"f1": "word_0", "f2": 0.0})
        assert pred is not None

    def test_single_text_feature(self):
        """Only 1 text feature."""
        data = [
            {"label": "A", "txt": "hello"},
            {"label": "B", "txt": "world"},
            {"label": "C", "txt": "foo"},
            {"label": "A", "txt": "hello there"},
            {"label": "B", "txt": "world wide"},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        prob = s.get_probability({"txt": "hello"})
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_single_numeric_feature(self):
        """Only 1 numeric feature."""
        data = (
            [{"label": "low", "val": float(i)} for i in range(5)]
            + [{"label": "high", "val": float(i + 100)} for i in range(5)]
        )
        s = Snake(data, n_layers=2, bucket=5, vocal=False)
        prob = s.get_probability({"val": 2.0})
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_many_features(self):
        """10 features — broader feature space."""
        data = []
        for i in range(15):
            row = {"label": f"c{i % 3}"}
            for f in range(10):
                row[f"f{f}"] = f"v{(i + f) % 5}"
            data.append(row)
        s = Snake(data, n_layers=1, bucket=10, vocal=False)
        query = {f"f{f}": "v0" for f in range(10)}
        pred = s.get_prediction(query)
        assert pred in ["c0", "c1", "c2"]

    def test_noise_zero(self):
        """noise=0 (strict partitioning)."""
        data = [
            {"label": chr(65 + i % 4), "x": f"val_{i}", "y": float(i)}
            for i in range(12)
        ]
        s = Snake(data, n_layers=2, bucket=5, noise=0.0, vocal=False)
        assert len(s.layers) == 2

    def test_noise_max(self):
        """noise=1.0 (maximum noise injection)."""
        data = [
            {"label": chr(65 + i % 4), "x": f"val_{i}", "y": float(i)}
            for i in range(12)
        ]
        s = Snake(data, n_layers=2, bucket=5, noise=1.0, vocal=False)
        assert len(s.layers) == 2

    def test_bucket_larger_than_population(self):
        """bucket > population — everything in one ELSE bucket."""
        data = [
            {"label": "A", "x": "foo"},
            {"label": "B", "x": "bar"},
            {"label": "C", "x": "baz"},
        ]
        s = Snake(data, n_layers=2, bucket=1000, vocal=False)
        assert len(s.layers) == 2
        pred = s.get_prediction({"x": "foo"})
        assert pred in ["A", "B", "C"]

    def test_bucket_size_one(self):
        """bucket=1 — extreme splitting."""
        data = [
            {"label": chr(65 + i % 3), "x": f"word_{i}", "y": float(i)}
            for i in range(9)
        ]
        s = Snake(data, n_layers=1, bucket=1, vocal=False)
        assert len(s.layers) == 1

    def test_special_characters_glass(self):
        """Text with /, :, -, commas — real glass data patterns."""
        data = [
            {"label": "A", "txt": "44.2 LowE / bronze"},
            {"label": "B", "txt": "16 THN thermix-noir"},
            {"label": "C", "txt": "VT4: Stopsol gris, clair"},
            {"label": "A", "txt": "44.2 feuillete"},
            {"label": "B", "txt": "16 TGI argon"},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        pred = s.get_prediction({"txt": "44.2 LowE"})
        assert pred in ["A", "B", "C"]

    def test_empty_string_features(self):
        """Empty strings as feature values."""
        data = [
            {"label": "A", "x": "", "y": "something"},
            {"label": "B", "x": "notempty", "y": ""},
            {"label": "A", "x": "", "y": "else"},
            {"label": "B", "x": "filled", "y": ""},
        ]
        s = Snake(data, n_layers=2, bucket=3, vocal=False)
        pred = s.get_prediction({"x": "", "y": "something"})
        assert pred in ["A", "B"]

    def test_save_load_roundtrip(self):
        """Train, save, load — loaded model gives valid predictions."""
        import tempfile
        data = [
            {"label": "A", "x": "alpha", "y": 1.0},
            {"label": "A", "x": "beta", "y": 2.0},
            {"label": "B", "x": "gamma", "y": 10.0},
            {"label": "B", "x": "delta", "y": 20.0},
        ]
        s = Snake(data, n_layers=2, bucket=5, vocal=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            s2 = Snake(path)
            pred = s2.get_prediction({"x": "alpha", "y": 1.0})
            assert pred in ["A", "B"]
        finally:
            os.unlink(path)

    def test_many_layers(self):
        """n_layers=5 on small data."""
        data = [
            {"label": "A", "x": "foo", "y": 1.0},
            {"label": "B", "x": "bar", "y": 2.0},
            {"label": "C", "x": "baz", "y": 3.0},
            {"label": "A", "x": "foo2", "y": 4.0},
            {"label": "B", "x": "bar2", "y": 5.0},
        ]
        s = Snake(data, n_layers=5, bucket=3, vocal=False)
        assert len(s.layers) == 5
        prob = s.get_probability({"x": "foo", "y": 1.0})
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_batch_prediction(self):
        """Batch predict 50 queries — all must return valid results."""
        random.seed(42)
        data = [
            {"label": f"c{i % 3}", "x": f"w{i}", "y": float(i)}
            for i in range(15)
        ]
        s = Snake(data, n_layers=2, bucket=5, vocal=False)
        queries = [{"x": f"w{random.randint(0, 14)}", "y": random.uniform(0, 15)} for _ in range(50)]
        results = s.get_batch_prediction(queries)
        assert len(results) == 50
        for r in results:
            assert "prediction" in r
            assert "probability" in r
            assert "confidence" in r
            assert abs(sum(r["probability"].values()) - 1.0) < 1e-9
