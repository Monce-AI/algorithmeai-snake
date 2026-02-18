"""Ultimate stress test for Snake — adversarial parameters, pathological data, Cython parity.

Every test has a 10s hard timeout via signal.alarm.
No Snake constructor receives more than 500 training samples.
All training runs with vocal=True to surface hangs immediately.
"""
import os
import random
import signal
import sys
import time
import pytest

from algorithmeai import Snake
from algorithmeai.snake import _HAS_ACCEL

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES, "sample.csv")


class _Timeout:
    """Context manager: raises TimeoutError after `seconds`."""
    def __init__(self, seconds=10):
        self.seconds = seconds
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self
    def __exit__(self, *args):
        signal.alarm(0)
    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("Snake operation exceeded timeout — possible infinite loop")


# ---------------------------------------------------------------------------
# Pathological data generators
# ---------------------------------------------------------------------------

def _gen_high_cardinality(n_classes, samples_per_class, n_features):
    """Many classes, few samples each."""
    data = []
    for c in range(n_classes):
        for s in range(samples_per_class):
            row = {"label": f"class_{c}"}
            for f in range(n_features):
                row[f"f{f}"] = f"v{(c * samples_per_class + s + f) % (n_classes * 2)}"
            data.append(row)
    return data


def _gen_overlapping_text(n_samples, n_classes):
    """Text features where values are substrings/superstrings of each other."""
    base = "abcdefghijklmnopqrstuvwxyz"
    data = []
    for i in range(n_samples):
        prefix = base[:3 + (i % 20)]
        suffix = base[(i % 7):(i % 7) + 5]
        data.append({
            "label": f"c{i % n_classes}",
            "txt": prefix + " " + suffix,
            "num": float(i),
        })
    return data


def _gen_near_duplicate(n_samples, n_classes):
    """Samples that differ on exactly 1 feature — stress for oppose() and clause construction."""
    data = []
    for i in range(n_samples):
        data.append({
            "label": f"c{i % n_classes}",
            "shared": "identical_for_all",
            "diff": f"unique_{i}",
        })
    return data


def _gen_all_numeric(n_samples, n_classes, n_features):
    """Pure numeric features with tight clustering."""
    random.seed(42)
    data = []
    for i in range(n_samples):
        row = {"label": f"c{i % n_classes}"}
        for f in range(n_features):
            center = (i % n_classes) * 10 + f
            row[f"f{f}"] = center + random.uniform(-0.5, 0.5)
        data.append(row)
    return data


def _gen_mixed_heavy(n_samples, n_classes, n_text, n_numeric):
    """Many mixed features — realistic production-like data."""
    random.seed(42)
    words = ["glass", "float", "lowe", "bronze", "clair", "gris", "noir",
             "blanc", "thermix", "planitherm", "stadip", "parsol", "satinovo",
             "feuillete", "trempe", "securit", "argon", "TGI", "THN", "VT4"]
    data = []
    for i in range(n_samples):
        row = {"label": f"art_{i % n_classes}"}
        for t in range(n_text):
            row[f"txt{t}"] = f"{random.choice(words)} {random.choice(words)} {i % 7}"
        for n in range(n_numeric):
            row[f"num{n}"] = random.uniform(0, 100) + (i % n_classes) * 5
        data.append(row)
    return data


# ---------------------------------------------------------------------------
# TestUltimateOppose — oppose() must NEVER return None on valid cross-class
# pairs, across every pathological data shape we can throw at it.
# ---------------------------------------------------------------------------

class TestUltimateOppose:

    def test_high_cardinality_50_classes(self):
        """50 classes, 3 samples each = 150 samples — oppose on all cross-class pairs."""
        data = _gen_high_cardinality(50, 3, 4)
        with _Timeout(10):
            s = Snake(data, n_layers=1, bucket=50, vocal=True)
        # Random sample of 500 cross-class pairs
        random.seed(42)
        tested = 0
        for _ in range(1000):
            i = random.randint(0, len(s.population) - 1)
            j = random.randint(0, len(s.population) - 1)
            if s.targets[i] == s.targets[j]:
                continue
            lit = s.oppose(s.population[i], s.population[j])
            assert lit is not None and len(lit) == 4
            tested += 1
            if tested >= 500:
                break
        assert tested >= 300

    def test_overlapping_text_200_samples(self):
        """200 samples with substring-heavy text features."""
        data = _gen_overlapping_text(200, 5)
        with _Timeout(10):
            s = Snake(data, n_layers=1, bucket=100, vocal=True)
        random.seed(42)
        for _ in range(500):
            i = random.randint(0, len(s.population) - 1)
            j = random.randint(0, len(s.population) - 1)
            if s.targets[i] == s.targets[j]:
                continue
            lit = s.oppose(s.population[i], s.population[j])
            assert lit is not None

    def test_near_duplicate_features(self):
        """All rows share 1 feature, differ on 1 — clause construction must cope."""
        data = _gen_near_duplicate(100, 5)
        with _Timeout(10):
            s = Snake(data, n_layers=1, bucket=50, vocal=True)
        for i in range(len(s.population)):
            for j in range(i + 1, min(i + 10, len(s.population))):
                if s.targets[i] == s.targets[j]:
                    continue
                lit = s.oppose(s.population[i], s.population[j])
                assert lit is not None

    def test_all_numeric_tight_clusters(self):
        """Pure numeric data with overlapping clusters — midpoint splits galore."""
        data = _gen_all_numeric(200, 10, 5)
        with _Timeout(10):
            s = Snake(data, n_layers=1, bucket=100, vocal=True)
        random.seed(42)
        for _ in range(500):
            i = random.randint(0, len(s.population) - 1)
            j = random.randint(0, len(s.population) - 1)
            if s.targets[i] == s.targets[j]:
                continue
            lit = s.oppose(s.population[i], s.population[j])
            assert lit is not None


# ---------------------------------------------------------------------------
# TestUltimateTraining — full training under extreme parameters, all within
# 10s and 500 samples. Proves no infinite loops anywhere in the pipeline.
# ---------------------------------------------------------------------------

class TestUltimateTraining:

    def test_500_samples_10_classes_5_layers(self):
        """Max scale: 500 samples, 10 classes, 5 layers, bucket=100."""
        data = _gen_mixed_heavy(500, 10, 3, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=5, bucket=100, vocal=True)
        assert len(s.layers) == 5
        prob = s.get_probability({"txt0": "glass lowe 0", "txt1": "bronze clair 0",
                                   "txt2": "parsol noir 0", "num0": 50.0, "num1": 25.0})
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_500_samples_50_classes(self):
        """High cardinality: 500 samples across 50 classes."""
        data = _gen_high_cardinality(50, 10, 3)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=50, vocal=True)
        assert len(s.layers) == 3

    def test_500_samples_bucket_1(self):
        """Extreme bucket=1 on 100 samples — maximum chain splitting."""
        data = _gen_mixed_heavy(100, 5, 2, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=1, bucket=1, vocal=True)
        assert len(s.layers) == 1
        pred = s.get_prediction({"txt0": "glass lowe 0", "txt1": "clair noir 0",
                                  "num0": 50.0, "num1": 25.0})
        assert pred is not None

    def test_500_samples_bucket_500(self):
        """bucket >= population — one giant ELSE bucket, heavy SAT."""
        data = _gen_mixed_heavy(200, 8, 3, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=2, bucket=500, vocal=True)
        assert len(s.layers) == 2

    def test_300_samples_noise_zero(self):
        """noise=0 strict partitioning — no cross-bucket bleeding."""
        data = _gen_mixed_heavy(300, 6, 2, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=50, noise=0.0, vocal=True)
        assert len(s.layers) == 3

    def test_300_samples_noise_max(self):
        """noise=1.0 — every bucket gets full noise injection."""
        data = _gen_mixed_heavy(300, 6, 2, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=50, noise=1.0, vocal=True)
        assert len(s.layers) == 3

    def test_overlapping_text_heavy(self):
        """200 samples, substring-heavy text, 3 layers."""
        data = _gen_overlapping_text(200, 5)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=40, vocal=True)
        assert len(s.layers) == 3

    def test_near_duplicate_training(self):
        """100 near-duplicate samples — clause construction must handle shared features."""
        data = _gen_near_duplicate(100, 5)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=20, vocal=True)
        assert len(s.layers) == 3

    def test_all_numeric_200(self):
        """200 pure-numeric samples, 5 features, 10 classes."""
        data = _gen_all_numeric(200, 10, 5)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=50, vocal=True)
        assert len(s.layers) == 3

    def test_single_feature_300_samples(self):
        """Only 1 text feature — entire discrimination from one column."""
        random.seed(42)
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        data = [{"label": f"c{i % 4}", "txt": f"{random.choice(words)} {random.choice(words)} {random.choice(words)}"}
                for i in range(300)]
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=75, vocal=True)
        assert len(s.layers) == 3


# ---------------------------------------------------------------------------
# TestUltimatePrediction — prediction pipeline under stress
# ---------------------------------------------------------------------------

class TestUltimatePrediction:

    def test_batch_500_queries(self):
        """Train on 200 samples, batch predict 500 queries."""
        data = _gen_mixed_heavy(200, 5, 2, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=50, vocal=True)
        random.seed(42)
        queries = [{"txt0": f"glass {random.choice(['lowe','bronze','clair'])} {i%5}",
                     "txt1": f"noir planitherm {i%3}",
                     "num0": random.uniform(0, 100),
                     "num1": random.uniform(0, 50)}
                    for i in range(500)]
        t0 = time.time()
        results = s.get_batch_prediction(queries)
        elapsed = time.time() - t0
        assert len(results) == 500
        for r in results:
            assert abs(sum(r["probability"].values()) - 1.0) < 1e-9
        avg_ms = elapsed / 500 * 1000
        print(f"\n  Batch 500 queries: {elapsed:.2f}s total, {avg_ms:.2f}ms/query")

    def test_augmented_prediction(self):
        """get_augmented on 200 samples — single pass, returns all fields."""
        data = _gen_mixed_heavy(200, 5, 2, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=50, vocal=True)
        query = {"txt0": "glass lowe 0", "txt1": "clair noir 0", "num0": 50.0, "num1": 25.0}
        aug = s.get_augmented(query)
        assert "Prediction" in aug
        assert "Probability" in aug
        assert "Lookalikes" in aug
        assert "Audit" in aug
        assert abs(sum(aug["Probability"].values()) - 1.0) < 1e-9

    def test_audit_readable(self):
        """Audit string must be non-empty and contain layer/bucket info."""
        data = _gen_mixed_heavy(100, 4, 2, 1)
        with _Timeout(10):
            s = Snake(data, n_layers=2, bucket=25, vocal=True)
        query = {"txt0": "glass lowe 0", "txt1": "clair noir 0", "num0": 50.0}
        audit = s.get_audit(query)
        assert "LAYER 0" in audit
        assert "PREDICTION" in audit
        assert "bucket" in audit.lower()

    def test_empty_query(self):
        """Empty dict {} — uniform probability, no crash."""
        data = _gen_mixed_heavy(100, 4, 2, 1)
        with _Timeout(10):
            s = Snake(data, n_layers=2, bucket=25, vocal=True)
        prob = s.get_probability({})
        assert abs(sum(prob.values()) - 1.0) < 1e-9

    def test_partial_features(self):
        """Only some features present — graceful degradation."""
        data = _gen_mixed_heavy(100, 4, 3, 3)
        with _Timeout(10):
            s = Snake(data, n_layers=2, bucket=25, vocal=True)
        # Only 1 out of 6 features
        pred = s.get_prediction({"txt0": "glass lowe 0"})
        assert pred is not None


# ---------------------------------------------------------------------------
# TestUltimateSaveLoad — serialization roundtrip under stress
# ---------------------------------------------------------------------------

class TestUltimateSaveLoad:

    def test_save_load_500_samples(self):
        """Train 500 samples, save, load, predictions match."""
        import tempfile
        data = _gen_mixed_heavy(500, 10, 3, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=3, bucket=100, vocal=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            s.to_json(path)
            s2 = Snake(path)
            # Same predictions on 50 random queries
            random.seed(42)
            for _ in range(50):
                q = {"txt0": f"glass {random.choice(['lowe','bronze'])} {random.randint(0,5)}",
                     "txt1": f"clair {random.randint(0,3)}",
                     "txt2": f"noir {random.randint(0,3)}",
                     "num0": random.uniform(0, 100),
                     "num1": random.uniform(0, 50)}
                p1 = s.get_prediction(q)
                p2 = s2.get_prediction(q)
                assert p1 == p2, f"Prediction mismatch after save/load: {p1} != {p2}"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestUltimateCythonParity — when Cython is loaded, results must match
# pure Python logic on the same data.
# ---------------------------------------------------------------------------

class TestUltimateCythonParity:

    @pytest.mark.skipif(not _HAS_ACCEL, reason="Cython not compiled")
    def test_apply_literal_parity(self):
        """Cython apply_literal matches Python on 500 random literal evaluations."""
        from algorithmeai.snake import apply_literal_fast
        data = _gen_mixed_heavy(50, 3, 2, 2)
        s = Snake(data, n_layers=1, bucket=25, vocal=True)
        random.seed(42)
        for _ in range(500):
            idx = random.randint(0, len(s.population) - 1)
            X = s.population[idx]
            # Generate a random literal by opposing two random samples
            i = random.randint(0, len(s.population) - 1)
            j = random.randint(0, len(s.population) - 1)
            if s.targets[i] == s.targets[j]:
                continue
            lit = s.oppose(s.population[i], s.population[j])
            if lit is None:
                continue
            cy_result = apply_literal_fast(X, lit, s.header)
            # Pure Python evaluation
            index, value, negat, datat = lit
            h = s.header[index]
            if h not in X:
                py_result = False
            else:
                field = X[h]
                if datat == "T":
                    py_result = (value not in field) if negat else (value in field)
                elif datat == "TN":
                    py_result = (value <= len(field)) if negat else (value > len(field))
                elif datat == "TLN":
                    py_result = (value <= len(set(field))) if negat else (value > len(set(field)))
                elif datat == "TWS":
                    py_result = (value <= len(field.split(" "))) if negat else (value > len(field.split(" ")))
                elif datat == "TPS":
                    py_result = (value <= len(field.split(","))) if negat else (value > len(field.split(",")))
                elif datat == "TSS":
                    py_result = (value <= len(field.split("."))) if negat else (value > len(field.split(".")))
                elif datat == "N":
                    py_result = (value <= field) if negat else (value > field)
                else:
                    py_result = False
            assert cy_result == py_result, f"Parity fail: lit={lit}, X[{h}]={X.get(h)}, cy={cy_result}, py={py_result}"

    @pytest.mark.skipif(not _HAS_ACCEL, reason="Cython not compiled")
    def test_traverse_chain_parity(self):
        """Cython traverse_chain matches Python on 200 queries."""
        from algorithmeai.snake import traverse_chain_fast, traverse_chain
        data = _gen_mixed_heavy(100, 4, 2, 2)
        s = Snake(data, n_layers=3, bucket=25, vocal=True)
        random.seed(42)
        for _ in range(200):
            q = {"txt0": f"glass {random.choice(['lowe','bronze','clair'])} {random.randint(0,5)}",
                 "txt1": f"noir {random.randint(0,3)}",
                 "num0": random.uniform(0, 100),
                 "num1": random.uniform(0, 50)}
            for layer in s.layers:
                py_bucket = traverse_chain(layer, q, s.apply_literal)
                cy_bucket = traverse_chain_fast(layer, q, s.header)
                if py_bucket is None:
                    assert cy_bucket is None
                else:
                    assert py_bucket["members"] == cy_bucket["members"]


# ---------------------------------------------------------------------------
# TestUltimateConstructSat — construct_sat (legacy) termination guarantee
# ---------------------------------------------------------------------------

class TestUltimateConstructSat:

    def test_construct_sat_200_samples(self):
        """construct_sat on 200 mixed samples, every target, within 10s."""
        data = _gen_mixed_heavy(200, 8, 2, 2)
        with _Timeout(10):
            s = Snake(data, n_layers=1, bucket=100, vocal=True)
        with _Timeout(10):
            for tv in s._unique_targets():
                sat = s.construct_sat(tv)
                assert isinstance(sat, list)
                # All positives covered
                positives = {i for i in range(len(s.population)) if s.targets[i] == tv}
                covered = set()
                for clause, consequence in sat:
                    covered.update(consequence)
                assert covered == positives, f"Target {tv}: {positives - covered} not covered"

    def test_construct_sat_near_duplicates(self):
        """Pathological near-duplicate data — empty clause guard must fire."""
        data = _gen_near_duplicate(50, 5)
        with _Timeout(10):
            s = Snake(data, n_layers=1, bucket=25, vocal=True)
        with _Timeout(10):
            for tv in s._unique_targets():
                sat = s.construct_sat(tv)
                assert isinstance(sat, list)


# ---------------------------------------------------------------------------
# TestUltimateValidation — make_validation pruning under stress
# ---------------------------------------------------------------------------

class TestUltimateValidation:

    def test_validation_pruning_200_samples(self):
        """Train 200 samples with 5 layers, prune to 50%."""
        data = _gen_mixed_heavy(200, 5, 2, 2)
        random.seed(42)
        random.shuffle(data)
        train = data[:160]
        val = data[160:]
        with _Timeout(10):
            s = Snake(train, n_layers=5, bucket=50, vocal=True)
        assert len(s.layers) == 5
        with _Timeout(10):
            s.make_validation(val, pruning_coef=0.5)
        assert len(s.layers) <= 3  # 50% of 5 = 2 or 3
        # Predictions still valid after pruning
        prob = s.get_probability({"txt0": "glass lowe 0", "txt1": "clair noir 0",
                                   "num0": 50.0, "num1": 25.0})
        assert abs(sum(prob.values()) - 1.0) < 1e-9
