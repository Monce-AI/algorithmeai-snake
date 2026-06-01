"""Tests for v5.4.8 — polymorphic batch inference + stripped serialization.

Every single-dict inference method also accepts a list of dicts and divides the
work across a proportionate number of CPU processes. The parallel result must be
element-for-element identical to the sequential one (layer independence + the
non-dedup vote merge make batching exact, not approximate). Stripped models drop
the population and serve the hot path only.
"""
import json
import os
import random

import pytest

from algorithmeai import Snake


def _make_model(n=400, n_layers=5, bucket=120, workers=1):
    random.seed(13)
    rows = []
    for _ in range(n):
        a, b = random.random(), random.random()
        lab = "A" if a + b > 1.0 else ("B" if a > 0.5 else "C")
        rows.append({
            "label": lab,
            "fa": round(a, 3),
            "fb": round(b, 3),
            "name": random.choice(["riou", "vit", "euro", "tgvi"]),
        })
    return Snake(rows, target_index="label", n_layers=n_layers,
                 bucket=bucket, noise=0.25, workers=workers)


def _make_numeric_model(n=400, n_layers=5, bucket=120):
    """Continuous-target model so candle/regression return real numbers."""
    random.seed(21)
    rows = []
    for _ in range(n):
        a, b = random.random(), random.random()
        rows.append({
            "y": round(2.5 * a + b * b, 4),
            "fa": round(a, 3),
            "fb": round(b, 3),
        })
    return Snake(rows, target_index="y", n_layers=n_layers,
                 bucket=bucket, noise=0.25, workers=1)


def _queries(k):
    random.seed(99)
    return [{"fa": round(random.random(), 3),
             "fb": round(random.random(), 3),
             "name": random.choice(["riou", "vit"])} for _ in range(k)]


# ---------------------------------------------------------------------------
# Polymorphic dispatch: list[dict] in, list[result] out, order preserved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [
    "get_prediction", "get_probability", "get_lookalikes",
    "get_lookalikes_labeled",
])
def test_batch_matches_sequential_inline(method):
    """Small batches run inline but must still return a list matching the
    per-item sequential calls exactly."""
    m = _make_model()
    Xs = _queries(20)
    seq = [getattr(m, method)(X) for X in Xs]
    batch = getattr(m, method)(Xs)
    assert isinstance(batch, list)
    assert len(batch) == len(Xs)
    assert batch == seq


@pytest.mark.parametrize("method", ["get_candle", "get_regression"])
def test_batch_matches_sequential_numeric(method):
    """candle/regression need a continuous target to return real numbers."""
    m = _make_numeric_model()
    Xs = [{"fa": round(random.random(), 3), "fb": round(random.random(), 3)}
          for _ in range(20)]
    seq = [getattr(m, method)(X) for X in Xs]
    batch = getattr(m, method)(Xs)
    assert isinstance(batch, list)
    assert len(batch) == len(Xs)
    # Compare numerically (regression -> float, candle -> Candle with .median)
    if method == "get_regression":
        assert batch == pytest.approx(seq, nan_ok=True)
    else:
        assert [c.median for c in batch] == pytest.approx(
            [c.median for c in seq], nan_ok=True)


@pytest.mark.parametrize("method", ["get_prediction", "get_probability"])
def test_batch_matches_sequential_parallel(method):
    """Above the parallel threshold the pool forks — result must stay exact."""
    m = _make_model()
    m._parallel_threshold = 8  # force the parallel path on a small batch
    Xs = _queries(200)
    seq = [getattr(m, method)(X) for X in Xs]
    batch = getattr(m, method)(Xs)
    assert batch == seq


def test_empty_batch_returns_empty_list():
    m = _make_model()
    assert m.get_prediction([]) == []
    assert m.get_probability([]) == []


def test_single_dict_unchanged():
    """A single dict still returns a single result, not a list."""
    m = _make_model()
    X = {"fa": 0.3, "fb": 0.9, "name": "riou"}
    out = m.get_prediction(X)
    assert not isinstance(out, list)
    assert out in ("A", "B", "C")


def test_pool_size_proportionate():
    m = _make_model()
    cap = os.cpu_count() or 1
    assert m._infer_pool_size(1) == 1
    assert m._infer_pool_size(2) == min(2, cap)
    assert m._infer_pool_size(10_000) == cap
    m._max_workers = 3
    assert m._infer_pool_size(10_000) == 3


# ---------------------------------------------------------------------------
# Stripped serialization
# ---------------------------------------------------------------------------

def test_stripped_roundtrip_predicts(tmp_path):
    m = _make_model()
    full = tmp_path / "full.json"
    strip = tmp_path / "strip.json"
    m.to_json(str(full), stripped=False)
    m.to_json(str(strip), stripped=True)

    # Stripped JSON carries no population rows.
    with open(strip) as f:
        blob = json.load(f)
    assert blob["stripped"] is True
    assert blob["population"] == []
    assert len(blob["layers"]) == len(m.layers)

    ms = Snake(str(strip))
    assert ms._stripped is True
    X = {"fa": 0.3, "fb": 0.9, "name": "riou"}
    # Hot-path methods work and match the full model's clause logic.
    assert ms.get_prediction(X) == m.get_prediction(X)
    assert ms.get_probability(X) == m.get_probability(X)


def test_stripped_blocks_population_methods(tmp_path):
    m = _make_model()
    strip = tmp_path / "strip.json"
    m.to_json(str(strip), stripped=True)
    ms = Snake(str(strip))
    X = {"fa": 0.3, "fb": 0.9, "name": "riou"}
    with pytest.raises(RuntimeError, match="stripped"):
        ms.get_audit(X)
    with pytest.raises(RuntimeError, match="stripped"):
        ms.get_augmented(X)


def test_full_model_still_audits(tmp_path):
    """The default save keeps the population, so audit/augmented still work."""
    m = _make_model()
    full = tmp_path / "full.json"
    m.to_json(str(full))  # stripped defaults to False
    mf = Snake(str(full))
    assert mf._stripped is False
    X = {"fa": 0.3, "fb": 0.9, "name": "riou"}
    assert isinstance(mf.get_audit(X), str)
    assert "Prediction" in mf.get_augmented(X)
