"""Tests for get_synthetic — the v5.4.8 synthetic audit.

The deterministic local path (interpret=False) is fully tested offline: it must
never need the network and must stay honest (noise -> is_noise, signal -> strong,
labels -> held-out accuracy/AUROC without any retrain).

The cloud narration path (interpret=True) needs the monceai SDK and a live
backend, so it is exercised only when monceai is importable, and network
failures are tolerated (skip, not fail).
"""
import math
import random

import pytest

from algorithmeai import Snake


def _signal_data(n=400, seed=1):
    """label is a deterministic function of a,b -> Snake must find strong signal."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.gauss(0, 1)
        b = rng.gauss(0, 1)
        logit = 1.8 * a - 1.3 * b
        p = 1 / (1 + math.exp(-logit))
        rows.append({"a": round(a, 3), "b": round(b, 3),
                     "c": round(rng.gauss(0, 1), 3),
                     "label": 1 if rng.random() < p else 0})
    return rows


def _noise_data(n=400, seed=2):
    """label is an independent coin flip -> Snake must report is_noise=True."""
    rng = random.Random(seed)
    return [{"a": round(rng.gauss(0, 1), 3), "b": round(rng.gauss(0, 1), 3),
             "c": round(rng.gauss(0, 1), 3), "label": rng.randint(0, 1)}
            for _ in range(n)]


def _split(rows, frac=0.8, seed=7):
    r = list(rows)
    random.Random(seed).shuffle(r)
    cut = int(frac * len(r))
    return r[:cut], r[cut:]


# --------------------------------------------------------------------------
# Deterministic local path (interpret=False) — offline, the contract that
# the public free tier relies on.
# --------------------------------------------------------------------------

def test_local_path_needs_no_network():
    """interpret=False returns a summary dict with no import of monceai."""
    tr, te = _split(_signal_data())
    m = Snake(tr, target_index="label", n_layers=10, bucket=200,
              oppose_profile="scientific", vocal=False)
    out = m.get_synthetic(te, interpret=False)
    assert set(out.keys()) == {"summary"}
    s = out["summary"]
    # shape contract
    for key in ("n_points", "n_classes", "task", "signal_strength", "is_noise",
                "top_features_mi", "true_base_rate_pct", "calibration_gap_pts"):
        assert key in s
    assert s["n_points"] == len(te)
    assert s["task"] == "binary"
    assert s["n_classes"] == 2


def test_signal_data_is_not_flagged_as_noise():
    tr, te = _split(_signal_data())
    m = Snake(tr, target_index="label", n_layers=12, bucket=200,
              oppose_profile="scientific", vocal=False)
    s = m.get_synthetic(te, interpret=False)["summary"]
    assert s["is_noise"] is False
    assert s["signal_strength"] in ("weak", "moderate", "strong")
    # labels were present -> held-out metrics computed without retrain
    assert s["n_labeled"] == len(te)
    assert s["holdout_accuracy_pct"] is not None
    assert s["holdout_auroc"] is not None
    assert s["holdout_auroc"] > 0.65  # real signal beats chance comfortably


def test_random_data_is_flagged_as_noise():
    """The honesty property: pure noise must not manufacture signal."""
    tr, te = _split(_noise_data())
    m = Snake(tr, target_index="label", n_layers=12, bucket=200,
              oppose_profile="scientific", vocal=False)
    s = m.get_synthetic(te, interpret=False)["summary"]
    assert s["is_noise"] is True
    assert s["signal_strength"] == "none"


def test_unlabeled_points_skip_holdout_metrics():
    """Without the target key in X, no accuracy/AUROC — but everything else works."""
    tr, te = _split(_signal_data())
    te_unlabeled = [{k: v for k, v in row.items() if k != "label"} for row in te]
    m = Snake(tr, target_index="label", n_layers=10, bucket=200,
              oppose_profile="scientific", vocal=False)
    s = m.get_synthetic(te_unlabeled, interpret=False)["summary"]
    assert s["n_labeled"] == 0
    assert s["holdout_accuracy_pct"] is None
    assert s["holdout_auroc"] is None
    # MI / noise detector still available (computed from training, not the batch)
    assert s["signal_strength"] in ("none", "weak", "moderate", "strong")


def test_single_dict_is_accepted():
    """Polymorphic-friendly: a lone dict is treated as a batch of one."""
    tr, _ = _split(_signal_data())
    m = Snake(tr, target_index="label", n_layers=8, bucket=200,
              oppose_profile="scientific", vocal=False)
    s = m.get_synthetic({"a": 0.5, "b": -0.3, "c": 0.1}, interpret=False)["summary"]
    assert s["n_points"] == 1


def test_calibration_gap_is_signed_number():
    tr, te = _split(_signal_data())
    m = Snake(tr, target_index="label", n_layers=10, bucket=200,
              oppose_profile="scientific", vocal=False)
    s = m.get_synthetic(te, interpret=False)["summary"]
    assert isinstance(s["calibration_gap_pts"], (int, float))


# --------------------------------------------------------------------------
# Cloud narration path (interpret=True) — only when monceai is available.
# --------------------------------------------------------------------------

def test_interpret_true_without_monceai_gives_clear_error():
    """If monceai is missing, the error must point at the install + the
    interpret=False escape hatch — never a bare ImportError."""
    pytest.importorskip  # noqa: keep import-time symbol
    try:
        import monceai  # noqa: F401
        have_monceai = True
    except ImportError:
        have_monceai = False
    if have_monceai:
        pytest.skip("monceai installed — cannot test the missing-dep message here")
    tr, te = _split(_signal_data(n=120))
    m = Snake(tr, target_index="label", n_layers=6, bucket=200,
              oppose_profile="scientific", vocal=False)
    with pytest.raises(RuntimeError) as exc:
        m.get_synthetic(te)
    msg = str(exc.value)
    assert "monceai" in msg
    assert "interpret=False" in msg


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("monceai") is None,
    reason="monceai SDK not installed",
)
def test_cloud_narration_shape_when_available():
    """Best-effort: if monceai + network are up, narration has the 3 keys.
    Network failures are tolerated (the deterministic summary is always there)."""
    tr, te = _split(_signal_data(n=160))
    m = Snake(tr, target_index="label", n_layers=8, bucket=200,
              oppose_profile="scientific", vocal=False)
    try:
        out = m.get_synthetic(te)
    except Exception as e:  # network/backend hiccup — not a code failure
        pytest.skip(f"cloud backend unavailable: {e}")
    assert "summary" in out
    for key in ("hypothese", "experience", "resultat"):
        assert key in out
