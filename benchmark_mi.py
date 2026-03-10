"""Benchmark: Snake v5.3 (MI + lookahead) vs v5.2.1 baseline.

A/B test: disable MI + set K=1 to simulate v5.2.1 behavior.

Usage:
    python benchmark_mi.py
"""
import random
import math
import time
from algorithmeai import Snake


def _auroc(y_true, y_scores):
    pairs = list(zip(y_true, y_scores))
    pairs.sort(key=lambda x: -x[1])
    tp, fp, prev_score = 0, 0, None
    tps, fps = [0], [0]
    for truth, score in pairs:
        if score != prev_score:
            tps.append(tp)
            fps.append(fp)
            prev_score = score
        if truth == 1:
            tp += 1
        else:
            fp += 1
    tps.append(tp)
    fps.append(fp)
    if tp == 0 or fp == 0:
        return 0.5
    auc = 0.0
    for i in range(1, len(tps)):
        auc += (fps[i] - fps[i-1]) * (tps[i] + tps[i-1]) / 2
    return auc / (tp * fp)


def _multiclass_auroc(y_true, probs, classes):
    aurocs = []
    for cls in classes:
        y_bin = [1 if y == cls else 0 for y in y_true]
        scores = [p.get(cls, 0.0) for p in probs]
        if len(set(y_bin)) < 2:
            continue
        aurocs.append(_auroc(y_bin, scores))
    return sum(aurocs) / len(aurocs) if aurocs else 0.5


def gen_hard_madelon(n=600, m=500, n_informative=20, signal_strength=0.8, seed=42):
    rng = random.Random(seed)
    data = []
    for i in range(n):
        label = "pos" if i < n // 2 else "neg"
        row = {"label": label}
        for j in range(m):
            if j < n_informative:
                base = signal_strength if label == "pos" else -signal_strength
                row[f"f{j}"] = round(base + rng.gauss(0, 3.0), 4)
            else:
                row[f"f{j}"] = round(rng.gauss(0, 3.0), 4)
        data.append(row)
    return data


def load_sklearn_dataset(name):
    try:
        from sklearn import datasets
    except ImportError:
        return None, None
    loaders = {
        "breast_cancer": datasets.load_breast_cancer,
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
    }
    ds = loaders[name]()
    X, y = ds.data, ds.target
    names = [str(n) for n in (ds.feature_names if hasattr(ds, 'feature_names') else [f"f{i}" for i in range(X.shape[1])])]
    data = []
    for i in range(len(y)):
        row = {"label": int(y[i])}
        for j, fname in enumerate(names):
            row[fname] = float(X[i, j])
        data.append(row)
    return data, sorted(set(int(yi) for yi in y))


def evaluate(data, classes, n_layers=15, bucket=250, lookahead=5, noise=0.25, seed=0, disable_mi=False):
    """Train/test split, evaluate. disable_mi=True simulates v5.2.1."""
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split = int(0.8 * len(data))
    train = [data[i] for i in indices[:split]]
    test = [data[i] for i in indices[split:]]

    if disable_mi:
        # Simulate v5.2.1: train normally, then wipe MI and retrain
        # Cleaner: patch _precompute_feature_mi to no-op
        _orig = Snake._precompute_feature_mi
        Snake._precompute_feature_mi = lambda self: setattr(self, '_feature_mi', {})
        try:
            t0 = time.time()
            model = Snake(train, n_layers=n_layers, bucket=bucket, noise=noise,
                          lookahead=lookahead, workers=1)
            train_time = time.time() - t0
        finally:
            Snake._precompute_feature_mi = _orig
    else:
        t0 = time.time()
        model = Snake(train, n_layers=n_layers, bucket=bucket, noise=noise,
                      lookahead=lookahead, workers=1)
        train_time = time.time() - t0

    test_features = [{k: v for k, v in row.items() if k != "label"} for row in test]
    test_labels = [row["label"] for row in test]
    probs = [model.get_probability(x) for x in test_features]

    if len(classes) == 2:
        pos_class = classes[0]
        y_true = [1 if y == pos_class else 0 for y in test_labels]
        y_scores = [p.get(pos_class, 0.0) for p in probs]
        auroc = _auroc(y_true, y_scores)
    else:
        auroc = _multiclass_auroc(test_labels, probs, classes)

    preds = [model.get_prediction(x) for x in test_features]
    accuracy = sum(1 for p, t in zip(preds, test_labels) if p == t) / len(test_labels)

    total_clauses = sum(len(e["clauses"]) for layer in model.layers for e in layer)
    total_lits = sum(len(c[0]) for layer in model.layers for e in layer for c in e["clauses"])

    return auroc, accuracy, train_time, total_clauses, total_lits


def run_config(label, data, classes, n_layers, bucket, lookahead, n_trials, disable_mi=False):
    aurocs, accs, times, clauses, lits = [], [], [], [], []
    for trial in range(n_trials):
        a, c, t, cl, li = evaluate(data, classes, n_layers=n_layers, bucket=bucket,
                                    lookahead=lookahead, seed=trial, disable_mi=disable_mi)
        aurocs.append(a)
        accs.append(c)
        times.append(t)
        clauses.append(cl)
        lits.append(li)
    return {
        "label": label,
        "auroc": sum(aurocs)/len(aurocs),
        "acc": sum(accs)/len(accs),
        "time": sum(times)/len(times),
        "clauses": sum(clauses)/len(clauses),
        "lits": sum(lits)/len(lits),
        "auroc_range": (min(aurocs), max(aurocs)),
    }


def print_row(r, baseline_auroc=None):
    delta = ""
    if baseline_auroc is not None:
        d = r["auroc"] - baseline_auroc
        sign = "+" if d >= 0 else ""
        delta = f"  Δ={sign}{d:.4f}"
    lo, hi = r["auroc_range"]
    print(f"  {r['label']:35s}  AUROC={r['auroc']:.4f} ({lo:.4f}-{hi:.4f})  Acc={r['acc']:.3f}  {r['time']:5.2f}s  {r['clauses']:5.0f}cl  {r['lits']:5.0f}lit{delta}")


def benchmark_ab():
    """A/B: v5.2.1 baseline vs v5.3 (MI + lookahead)."""
    print("=" * 100)
    print("A/B TEST: v5.2.1 (uniform, K=1) vs v5.3 (MI + lookahead)")
    print("=" * 100)

    # --- Hard Madelon ---
    configs = [
        ("20/500 features, signal=0.8", 600, 500, 20, 0.8),
        ("10/500 features, signal=0.8", 600, 500, 10, 0.8),
        ("5/500 features, signal=1.2", 600, 500, 5, 1.2),
        ("20/500 features, signal=0.5 (very hard)", 600, 500, 20, 0.5),
        ("50/500 features, signal=0.8", 600, 500, 50, 0.8),
    ]

    for desc, n, m, n_inf, strength in configs:
        data = gen_hard_madelon(n, m, n_inf, signal_strength=strength, seed=42)
        classes = ["neg", "pos"]
        print(f"\n  --- Hard Madelon: {desc} ---")

        baseline = run_config("v5.2.1 (uniform, K=1)", data, classes, 10, 250, 1, 5, disable_mi=True)
        mi_only = run_config("+ MI weights, K=1", data, classes, 10, 250, 1, 5, disable_mi=False)
        mi_k5 = run_config("+ MI weights, K=5 (default)", data, classes, 10, 250, 5, 5, disable_mi=False)

        print_row(baseline)
        print_row(mi_only, baseline["auroc"])
        print_row(mi_k5, baseline["auroc"])

    # --- Classical ---
    print(f"\n  --- Classical datasets (15 layers) ---")
    for ds_name in ["breast_cancer", "iris", "wine", "digits"]:
        data, classes = load_sklearn_dataset(ds_name)
        if data is None:
            print(f"  {ds_name}: sklearn not available")
            continue
        nf = len(data[0]) - 1

        baseline = run_config(f"{ds_name} v5.2.1", data, classes, 15, 250, 1, 3, disable_mi=True)
        new = run_config(f"{ds_name} v5.3", data, classes, 15, 250, 5, 3, disable_mi=False)
        print_row(baseline)
        print_row(new, baseline["auroc"])
        print()


def benchmark_mi_distribution():
    """Show MI weight distribution on hard Madelon."""
    print()
    print("=" * 100)
    print("MI WEIGHT DISTRIBUTION")
    print("=" * 100)

    for n_inf in [5, 20, 50]:
        data = gen_hard_madelon(600, 500, n_inf, signal_strength=0.8, seed=42)
        random.seed(0)
        model = Snake(data, n_layers=1, bucket=250, lookahead=1)

        mi = model._feature_mi
        informative_mi = [mi.get(i+1, 0) for i in range(n_inf)]
        noise_mi = [mi.get(i+1, 0) for i in range(n_inf, 500)]

        avg_signal = sum(informative_mi) / len(informative_mi)
        avg_noise = sum(noise_mi) / len(noise_mi)
        total_mi = sum(mi.values())
        signal_share = sum(informative_mi) / total_mi * 100 if total_mi > 0 else 0
        uniform_share = n_inf / 500 * 100

        print(f"  {n_inf}/500 informative:  avg_signal={avg_signal:.4f}  avg_noise={avg_noise:.4f}  ratio={avg_signal/max(avg_noise,1e-10):.1f}x  P(signal)={signal_share:.1f}% (was {uniform_share:.1f}%)")


if __name__ == "__main__":
    print("Snake v5.3 Benchmark: MI-Weighted Feature Selection + Lookahead")
    print()
    benchmark_mi_distribution()
    benchmark_ab()
    print("\nDone.")
