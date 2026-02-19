"""
Titanic dual probability — core vs noise lookalike origin analysis.

Trains Snake on 80% Titanic, evaluates 3 probability signals on 20% holdout:
  P_all   = standard (all lookalikes weighted equally)
  P_core  = only core (c) lookalikes
  P_noise = only noise (n) lookalikes

Reports AUROC for each signal, plus weighted AUROC with tunable (w_c, w_n).
Also computes accuracy at optimal threshold per signal.
"""
import csv
import random
from collections import Counter

random.seed(42)

from algorithmeai import Snake, floatconversion


# ── Helpers ──────────────────────────────────────────────────────────

def auroc(scores, labels, pos_label="1"):
    """Compute AUROC from (score, label) pairs. Works for any binary task."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l == pos_label)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = 0
    fp = 0
    auc = 0.0
    prev_score = None
    prev_tp = 0
    prev_fp = 0
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_tp = tp
            prev_fp = fp
        if label == pos_label:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2
    return auc / (n_pos * n_neg)


def best_accuracy(scores, labels, pos_label="1"):
    """Find threshold that maximizes accuracy. Returns (threshold, accuracy)."""
    thresholds = sorted(set(scores))
    best_t, best_acc = 0.5, 0.0
    for t in thresholds:
        correct = sum(
            1 for s, l in zip(scores, labels)
            if (l == pos_label) == (s >= t)
        )
        acc = correct / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    # Also check threshold above max score (predict all negative)
    correct = sum(1 for l in labels if l != pos_label)
    acc = correct / len(labels)
    if acc > best_acc:
        best_acc = acc
        best_t = max(scores) + 0.01
    return best_t, best_acc


def weighted_prob(lookalikes, target_class, w_c=1, w_n=1):
    """Compute weighted P(target_class) from labeled lookalikes.
    Each lookalike is [global_idx, target_value, condition, origin].
    Returns probability as float."""
    total_weight = 0.0
    class_weight = 0.0
    for la in lookalikes:
        target_val, origin = la[1], la[3]
        w = w_c if origin == "c" else w_n
        total_weight += w
        if str(target_val) == str(target_class):
            class_weight += w
    if total_weight == 0:
        return 0.5  # no lookalikes → uniform
    return class_weight / total_weight


def split_prob(lookalikes, target_class, origin_filter=None):
    """P(target_class) from lookalikes, optionally filtered by origin."""
    filtered = lookalikes if origin_filter is None else [la for la in lookalikes if la[3] == origin_filter]
    if len(filtered) == 0:
        return None  # no signal
    count = sum(1 for la in filtered if str(la[1]) == str(target_class))
    return count / len(filtered)


# ── Load Titanic ─────────────────────────────────────────────────────

NUMERIC = {"PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"}

with open("titanic/train.csv") as f:
    reader = csv.DictReader(f)
    data = []
    for row in reader:
        for k in NUMERIC:
            row[k] = floatconversion(row[k])
        data.append(row)

print(f"Total rows: {len(data)}")

# ── 80/20 split ──
random.shuffle(data)
split = int(len(data) * 0.8)
train = data[:split]
test = data[split:]
print(f"Train: {len(train)}, Test: {len(test)}")

# ── Train with small bucket to force multiple buckets + noise ──
model = Snake(train, target_index="Survived", n_layers=77, bucket=150, noise=0.40, vocal=True, workers=10)

# ── Evaluate holdout ─────────────────────────────────────────────────

pos_label = "1"
scores_all = []
scores_core = []
scores_noise = []
scores_w = []  # weighted
labels = []
divergence_cases = []

W_C = 2  # weight for core lookalikes
W_N = 1  # weight for noise lookalikes

n_pure_core = 0
n_pure_noise = 0
n_mixed = 0
n_empty = 0

for row in test:
    actual = str(row["Survived"])
    X = {k: v for k, v in row.items() if k != "Survived"}

    lookalikes = model.get_lookalikes_labeled(X)
    labels.append(actual)

    p_all = split_prob(lookalikes, pos_label)
    p_core = split_prob(lookalikes, pos_label, origin_filter="c")
    p_noise = split_prob(lookalikes, pos_label, origin_filter="n")
    p_weighted = weighted_prob(lookalikes, pos_label, w_c=W_C, w_n=W_N)

    # Count origin composition
    origins = Counter(la[3] for la in lookalikes)
    n_c = origins.get("c", 0)
    n_n = origins.get("n", 0)
    if n_c == 0 and n_n == 0:
        n_empty += 1
    elif n_n == 0:
        n_pure_core += 1
    elif n_c == 0:
        n_pure_noise += 1
    else:
        n_mixed += 1

    scores_all.append(p_all if p_all is not None else 0.5)
    scores_core.append(p_core if p_core is not None else 0.5)
    scores_noise.append(p_noise if p_noise is not None else 0.5)
    scores_w.append(p_weighted)

    # Divergence: core and noise disagree on majority class
    if p_core is not None and p_noise is not None:
        core_pred = "1" if p_core >= 0.5 else "0"
        noise_pred = "1" if p_noise >= 0.5 else "0"
        if core_pred != noise_pred:
            divergence_cases.append({
                "name": X.get("Name", "?"),
                "actual": actual,
                "p_core": p_core,
                "p_noise": p_noise,
                "p_all": p_all,
                "n_core": n_c,
                "n_noise": n_n,
            })

# ── Results ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("LOOKALIKE ORIGIN COMPOSITION")
print("=" * 70)
print(f"  Pure core (no noise lookalikes): {n_pure_core}")
print(f"  Pure noise (no core lookalikes): {n_pure_noise}")
print(f"  Mixed (both core + noise):       {n_mixed}")
print(f"  Empty (no lookalikes):           {n_empty}")

print("\n" + "=" * 70)
print("AUROC COMPARISON")
print("=" * 70)

auc_all = auroc(scores_all, labels, pos_label)
auc_core = auroc(scores_core, labels, pos_label)
auc_noise = auroc(scores_noise, labels, pos_label)
auc_w = auroc(scores_w, labels, pos_label)

print(f"  P_all   (equal weight):     AUROC = {auc_all:.4f}")
print(f"  P_core  (core only):        AUROC = {auc_core:.4f}")
print(f"  P_noise (noise only):       AUROC = {auc_noise:.4f}")
print(f"  P_weighted (w_c={W_C}, w_n={W_N}): AUROC = {auc_w:.4f}")

print("\n" + "=" * 70)
print("ACCURACY AT OPTIMAL THRESHOLD")
print("=" * 70)

for name, scores in [("P_all", scores_all), ("P_core", scores_core),
                      ("P_noise", scores_noise), (f"P_weighted(w_c={W_C},w_n={W_N})", scores_w)]:
    t, acc = best_accuracy(scores, labels, pos_label)
    print(f"  {name:30s}: acc = {acc:.4f}  (threshold = {t:.3f})")

# ── Divergence cases ─────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print(f"DIVERGENCE CASES (core vs noise disagree): {len(divergence_cases)}")
print("=" * 70)

# Show who wins when they diverge
core_right = 0
noise_right = 0
for d in divergence_cases:
    core_pred = "1" if d["p_core"] >= 0.5 else "0"
    noise_pred = "1" if d["p_noise"] >= 0.5 else "0"
    if core_pred == d["actual"]:
        core_right += 1
    if noise_pred == d["actual"]:
        noise_right += 1

if divergence_cases:
    print(f"  When they disagree ({len(divergence_cases)} cases):")
    print(f"    Core correct:  {core_right}/{len(divergence_cases)} = {core_right/len(divergence_cases):.1%}")
    print(f"    Noise correct: {noise_right}/{len(divergence_cases)} = {noise_right/len(divergence_cases):.1%}")

print(f"\n  First 10 divergence cases:")
for d in divergence_cases[:10]:
    marker = "✓" if ("1" if d["p_core"] >= 0.5 else "0") == d["actual"] else "✗"
    print(f"    {d['name'][:40]:40s}  actual={d['actual']}  "
          f"P_core={d['p_core']:.2f}({d['n_core']})"
          f"  P_noise={d['p_noise']:.2f}({d['n_noise']})"
          f"  P_all={d['p_all']:.2f}  core_correct={marker}")

# ── Weight sweep ─────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("WEIGHT SWEEP (w_c, w_n) — AUROC and best accuracy")
print("=" * 70)

for w_c, w_n in [(1, 0), (2, 1), (3, 1), (5, 1), (1, 1), (1, 2), (1, 3), (0, 1)]:
    sw = [weighted_prob(model.get_lookalikes_labeled({k: v for k, v in row.items() if k != "Survived"}),
                        pos_label, w_c=w_c, w_n=w_n) for row in test]
    a = auroc(sw, labels, pos_label)
    _, acc = best_accuracy(sw, labels, pos_label)
    print(f"  w_c={w_c}, w_n={w_n}:  AUROC={a:.4f}  best_acc={acc:.4f}")
