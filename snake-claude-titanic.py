"""
snake-claude-titanic.py — Snake + FN meta-classifier for Titanic (Kaggle)

Usage:
    python snake-claude-titanic.py train.csv test.csv

If test.csv contains the target column (Survived) → validation mode:
    - 5 random 80/20 splits with FN detection + correction
    - Reports base accuracy, corrected accuracy, AUROC, FN detection AUROC

If test.csv has no target column → submission mode:
    - Trains on full train, builds FN detector via 5-fold cross-val
    - Generates submission.csv ready for Kaggle

Author: Charles Dana / Monce SAS — powered by Snake v4.4.2
"""

import sys
import random
import csv
from collections import defaultdict
from algorithmeai import Snake

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_COL    = "Survived"
TARGET_IDX    = 1
N_LAYERS      = 12
BUCKET        = 400
FN_LAYERS     = 30
FN_BUCKET     = 80
FN_THRESHOLD  = 0.60
N_SPLITS      = 5
ID_COL        = "PassengerId"
# ────────────────────────────────────────────────────────────────────────────

BANNER = r"""
   ____              __           _______ __              _
  / __/__  ___ _____/ /_____     /_  __(_) /____ ____  (_)___
 _\ \/ _ \/ _ `/  '_/ -_)_(_)    / / / / __/ _ `/ _ \/ / __/
/___/_//_/\_,_/_/\_\\__/(_)     /_/ /_/\__/\_,_/_//_/_/\__/

          Snake v4.4.2 + FN Meta-Classifier
          Charles Dana / Monce SAS
"""

BAR_WIDTH = 20


def bar(ratio, width=BAR_WIDTH):
    filled = int(ratio * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def load_data(csv_path):
    """Load CSV via Snake for proper type detection."""
    probe = Snake(csv_path, target_index=TARGET_IDX, n_layers=1, bucket=999, vocal=False)
    return probe.population, probe


def has_ground_truth(test_path):
    """Check if test CSV contains the target column."""
    with open(test_path) as f:
        header = f.readline().strip().split(",")
    return TARGET_COL in header


def build_fn_profiles(snake_model, test_rows):
    """Predict on test_rows, return (profiles_for_predicted_died, all_preds)."""
    profiles = []
    all_preds = []
    target = snake_model.target

    for row in test_rows:
        features = {k: v for k, v in row.items() if k != target}
        prob = snake_model.get_probability(features)
        pred = int(snake_model.get_prediction(features))
        true = int(row[target]) if target in row else None
        all_preds.append((true, pred, prob, features, row))

        if pred == 0:
            label = None
            if true is not None:
                label = "FN" if true == 1 else "TN"
            profiles.append({
                "is_fn": label,
                "Sex": row.get("Sex", ""),
                "Pclass": row.get("Pclass", 0),
                "Age": row.get("Age", 0),
                "SibSp": row.get("SibSp", 0),
                "Parch": row.get("Parch", 0),
                "Fare": row.get("Fare", 0),
                "Embarked": row.get("Embarked", ""),
                "P_survived": round(prob.get(1, 0.0), 4),
                "P_died": round(prob.get(0, 0.0), 4),
            })

    return profiles, all_preds


def apply_fn_correction(fn_model, all_preds, profiles):
    """Apply FN detector to flip predictions. Returns corrected pred list."""
    corrected = []
    profile_idx = 0

    for true, pred, prob, features, row in all_preds:
        if pred == 0:
            p = profiles[profile_idx]
            fn_features = {k: v for k, v in p.items() if k != "is_fn"}
            fn_prob = fn_model.get_probability(fn_features)
            fn_conf = fn_prob.get("FN", 0.0)
            if fn_model.get_prediction(fn_features) == "FN" and fn_conf >= FN_THRESHOLD:
                corrected.append((true, 1, prob, features, row))  # flip 0→1
            else:
                corrected.append((true, pred, prob, features, row))
            profile_idx += 1
        else:
            corrected.append((true, pred, prob, features, row))

    return corrected


def compute_metrics(preds):
    """Compute accuracy, FN, FP from (true, pred, ...) tuples."""
    correct = sum(1 for t, p, *_ in preds if t == p)
    total = len(preds)
    fn = sum(1 for t, p, *_ in preds if t == 1 and p == 0)
    fp = sum(1 for t, p, *_ in preds if t == 0 and p == 1)
    return correct, total, fn, fp


def compute_auroc(preds):
    """Compute AUROC from (true, pred, prob, ...) tuples."""
    y_true = [t for t, p, prob, *_ in preds]
    y_score = [prob.get(1, 0.0) for t, p, prob, *_ in preds]
    # Manual AUROC (no sklearn dependency)
    pairs = list(zip(y_true, y_score))
    n_pos = sum(1 for t, _ in pairs if t == 1)
    n_neg = sum(1 for t, _ in pairs if t == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    concordant = 0
    for t1, s1 in pairs:
        for t2, s2 in pairs:
            if t1 == 1 and t2 == 0:
                if s1 > s2:
                    concordant += 1
                elif s1 == s2:
                    concordant += 0.5
    return concordant / (n_pos * n_neg)


def fn_auroc(fn_model, profiles):
    """Compute AUROC of FN detector on labeled profiles."""
    y_true, y_score = [], []
    for p in profiles:
        if p["is_fn"] is None:
            continue
        features = {k: v for k, v in p.items() if k != "is_fn"}
        prob = fn_model.get_probability(features)
        y_true.append(1 if p["is_fn"] == "FN" else 0)
        y_score.append(prob.get("FN", 0.0))
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    concordant = 0
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if y_true[i] == 1 and y_true[j] == 0:
                if y_score[i] > y_score[j]:
                    concordant += 1
                elif y_score[i] == y_score[j]:
                    concordant += 0.5
    return concordant / (n_pos * n_neg)


# ── Validation Mode ─────────────────────────────────────────────────────────

def run_validation(train_path, test_path):
    print("  Mode: VALIDATION (test has ground truth)")
    print("  Splits: 5 random 80/20 on train")
    print(f"  Snake: n_layers={N_LAYERS}, bucket={BUCKET}")
    print(f"  FN detector: n_layers={FN_LAYERS}, bucket={FN_BUCKET}, threshold={FN_THRESHOLD}")
    print()

    train_pop, _ = load_data(train_path)
    print(f"  Train population: {len(train_pop)} samples")
    print()

    # Phase 1: 5 splits, collect profiles
    print("  PHASE 1: Training main classifiers on 5 splits")
    print("  " + "-" * 50)
    split_profiles = {}
    split_all_preds = {}

    for i in range(N_SPLITS):
        random.seed(i * 111)
        indices = list(range(len(train_pop)))
        random.shuffle(indices)
        cut = int(0.8 * len(indices))
        tr = [train_pop[j] for j in indices[:cut]]
        te = [train_pop[j] for j in indices[cut:]]

        random.seed(i * 111)
        s = Snake(tr, target_index=TARGET_COL, n_layers=N_LAYERS, bucket=BUCKET, vocal=False)

        profiles, all_preds = build_fn_profiles(s, te)
        split_profiles[i] = profiles
        split_all_preds[i] = all_preds

        c, t, fn, fp = compute_metrics(all_preds)
        n_fn = sum(1 for p in profiles if p["is_fn"] == "FN")
        n_tn = sum(1 for p in profiles if p["is_fn"] == "TN")
        print(f"    Split {i}: {c}/{t} = {c/t:.4f}  (FN={n_fn}, TN={n_tn}, FP={fp})")

    # Phase 2: Leave-one-out FN detection
    print()
    print("  PHASE 2: FN meta-classifier (leave-one-out)")
    print("  " + "-" * 50)
    print()

    results = []
    for test_split in range(N_SPLITS):
        # Train FN detector on other 4 splits
        fn_train = []
        for i in range(N_SPLITS):
            if i == test_split:
                continue
            fn_train.extend([p for p in split_profiles[i] if p["is_fn"] is not None])

        random.seed(42)
        fn_model = Snake(fn_train, target_index="is_fn", n_layers=FN_LAYERS, bucket=FN_BUCKET, vocal=False)

        # Apply correction
        corrected = apply_fn_correction(fn_model, split_all_preds[test_split], split_profiles[test_split])

        base_c, total, base_fn, base_fp = compute_metrics(split_all_preds[test_split])
        corr_c, _, corr_fn, corr_fp = compute_metrics(corrected)

        base_acc = base_c / total
        corr_acc = corr_c / total
        delta = corr_acc - base_acc

        # AUROC on main classifier
        main_auc = compute_auroc(split_all_preds[test_split])

        # AUROC on FN detector
        fn_auc = fn_auroc(fn_model, split_profiles[test_split])

        flips = sum(1 for (t1, p1, *_), (t2, p2, *_) in zip(split_all_preds[test_split], corrected) if p1 != p2)
        good_flips = sum(1 for (t1, p1, *_), (t2, p2, *_) in zip(split_all_preds[test_split], corrected) if p1 != p2 and t2 == p2)

        results.append({
            "base_acc": base_acc, "corr_acc": corr_acc, "delta": delta,
            "main_auc": main_auc, "fn_auc": fn_auc,
            "base_fn": base_fn, "corr_fn": corr_fn,
            "base_fp": base_fp, "corr_fp": corr_fp,
            "flips": flips, "good_flips": good_flips,
        })

        print(f"    Split {test_split}:")
        print(f"      Base:      {base_c}/{total} = {base_acc:.4f}  {bar(base_acc)}")
        print(f"      Corrected: {corr_c}/{total} = {corr_acc:.4f}  {bar(corr_acc)}  ({delta*100:+.2f}pp)")
        print(f"      Flips: {flips} ({good_flips} correct)")
        print(f"      FN: {base_fn} -> {corr_fn}  |  FP: {base_fp} -> {corr_fp}")
        print(f"      Main AUROC: {main_auc:.4f}  |  FN detect AUROC: {fn_auc:.4f}")
        print()

    # Summary
    avg_base = sum(r["base_acc"] for r in results) / N_SPLITS
    avg_corr = sum(r["corr_acc"] for r in results) / N_SPLITS
    avg_delta = sum(r["delta"] for r in results) / N_SPLITS
    avg_main_auc = sum(r["main_auc"] for r in results) / N_SPLITS
    avg_fn_auc = sum(r["fn_auc"] for r in results) / N_SPLITS
    min_corr = min(r["corr_acc"] for r in results)
    max_corr = max(r["corr_acc"] for r in results)

    print("  " + "=" * 56)
    print("  RESULTS")
    print("  " + "=" * 56)
    print()
    print(f"    Avg base accuracy:       {avg_base:.4f}")
    print(f"    Avg corrected accuracy:  {avg_corr:.4f}  ({avg_delta*100:+.2f}pp)")
    print(f"    Range:                   [{min_corr:.4f} — {max_corr:.4f}]")
    print(f"    Avg main AUROC:          {avg_main_auc:.4f}")
    print(f"    Avg FN detection AUROC:  {avg_fn_auc:.4f}")
    print()


# ── Submission Mode ──────────────────────────────────────────────────────────

def run_submission(train_path, test_path):
    print("  Mode: SUBMISSION (no ground truth in test)")
    print(f"  Snake: n_layers={N_LAYERS}, bucket={BUCKET}")
    print(f"  FN detector: n_layers={FN_LAYERS}, bucket={FN_BUCKET}, threshold={FN_THRESHOLD}")
    print()

    train_pop, _ = load_data(train_path)
    print(f"  Train population: {len(train_pop)} samples")

    # Phase 1: 5-fold cross-val on train to collect FN profiles
    print()
    print("  PHASE 1: 5-fold cross-val for FN profile collection")
    print("  " + "-" * 50)

    random.seed(42)
    indices = list(range(len(train_pop)))
    random.shuffle(indices)
    fold_size = len(indices) // N_SPLITS
    folds = []
    for i in range(N_SPLITS):
        start = i * fold_size
        end = start + fold_size if i < N_SPLITS - 1 else len(indices)
        folds.append(indices[start:end])

    all_fn_profiles = []
    for fold_i in range(N_SPLITS):
        tr_idx = [j for i, fold in enumerate(folds) if i != fold_i for j in fold]
        te_idx = folds[fold_i]
        tr = [train_pop[j] for j in tr_idx]
        te = [train_pop[j] for j in te_idx]

        random.seed(fold_i * 111)
        s = Snake(tr, target_index=TARGET_COL, n_layers=N_LAYERS, bucket=BUCKET, vocal=False)

        profiles, all_preds = build_fn_profiles(s, te)
        labeled = [p for p in profiles if p["is_fn"] is not None]
        n_fn = sum(1 for p in labeled if p["is_fn"] == "FN")
        n_tn = sum(1 for p in labeled if p["is_fn"] == "TN")
        all_fn_profiles.extend(labeled)

        c, t, fn, fp = compute_metrics(all_preds)
        print(f"    Fold {fold_i}: {c}/{t} = {c/t:.4f}  (FN={n_fn}, TN={n_tn})")

    print(f"\n    Total FN profiles: {len(all_fn_profiles)} "
          f"(FN={sum(1 for p in all_fn_profiles if p['is_fn']=='FN')}, "
          f"TN={sum(1 for p in all_fn_profiles if p['is_fn']=='TN')})")

    # Phase 2: Train FN detector on all profiles
    print()
    print("  PHASE 2: Training FN detector")
    print("  " + "-" * 50)

    random.seed(42)
    fn_model = Snake(all_fn_profiles, target_index="is_fn", n_layers=FN_LAYERS, bucket=FN_BUCKET, vocal=False)
    print(f"    FN detector trained: {len(fn_model.population)} samples, {len(fn_model.layers)} layers")

    # Phase 3: Train final model on full train, predict test
    print()
    print("  PHASE 3: Final model + submission")
    print("  " + "-" * 50)

    random.seed(42)
    final_model = Snake(train_path, target_index=TARGET_IDX, n_layers=N_LAYERS, bucket=BUCKET, vocal=False)
    print(f"    Final model: {len(final_model.population)} samples, {len(final_model.layers)} layers")

    # Load test via make_population
    test_pop = final_model.make_population(test_path)
    print(f"    Test samples: {len(test_pop)}")

    # Predict
    predictions = []
    flips = 0
    for row in test_pop:
        features = {k: v for k, v in row.items() if k != final_model.target}
        prob = final_model.get_probability(features)
        pred = int(final_model.get_prediction(features))
        pid = int(row[ID_COL])

        # FN correction on predicted-died
        if pred == 0:
            fn_profile = {
                "Sex": row.get("Sex", ""),
                "Pclass": row.get("Pclass", 0),
                "Age": row.get("Age", 0),
                "SibSp": row.get("SibSp", 0),
                "Parch": row.get("Parch", 0),
                "Fare": row.get("Fare", 0),
                "Embarked": row.get("Embarked", ""),
                "P_survived": round(prob.get(1, 0.0), 4),
                "P_died": round(prob.get(0, 0.0), 4),
            }
            fn_prob = fn_model.get_probability(fn_profile)
            fn_pred = fn_model.get_prediction(fn_profile)
            if fn_pred == "FN" and fn_prob.get("FN", 0.0) >= FN_THRESHOLD:
                pred = 1
                flips += 1

        predictions.append((pid, pred))

    survived = sum(1 for _, p in predictions if p == 1)
    died = sum(1 for _, p in predictions if p == 0)
    print(f"\n    Predictions: {survived} survived, {died} died")
    print(f"    FN corrections (flips 0->1): {flips}")

    # Write submission
    out_path = "submission.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ID_COL, TARGET_COL])
        for pid, pred in sorted(predictions):
            writer.writerow([pid, pred])

    print(f"\n    Submission written: {out_path}")
    print(f"    Rows: {len(predictions)}")

    # Distribution bar
    surv_pct = survived / len(predictions) * 100
    died_pct = died / len(predictions) * 100
    print()
    print(f"    Survived  {surv_pct:5.1f}%  {bar(survived / len(predictions))}")
    print(f"    Died      {died_pct:5.1f}%  {bar(died / len(predictions))}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python snake-claude-titanic.py train.csv test.csv")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    print(BANNER)
    print("=" * 60)

    gt = has_ground_truth(test_path)

    if gt:
        run_validation(train_path, test_path)
    else:
        run_submission(train_path, test_path)

    print("=" * 60)
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
