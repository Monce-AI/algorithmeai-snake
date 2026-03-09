"""
Snake v5.2.1 — Profile Empirical Validation
=============================================
8 datasets × 8 profiles. AUROC + optimal accuracy + train time.
Each dataset has a theoretical best profile in oppose_types.snake.
A profile wins if it beats original on AUROC AND optimal accuracy.

Requires: pip install pandas scikit-learn
"""
import os
import random
import sys
import time

import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_20newsgroups,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from algorithmeai import Snake

SEED = 42
LAYERS = 10
BUCKET = 250
NOISE = 0.25
WORKERS = 10
MAX_SAMPLES = 5000

PROFILES = ["original", "balanced", "industrial", "scientific",
            "categorical", "cryptographic", "linguistic"]


class OriginalSnake(Snake):
    def _init_oppose_profile(self):
        self._active_oppose = self.oppose
        self.oppose_profile = "original"


# ---------------------------------------------------------------------------
# Dataset loaders → list[dict]
# ---------------------------------------------------------------------------

def load_sklearn_dicts(loader, target_type="str"):
    ds = loader()
    fnames = [f.replace(" ", "_").replace("(", "").replace(")", "") for f in ds.feature_names]
    data = []
    for i in range(len(ds.data)):
        t = str(ds.target[i]) if target_type == "str" else int(ds.target[i])
        row = {"target": t}
        for j, name in enumerate(fnames):
            row[name] = float(ds.data[i][j])
        data.append(row)
    return data


def load_pima():
    """PIMA Indians Diabetes — 768 rows, 8 numeric, binary. Zeros = missing."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    path = "/tmp/pima.csv"
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(url, path)
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
    df = pd.read_csv(path, names=cols)
    data = []
    for _, row in df.iterrows():
        data.append({
            "target": int(row["Outcome"]),
            "Pregnancies": float(row["Pregnancies"]),
            "Glucose": float(row["Glucose"]),
            "BloodPressure": float(row["BloodPressure"]),
            "SkinThickness": float(row["SkinThickness"]),
            "Insulin": float(row["Insulin"]),
            "BMI": float(row["BMI"]),
            "DiabetesPedigree": float(row["DiabetesPedigree"]),
            "Age": float(row["Age"]),
        })
    return data


def load_adult():
    """Adult Census Income — ~48K rows, 14 mixed, binary (>50K / <=50K)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    path = "/tmp/adult.csv"
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(url, path)
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    df = pd.read_csv(path, names=cols, skipinitialspace=True, na_values="?")
    df = df.dropna()
    # Cap at 5000 for speed
    if len(df) > 5000:
        df = df.sample(5000, random_state=SEED)
    data = []
    num_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    for _, row in df.iterrows():
        d = {"target": row["income"].strip()}
        for c in df.columns:
            if c == "income":
                continue
            if c in num_cols:
                d[c] = float(row[c])
            else:
                d[c] = str(row[c]).strip()
        data.append(d)
    return data


def load_mushroom():
    """Mushroom — 8124 rows, 22 categorical features, binary (edible/poisonous)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    path = "/tmp/mushroom.csv"
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(url, path)
    cols = ["class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
            "gill_attachment", "gill_spacing", "gill_size", "gill_color",
            "stalk_shape", "stalk_root", "stalk_surface_above", "stalk_surface_below",
            "stalk_color_above", "stalk_color_below", "veil_type", "veil_color",
            "ring_number", "ring_type", "spore_print_color", "population", "habitat"]
    df = pd.read_csv(path, names=cols, na_values="?")
    df = df.dropna()
    # Cap at 3000
    if len(df) > 3000:
        df = df.sample(3000, random_state=SEED)
    data = []
    for _, row in df.iterrows():
        d = {"target": row["class"]}
        for c in cols[1:]:
            d[c] = str(row[c])
        data.append(d)
    return data


def load_wine_quality():
    """Wine Quality — 6497 rows (red+white), 11 numeric, multiclass."""
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    path_r, path_w = "/tmp/wine_red.csv", "/tmp/wine_white.csv"
    import urllib.request
    if not os.path.exists(path_r):
        urllib.request.urlretrieve(url_red, path_r)
    if not os.path.exists(path_w):
        urllib.request.urlretrieve(url_white, path_w)
    df_r = pd.read_csv(path_r, sep=";")
    df_w = pd.read_csv(path_w, sep=";")
    df = pd.concat([df_r, df_w], ignore_index=True)
    # Cap at 3000
    if len(df) > 3000:
        df = df.sample(3000, random_state=SEED)
    data = []
    for _, row in df.iterrows():
        d = {"target": str(int(row["quality"]))}
        for c in df.columns:
            if c != "quality":
                d[c.replace(" ", "_")] = float(row[c])
        data.append(d)
    return data


def load_spaceship():
    """Spaceship Titanic — 8693 rows, 12 mixed features, binary."""
    url = "https://raw.githubusercontent.com/AmirFARES/Kaggle-Spaceship-Titanic/main/data/train.csv"
    path = "/tmp/spaceship_titanic_train.csv"
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path)
    df = df.drop(columns=["PassengerId", "Name"])
    df["CabinDeck"] = df["Cabin"].apply(lambda x: str(x).split("/")[0] if pd.notna(x) else "")
    df["CabinSide"] = df["Cabin"].apply(lambda x: str(x).split("/")[-1] if pd.notna(x) else "")
    df = df.drop(columns=["Cabin"])
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in df.columns:
        if c in num_cols:
            df[c] = df[c].fillna(0.0).astype(float)
        else:
            df[c] = df[c].fillna("").astype(str)
    tc = "Transported"
    df = df[[tc] + [c for c in df.columns if c != tc]]
    return df.to_dict("records"), tc


def load_newsgroups():
    """20 Newsgroups subset — 4 categories, raw text, linguistic playground."""
    cats = ["sci.space", "rec.sport.baseball", "talk.politics.guns", "comp.graphics"]
    data_train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
    data = []
    for text, label in zip(data_train.data, data_train.target):
        # Truncate to first 200 chars for speed
        clean = text.replace("\n", " ").strip()[:200]
        if len(clean) > 10:
            data.append({"target": data_train.target_names[label], "text": clean})
    # Cap at 2000
    if len(data) > 2000:
        random.seed(SEED)
        random.shuffle(data)
        data = data[:2000]
    return data


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_dataset(name, data, target_col="target", pos_class=None):
    """Run all profiles on one dataset. Returns list of result dicts."""
    random.seed(SEED)
    if len(data) > MAX_SAMPLES:
        random.shuffle(data)
        data = data[:MAX_SAMPLES]
    train, test = train_test_split(data, test_size=0.2, random_state=SEED)

    n_classes = len(set(r[target_col] for r in data))
    is_binary = n_classes == 2

    # Detect positive class for AUROC
    if is_binary and pos_class is None:
        # Minority class
        from collections import Counter
        tc = Counter(r[target_col] for r in data)
        pos_class = tc.most_common()[-1][0]

    results = []
    for profile in PROFILES:
        cls = OriginalSnake if profile == "original" else Snake
        kw = {} if profile == "original" else {"oppose_profile": profile}

        random.seed(SEED)
        t0 = time.time()
        m = cls(train, target_index=target_col, n_layers=LAYERS, bucket=BUCKET,
                noise=NOISE, workers=WORKERS, **kw)
        train_ms = (time.time() - t0) * 1000

        t0 = time.time()
        preds = []
        for row in test:
            feats = {k: v for k, v in row.items() if k != target_col}
            pb = m.get_probability(feats)
            true_val = row[target_col]
            # Map target type
            if m.targets and type(m.targets[0]) != type(true_val):
                if isinstance(m.targets[0], int):
                    if true_val in (True, "True", "1"):
                        true_val = 1
                    elif true_val in (False, "False", "0"):
                        true_val = 0
                    else:
                        try:
                            true_val = int(true_val)
                        except (ValueError, TypeError):
                            pass
            model_pos = pos_class
            if m.targets and type(m.targets[0]) != type(pos_class):
                if isinstance(m.targets[0], int):
                    if pos_class in (True, "True", "1"):
                        model_pos = 1
                    elif pos_class in (False, "False", "0"):
                        model_pos = 0
            preds.append((pb, true_val, model_pos))
        infer_ms = (time.time() - t0) * 1000

        # Optimal accuracy
        if is_binary:
            scores = [(pb.get(model_pos, 0.5), 1 if tv == model_pos else 0) for pb, tv, model_pos in preds]
            auroc = roc_auc_score([y for _, y in scores], [p for p, _ in scores])
            best_acc = 0
            best_t = 0.5
            for t in [i / 100 for i in range(20, 80)]:
                correct = sum(1 for p, y in scores if (1 if p >= t else 0) == y)
                acc = correct / len(scores) * 100
                if acc > best_acc:
                    best_acc = acc
                    best_t = t
        else:
            # Multiclass: use accuracy directly, AUROC = None
            correct = sum(1 for pb, tv, _ in preds if max(pb, key=pb.get) == tv)
            best_acc = correct / len(preds) * 100
            best_t = None
            auroc = None

        results.append({
            "profile": profile,
            "auroc": auroc,
            "opt_acc": best_acc,
            "opt_t": best_t,
            "train_ms": train_ms,
            "infer_ms": infer_ms,
        })

    return results


def print_results(name, results, n_train, n_test, n_feat, n_class, expected_winner):
    orig = next(r for r in results if r["profile"] == "original")
    best = max(results, key=lambda r: (r["auroc"] or 0, r["opt_acc"]))

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  {n_train} train / {n_test} test / {n_feat} features / {n_class} classes")
    print(f"  Expected winner: {expected_winner}")
    print(f"{'=' * 70}")

    is_binary = results[0]["auroc"] is not None

    if is_binary:
        print(f"{'Profile':<16} {'AUROC':>7} {'OptAcc':>7} {'OptT':>6} {'Train':>8} {'Infer':>8} {'vs orig':>8}")
        print("-" * 62)
        for r in sorted(results, key=lambda x: -(x["auroc"] or 0)):
            delta = (r["auroc"] - orig["auroc"]) * 100 if r["auroc"] and orig["auroc"] else 0
            marker = " *" if r["profile"] == best["profile"] and r["profile"] != "original" else ""
            print(f"{r['profile']:<16} {r['auroc']:>6.4f} {r['opt_acc']:>6.1f}% {r['opt_t']:>5.2f} {r['train_ms']:>7.0f}ms {r['infer_ms']:>7.0f}ms {delta:>+6.1f}pp{marker}")
    else:
        print(f"{'Profile':<16} {'Acc':>7} {'Train':>8} {'Infer':>8} {'vs orig':>8}")
        print("-" * 50)
        for r in sorted(results, key=lambda x: -x["opt_acc"]):
            delta = r["opt_acc"] - orig["opt_acc"]
            marker = " *" if r["profile"] == best["profile"] and r["profile"] != "original" else ""
            print(f"{r['profile']:<16} {r['opt_acc']:>6.1f}% {r['train_ms']:>7.0f}ms {r['infer_ms']:>7.0f}ms {delta:>+6.1f}pp{marker}")

    # Verdict
    winners = [r for r in results if r["profile"] != "original" and r["opt_acc"] > orig["opt_acc"]]
    if is_binary:
        winners = [r for r in results if r["profile"] != "original"
                   and r["opt_acc"] >= orig["opt_acc"] and (r["auroc"] or 0) >= (orig["auroc"] or 0)]
    if winners:
        w = max(winners, key=lambda r: (r["auroc"] or 0, r["opt_acc"]))
        print(f"  >> WINNER: {w['profile']} beats original")
        if w["profile"] == expected_winner:
            print(f"  >> EXPECTED: correct!")
        else:
            print(f"  >> EXPECTED: {expected_winner} (got {w['profile']})")
    else:
        print(f"  >> NO WINNER: original holds")

    return best


def main():
    print("Snake v5.2.1 — Profile Empirical Validation")
    print(f"Config: {LAYERS} layers, bucket={BUCKET}, noise={NOISE}, workers={WORKERS}")
    print()

    all_results = {}

    # 1. PIMA Diabetes
    data = load_pima()
    r = run_dataset("PIMA Diabetes", data, pos_class=1)
    all_results["pima"] = print_results(
        "PIMA DIABETES — 768 rows, 8 numeric, binary (zeros=missing)",
        r, 614, 154, 8, 2, "scientific")

    # 2. Wine Quality
    data = load_wine_quality()
    r = run_dataset("Wine Quality", data)
    all_results["wine_q"] = print_results(
        "WINE QUALITY — 3000 rows, 11 numeric, multiclass (3-9)",
        r, 2400, 600, 11, len(set(d["target"] for d in data)), "scientific")

    # 3. Adult Income
    data = load_adult()
    r = run_dataset("Adult Income", data, pos_class=">50K")
    all_results["adult"] = print_results(
        "ADULT INCOME — 5000 rows, 14 mixed, binary (>50K / <=50K)",
        r, 4000, 1000, 14, 2, "categorical")

    # 4. Mushroom
    data = load_mushroom()
    r = run_dataset("Mushroom", data, pos_class="p")
    all_results["mushroom"] = print_results(
        "MUSHROOM — 3000 rows, 22 categorical, binary (edible/poisonous)",
        r, 2400, 600, 22, 2, "categorical")

    # 5. Breast Cancer
    data = load_sklearn_dicts(load_breast_cancer, target_type="int")
    r = run_dataset("Breast Cancer", data, pos_class=1)
    all_results["bc"] = print_results(
        "BREAST CANCER — 569 rows, 30 numeric, binary",
        r, 455, 114, 30, 2, "scientific")

    # 6. Spaceship Titanic
    sp_data, tc = load_spaceship()
    r = run_dataset("Spaceship Titanic", sp_data, target_col=tc, pos_class="True")
    all_results["spaceship"] = print_results(
        "SPACESHIP TITANIC — 8693 rows, 12 mixed, binary",
        r, 6954, 1739, 12, 2, "industrial")

    # 7. 20 Newsgroups (4 categories)
    data = load_newsgroups()
    r = run_dataset("20 Newsgroups", data)
    all_results["news"] = print_results(
        "20 NEWSGROUPS — 2000 docs, 1 text feature, 4 classes",
        r, 1600, 400, 1, 4, "linguistic")

    # 8. Digits
    data = load_sklearn_dicts(load_digits)
    r = run_dataset("Digits", data)
    all_results["digits"] = print_results(
        "DIGITS — 1797 rows, 64 numeric, 10 classes",
        r, 1437, 360, 64, 10, "scientific")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
