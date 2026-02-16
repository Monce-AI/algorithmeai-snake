"""
Benchmark script for Snake classifier.

Trains Snake on classic sklearn datasets + Spaceship Titanic at n_layers=5, 15, 50
and reports accuracy in a markdown table ready to paste into the README.

Requirements (benchmark only — Snake itself has zero deps):
    pip install pandas scikit-learn

For Spaceship Titanic, the script downloads the dataset from GitHub automatically.
The CSV is stored in /tmp and never committed to the repo.
"""
import csv
import os
import random
import sys
import time
import urllib.request

import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from algorithmeai import Snake

SEED = 42
BUCKET = 250
NOISE = 0.25
LAYERS = [5, 15, 50]

SPACESHIP_URL = "https://raw.githubusercontent.com/AmirFARES/Kaggle-Spaceship-Titanic/main/data/train.csv"
SPACESHIP_PATH = "/tmp/spaceship_titanic_train.csv"

DATASETS = [
    ("Iris", load_iris, {"type": "Multi", "text_feats": 0}),
    ("Wine", load_wine, {"type": "Multi", "text_feats": 0}),
    ("Breast Cancer", load_breast_cancer, {"type": "Binary", "text_feats": 0}),
    ("Digits", load_digits, {"type": "Multi", "text_feats": 0}),
]


def load_as_dicts(loader):
    """Load a sklearn dataset as list[dict] with 'target' as first key."""
    ds = loader()
    feature_names = [f.replace(" ", "_").replace("(", "").replace(")", "") for f in ds.feature_names]
    records = []
    for i in range(len(ds.data)):
        row = {"target": str(ds.target[i])}
        for j, name in enumerate(feature_names):
            row[name] = ds.data[i][j]
        records.append(row)
    return records, len(ds.data), len(feature_names), len(set(ds.target))


def load_spaceship_titanic():
    """Download (if needed) and load Spaceship Titanic as list[dict]."""
    if not os.path.exists(SPACESHIP_PATH):
        print(f"  Downloading Spaceship Titanic to {SPACESHIP_PATH} ...")
        urllib.request.urlretrieve(SPACESHIP_URL, SPACESHIP_PATH)
    rows = []
    with open(SPACESHIP_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            cleaned = {"Transported": r["Transported"]}
            for k in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
                cleaned[k] = r[k] if r[k] else "unknown"
            cabin = r.get("Cabin", "")
            if cabin and "/" in cabin:
                parts = cabin.split("/")
                cleaned["CabinDeck"] = parts[0]
                cleaned["CabinSide"] = parts[2] if len(parts) > 2 else "unknown"
            else:
                cleaned["CabinDeck"] = "unknown"
                cleaned["CabinSide"] = "unknown"
            for k in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
                try:
                    cleaned[k] = float(r[k]) if r[k] else 0.0
                except ValueError:
                    cleaned[k] = 0.0
            rows.append(cleaned)
    return rows, len(rows), 12, 2


def accuracy(model, test_data, target_key="target"):
    correct = 0
    for row in test_data:
        features = {k: v for k, v in row.items() if k != target_key}
        pred = model.get_prediction(features)
        if str(pred) == str(row[target_key]):
            correct += 1
    return correct / len(test_data) if test_data else 0


def accuracy_binary(model, test_data, target_key="Transported"):
    """Accuracy for binary True/False targets stored as int 0/1 in the model."""
    correct = 0
    for row in test_data:
        features = {k: v for k, v in row.items() if k != target_key}
        pred = model.get_prediction(features)
        actual_str = row[target_key]
        actual = 1 if actual_str in ("True", "TRUE", "true") else 0
        if pred == actual:
            correct += 1
    return correct / len(test_data) if test_data else 0


def run_benchmarks():
    all_results = {n: [] for n in LAYERS}

    # sklearn datasets
    for ds_name, loader, meta in DATASETS:
        records, n_samples, n_features, n_classes = load_as_dicts(loader)
        train_data, test_data = train_test_split(records, test_size=0.2, random_state=SEED)

        for n_layers in LAYERS:
            t0 = time.time()
            model = Snake(train_data, n_layers=n_layers, bucket=BUCKET, noise=NOISE, vocal=True)
            train_time = time.time() - t0

            train_acc = accuracy(model, train_data)
            test_acc = accuracy(model, test_data)

            t0 = time.time()
            for row in test_data[:50]:
                features = {k: v for k, v in row.items() if k != "target"}
                model.get_prediction(features)
            n_infer = min(50, len(test_data))
            avg_infer = (time.time() - t0) / n_infer * 1000

            all_results[n_layers].append({
                "dataset": ds_name,
                "type": meta["type"],
                "samples": n_samples,
                "features": n_features,
                "text_feats": meta["text_feats"],
                "classes": n_classes,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_time": train_time,
                "avg_infer_ms": avg_infer,
            })
            print(f"  {ds_name} | n_layers={n_layers} | train={train_acc:.1%} test={test_acc:.1%} | {train_time:.1f}s")

    # Spaceship Titanic
    print("\n  --- Spaceship Titanic ---")
    records, n_samples, n_features, n_classes = load_spaceship_titanic()
    random.seed(SEED)
    random.shuffle(records)
    split = int(len(records) * 0.8)
    train_data = records[:split]
    test_data = records[split:]

    for n_layers in LAYERS:
        t0 = time.time()
        model = Snake(train_data, n_layers=n_layers, bucket=BUCKET, noise=NOISE, vocal=True)
        train_time = time.time() - t0

        train_acc = accuracy_binary(model, train_data)
        test_acc = accuracy_binary(model, test_data)

        t0 = time.time()
        for row in test_data[:50]:
            features = {k: v for k, v in row.items() if k != "Transported"}
            model.get_prediction(features)
        n_infer = min(50, len(test_data))
        avg_infer = (time.time() - t0) / n_infer * 1000

        all_results[n_layers].append({
            "dataset": "Spaceship Titanic",
            "type": "Binary",
            "samples": n_samples,
            "features": n_features,
            "text_feats": 6,
            "classes": n_classes,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_time": train_time,
            "avg_infer_ms": avg_infer,
        })
        print(f"  Spaceship Titanic | n_layers={n_layers} | train={train_acc:.1%} test={test_acc:.1%} | {train_time:.1f}s")

    # Print markdown tables
    print("\n\n## Benchmark Results\n")
    for n_layers in LAYERS:
        print(f"### `n_layers={n_layers}`\n")
        print("| Dataset | Type | Samples | Features | Classes | Train Acc | Test Acc | Train Time | Inference |")
        print("|---------|------|---------|----------|---------|-----------|----------|------------|-----------|")
        for r in all_results[n_layers]:
            feat_str = f"{r['features']}"
            print(
                f"| {r['dataset']:<20} "
                f"| {r['type']:<6} "
                f"| {r['samples']:<7} "
                f"| {feat_str:<8} "
                f"| {r['classes']:<7} "
                f"| {r['train_acc']:.1%}     "
                f"| {r['test_acc']:.1%}    "
                f"| {r['train_time']:.1f}s       "
                f"| {r['avg_infer_ms']:.1f}ms     |"
            )
        print()


if __name__ == "__main__":
    run_benchmarks()
