"""
Benchmark script for Snake classifier.

Trains Snake on classic sklearn datasets at n_layers=5, 15, 50 and reports
accuracy in a markdown table ready to paste into the README.

Requirements (benchmark only — Snake itself has zero deps):
    pip install pandas scikit-learn
"""
import sys
import time

import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from algorithmeai import Snake

SEED = 42
BUCKET = 250
NOISE = 0.25
LAYERS = [5, 15, 50]

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


def accuracy(model, test_data, target_key="target"):
    correct = 0
    for row in test_data:
        features = {k: v for k, v in row.items() if k != target_key}
        pred = model.get_prediction(features)
        if str(pred) == str(row[target_key]):
            correct += 1
    return correct / len(test_data) if test_data else 0


def run_benchmarks():
    # Header per n_layers configuration
    all_results = {n: [] for n in LAYERS}

    for ds_name, loader, meta in DATASETS:
        records, n_samples, n_features, n_classes = load_as_dicts(loader)
        train_data, test_data = train_test_split(records, test_size=0.2, random_state=SEED)

        for n_layers in LAYERS:
            t0 = time.time()
            model = Snake(train_data, n_layers=n_layers, bucket=BUCKET, noise=NOISE, vocal=True)
            train_time = time.time() - t0

            train_acc = accuracy(model, train_data)
            test_acc = accuracy(model, test_data)

            # Average inference time
            t0 = time.time()
            for row in test_data[:50]:
                features = {k: v for k, v in row.items() if k != "target"}
                model.get_prediction(features)
            n_infer = min(50, len(test_data))
            avg_infer = (time.time() - t0) / n_infer * 1000  # ms

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

    # Print markdown tables
    print("\n\n## Benchmark Results\n")
    for n_layers in LAYERS:
        print(f"### `n_layers={n_layers}`\n")
        print("| Dataset | Type | Samples | Features | Classes | Train Acc | Test Acc | Train Time | Inference |")
        print("|---------|------|---------|----------|---------|-----------|----------|------------|-----------|")
        for r in all_results[n_layers]:
            feat_str = f"{r['features']}"
            print(
                f"| {r['dataset']:<15} "
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
