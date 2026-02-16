
[![GitHub](https://img.shields.io/github/stars/Monce-AI/algorithmeai-snake?style=social)](https://github.com/Monce-AI/algorithmeai-snake)
[![GitHub last commit](https://img.shields.io/github/last-commit/Monce-AI/algorithmeai-snake?logo=github)](https://github.com/Monce-AI/algorithmeai-snake/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/Monce-AI/algorithmeai-snake)](https://github.com/Monce-AI/algorithmeai-snake)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAxTDMgNXY2YzcgNCA4LjUgOC40IDkgMTIuOEM5LjUgMjAuNCA4IDE2IDggMTFWNmw0LTIuNUwxNiA2djVjMCA1LTEuNSA5LjQtNCAxa)](LICENSE)
[![Version](https://img.shields.io/badge/v4.3.2-SAT_Bucketed-blueviolet.svg?logo=semanticrelease)](https://github.com/Monce-AI/algorithmeai-snake)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg?logo=githubactions&logoColor=white)](#)

[![Production](https://img.shields.io/badge/Production-Live_on_AWS-FF9900.svg?logo=amazonaws&logoColor=white)](https://snake.aws.monce.ai)
[![API](https://img.shields.io/badge/API-snake.aws.monce.ai-009688.svg?logo=fastapi&logoColor=white)](https://snake.aws.monce.ai/health)
[![Accuracy](https://img.shields.io/badge/Train_Accuracy-100%25-brightgreen.svg?logo=target)](#benchmarks)
[![Throughput](https://img.shields.io/badge/Throughput-201_qps-blue.svg?logo=speedtest)](#)

[![Algorithm](https://img.shields.io/badge/Algorithm-SAT--based_Lookalikes-7B2D8B.svg?logo=probot&logoColor=white)](#what-is-snake)
[![XAI](https://img.shields.io/badge/XAI-Fully_Explainable-FF6F00.svg?logo=opensourceinitiative&logoColor=white)](#prediction-api)
[![Complexity](https://img.shields.io/badge/Complexity-O(n·log(n)·m·b²)-lightgrey.svg?logo=wolframmathematica)](#architecture)
[![Architecture](https://img.shields.io/badge/Cascade-Snake_→_Fuzzy_→_LLM-E91E63.svg?logo=stackblitz&logoColor=white)](#architecture)

**Author:** Charles Dana · [Monce SAS](https://monce.ai)

# Snake

SAT-based explainable multiclass classifier. Zero dependencies. Pure Python.

## What Is Snake

Snake is a **SAT-based lookalike voting classifier**. For each prediction, it finds training samples that "look alike" via Boolean clause matching, then votes by their labels. The result: a fully explainable classifier where every prediction comes with a human-readable audit trail.

```
Input X  →  Match SAT clauses  →  Find lookalikes  →  Vote  →  Prediction + Audit
```

> **Predicted outcome:** `setosa` (93.3%)
> Because: `"petal_length" <= 2.45` AND `"petal_width" <= 0.8`
> Matched 15 lookalikes, all class `setosa`.

## Install

```bash
pip install git+https://github.com/Monce-AI/algorithmeai-snake.git
```

Dev install:
```bash
git clone https://github.com/Monce-AI/algorithmeai-snake.git
cd algorithmeai-snake
pip install -e .
```

Python 3.9+. Zero dependencies — uses only the standard library.

## Quick Start

```python
from algorithmeai import Snake

# Train from a list of dicts (production pattern)
data = [
    {"species": "setosa",     "petal_length": 1.4, "petal_width": 0.2},
    {"species": "setosa",     "petal_length": 1.3, "petal_width": 0.3},
    {"species": "versicolor", "petal_length": 4.5, "petal_width": 1.5},
    {"species": "versicolor", "petal_length": 4.1, "petal_width": 1.3},
    {"species": "virginica",  "petal_length": 5.2, "petal_width": 2.0},
    {"species": "virginica",  "petal_length": 5.0, "petal_width": 1.9},
    # ... more rows
]

model = Snake(data, n_layers=5, bucket=250)

# Predict
X = {"petal_length": 4.3, "petal_width": 1.4}
print(model.get_prediction(X))    # "versicolor"
print(model.get_probability(X))   # {"setosa": 0.0, "versicolor": 0.87, "virginica": 0.13}
print(model.get_audit(X))         # Full reasoning trace

# Save & reload
model.to_json("model.json")
model = Snake("model.json")       # Auto-detected by .json extension
print(model.get_prediction(X))    # Same result
```

## Input Formats

Snake accepts five input formats. The first key/column is the target by default.

| Format | Example | Notes |
|--------|---------|-------|
| List of dicts | `Snake([{"label": "A", ...}])` | Production pattern. First key = target |
| CSV file | `Snake("data.csv", target_index=3)` | Pandas-formatted CSV |
| DataFrame | `Snake(df, target_index="species")` | Duck-typed — no pandas dependency |
| List of tuples | `Snake([("cat", 4, "small"), ...])` | First element = target, auto-headers |
| List of scalars | `Snake(["apple", "banana", ...])` | Self-classing, dedupes to unique |

**List of dicts** (recommended):
```python
model = Snake([
    {"survived": 1, "class": 3, "sex": "male",   "age": 22},
    {"survived": 0, "class": 1, "sex": "female", "age": 38},
])
```

**CSV file:**
```python
model = Snake("titanic.csv", target_index=0)
```

**DataFrame:**
```python
model = Snake(df, target_index="survived")
```

**List of tuples:**
```python
model = Snake([("cat", 4, "small"), ("dog", 40, "large"), ("cat", 5, "small")])
```

**List of scalars** (self-classing — useful for synonym deduplication):
```python
model = Snake(["44.2 LowE", "44.2 bronze", "Float 4mm clair"])
```

**Complex targets** (dict/list values as targets):
```python
data = [
    {"label": {"color": "red", "size": "big"}, "feature": "round"},
    {"label": {"color": "blue", "size": "small"}, "feature": "square"},
]
model = Snake(data, n_layers=5)
pred = model.get_prediction({"feature": "round"})  # returns {"color": "red", "size": "big"}
```

## Constructor Reference

```python
Snake(Knowledge, target_index=0, excluded_features_index=(), n_layers=5, bucket=250, noise=0.25, vocal=False, saved=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Knowledge` | str / list / DataFrame | — | CSV path, JSON model path, list of dicts/tuples/scalars, or DataFrame |
| `target_index` | int / str | `0` | Target column index or name |
| `excluded_features_index` | tuple/list | `()` | Column indices to exclude from training |
| `n_layers` | int | `5` | Number of SAT layers to build (more = more accurate, slower) |
| `bucket` | int | `250` | Max samples per bucket before splitting |
| `noise` | float | `0.25` | Cross-bucket noise ratio for regularization |
| `vocal` | bool | `False` | Print training progress |
| `saved` | bool | `False` | Auto-save model after training (CSV flow only) |

## Prediction API

| Method | Returns | Description |
|--------|---------|-------------|
| `get_prediction(X)` | value | Most probable class |
| `get_probability(X)` | dict | `{class: probability}` for all classes |
| `get_lookalikes(X)` | list | `[[index, class, condition], ...]` matched training samples |
| `get_augmented(X)` | dict | Input enriched with Lookalikes, Probability, Prediction, Audit |
| `get_audit(X)` | str | Full human-readable reasoning trace |

```python
X = {"petal_length": 4.3, "petal_width": 1.4}

model.get_prediction(X)    # "versicolor"
model.get_probability(X)   # {"setosa": 0.0, "versicolor": 0.87, "virginica": 0.13}
model.get_lookalikes(X)    # [[42, "versicolor", [0, 5]], [87, "versicolor", [3]]]
model.get_augmented(X)     # {**X, "Lookalikes": ..., "Probability": ..., "Prediction": ..., "Audit": ...}
```

**Audit output** (bucketed IF/ELIF/ELSE routing):
```
==================================================
  LAYER 0
==================================================
  >>>   IF   "petal_width" > 0.8:
  >>>     -> BUCKET 0 (78 members)
        ELSE:
            -> BUCKET 1 (42 members)

  Within BUCKET 0:
    12 lookalikes found
    P(versicolor)       =  83.3% ████████████████░░░░
    P(virginica)        =  16.7% ███░░░░░░░░░░░░░░░░░

==================================================
  GLOBAL SUMMARY (5 layers)
==================================================
  Total lookalikes: 47
  P(versicolor) = 87.2%
  P(virginica) = 10.6%
  P(setosa) = 2.1%
  >> PREDICTION: versicolor
### END AUDIT ###
```

## Save & Load

```python
# Save
model.to_json("model.json")

# Load (auto-detected by .json extension)
model = Snake("model.json")
```

The JSON file contains: `version`, `population`, `header`, `targets`, `datatypes`, `config`, `layers`, `log`.

Backwards compatible — v0.1 flat JSON files (with `clauses` + `lookalikes` at top level) are automatically wrapped into the bucketed format on load.

## Validation & Pruning

```python
# Score each layer on validation data, keep the top N
model.make_validation(val_data, pruning_coef=0.5)

# pruning_coef=0.5 keeps the best 50% of layers
# Save the pruned model
model.to_json("model_pruned.json")
```

`val_data` is a list of dicts (same format as training data, must include the target field).

## CLI

```bash
# Train
snake train data.csv --layers 5 --bucket 250 --noise 0.25 -o model.json --vocal

# Predict
snake predict model.json -q '{"petal_length": 4.3, "petal_width": 1.4}'
snake predict model.json -q '{"petal_length": 4.3}' --audit

# Model info
snake info model.json
```

## Benchmarks

Accuracy on classic sklearn datasets + Kaggle Spaceship Titanic. 80/20 train/test split (seed=42), `bucket=250`, `noise=0.25`. Run `python benchmarks.py` to reproduce.

**`n_layers=5`**

| Dataset | Type | Samples | Features | Classes | Train Acc | Test Acc | Train Time | Inference |
|---------|------|---------|----------|---------|-----------|----------|------------|-----------|
| Iris | Multi | 150 | 4 | 3 | 100.0% | 96.7% | 0.0s | 0.2ms |
| Wine | Multi | 178 | 13 | 3 | 100.0% | 97.2% | 0.0s | 0.2ms |
| Breast Cancer | Binary | 569 | 30 | 2 | 100.0% | 96.5% | 0.1s | 0.6ms |
| Digits | Multi | 1797 | 64 | 10 | 100.0% | 90.6% | 1.9s | 1.2ms |
| Spaceship Titanic | Binary | 8693 | 12 | 2 | 94.4% | 76.7% | 4.9s | 1.1ms |

**`n_layers=15`**

| Dataset | Type | Samples | Features | Classes | Train Acc | Test Acc | Train Time | Inference |
|---------|------|---------|----------|---------|-----------|----------|------------|-----------|
| Iris | Multi | 150 | 4 | 3 | 100.0% | 100.0% | 0.0s | 0.5ms |
| Wine | Multi | 178 | 13 | 3 | 100.0% | 97.2% | 0.0s | 0.7ms |
| Breast Cancer | Binary | 569 | 30 | 2 | 100.0% | 95.6% | 0.3s | 1.9ms |
| Digits | Multi | 1797 | 64 | 10 | 100.0% | 95.3% | 5.6s | 3.6ms |
| Spaceship Titanic | Binary | 8693 | 12 | 2 | 95.0% | 77.6% | 15.0s | 2.5ms |

**`n_layers=50`**

| Dataset | Type | Samples | Features | Classes | Train Acc | Test Acc | Train Time | Inference |
|---------|------|---------|----------|---------|-----------|----------|------------|-----------|
| Iris | Multi | 150 | 4 | 3 | 100.0% | 100.0% | 0.1s | 1.7ms |
| Wine | Multi | 178 | 13 | 3 | 100.0% | 100.0% | 0.1s | 2.3ms |
| Breast Cancer | Binary | 569 | 30 | 2 | 100.0% | 97.4% | 0.9s | 6.3ms |
| Digits | Multi | 1797 | 64 | 10 | 100.0% | 96.1% | 19.1s | 11.8ms |
| Spaceship Titanic | Binary | 8693 | 12 | 2 | 94.8% | 78.4% | 51.1s | 7.4ms |

**Spaceship Titanic** ([Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic)): 8,693 passengers, binary classification (transported or not). Features: HomePlanet, CryoSleep, Destination, VIP, CabinDeck, CabinSide + 6 numeric spending columns. Minimal preprocessing — no feature engineering beyond cabin splitting. Kaggle leaderboard top scores are ~80-81%.

More layers improve test accuracy at the cost of training time and inference latency. Benchmark script requires `pandas` and `scikit-learn` for data loading/splitting only — Snake itself has zero dependencies.

## Architecture

```
Input X
   │
   ▼
┌─────────────────────────────────────────┐
│         Bucket Chain (IF/ELIF/ELSE)      │
│                                          │
│  IF  condition_0(X):  → Bucket 0         │
│  ELIF condition_1(X): → Bucket 1         │
│  ELIF condition_2(X): → Bucket 2         │
│  ELSE:                → Bucket N         │
│                                          │
│  Each condition = AND of SAT literals    │
│  Each bucket ≤ 250 samples               │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Local SAT (per bucket)           │
│                                          │
│  For each target class:                  │
│    Build minimal clauses separating      │
│    positive from negative samples        │
│                                          │
│  Discriminating literals:                │
│    T   — substring present/absent        │
│    TN  — string length threshold         │
│    TLN — alphabet size threshold         │
│    TWS — word count threshold            │
│    N   — numeric threshold               │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Lookalike Voting                 │
│                                          │
│  Find training samples matching X        │
│  via SAT clause satisfaction             │
│                                          │
│  Vote by target labels → probability     │
│  Return max probability class            │
└─────────────────────────────────────────┘
```

Repeated across `n_layers` independent layers. Final prediction aggregates all lookalikes across all layers.

Complexity: `O(n * log(n) * m * bucket²)` where n = samples, m = features.

## Optional: Cython Acceleration

Snake includes optional Cython-accelerated hot paths for `apply_literal`, `apply_clause`, and `traverse_chain`. When compiled, these provide significant speedups for both training and inference.

```bash
# Install with Cython support
pip install -e ".[fast]"
python setup.py build_ext --inplace
```

Without Cython, Snake runs in pure Python with identical behavior. The Cython extension is auto-detected at import time.

## License

Proprietary. Source code is available for viewing and reference only.

See [LICENSE](LICENSE) for details. For licensing inquiries: contact@monce.ai
