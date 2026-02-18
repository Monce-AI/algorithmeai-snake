
[![GitHub](https://img.shields.io/github/stars/Monce-AI/algorithmeai-snake?style=social)](https://github.com/Monce-AI/algorithmeai-snake)
[![GitHub last commit](https://img.shields.io/github/last-commit/Monce-AI/algorithmeai-snake?logo=github)](https://github.com/Monce-AI/algorithmeai-snake/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/Monce-AI/algorithmeai-snake)](https://github.com/Monce-AI/algorithmeai-snake)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAxTDMgNXY2YzcgNCA4LjUgOC40IDkgMTIuOEM5LjUgMjAuNCA4IDE2IDggMTFWNmw0LTIuNUwxNiA2djVjMCA1LTEuNSA5LjQtNCAxa)](LICENSE)
[![Version](https://img.shields.io/badge/v4.4.3-SAT_Bucketed-blueviolet.svg?logo=semanticrelease)](https://github.com/Monce-AI/algorithmeai-snake)
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
Snake(Knowledge, target_index=0, excluded_features_index=(), n_layers=5, bucket=250, noise=0.25, vocal=False, saved=False, progress_file=None, workers=1)
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
| `progress_file` | str/None | `None` | File path for JSON training progress updates |
| `workers` | int | `1` | Parallel workers for layer construction (`>1` uses multiprocessing) |

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

**Audit output** (v4.4.3 — Routing AND + Lookalike AND):
```
### BEGIN AUDIT ###
  Prediction: versicolor
  Layers: 5, Lookalikes: 47

  LOOKALIKE SUMMARY
  ================================================
  versicolor           87.2% (41/47) █████████████████░░░
    e.g. sample with petal_length 4.5
  virginica            10.6% (5/47)  ██░░░░░░░░░░░░░░░░░░
    e.g. sample with petal_length 5.0

  PROBABILITY
  ================================================
  P(versicolor) = 87.2% █████████████████░░░
  P(virginica) = 10.6% ██░░░░░░░░░░░░░░░░░░

  ================================================
  LAYER 0
  ================================================

  Routing AND (bucket 1/2, 78 members):
    "petal_width" > 0.8

  Lookalike AND (12 matches):
    Lookalike #42 [versicolor]: 4.5
      AND: "petal_length" <= 5.0 AND "petal_width" <= 1.7
    ...

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

**JSON structure (v4.4.3):**
```json
{
  "version": "4.4.0",
  "population": [...],
  "header": ["target", "f1", ...],
  "target": "target",
  "targets": [...],
  "datatypes": ["T", "N", ...],
  "config": {"n_layers": 5, "bucket": 250, "noise": 0.25, "vocal": false, "workers": 1},
  "layers": [...],
  "log": "..."
}
```

Datatype codes: `B` = binary, `I` = integer, `N` = numeric, `T` = text, `J` = complex JSON (dict/list).

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

## Data Type Detection

Snake auto-detects types for each column at training time.

**Target column** (checked in priority order):

| Priority | Condition | Type Code | Storage |
|----------|-----------|-----------|---------|
| 1 | Any value is `dict` or `list` | `J` (Complex JSON) | Raw Python objects |
| 2 | Values are exactly `{"0", "1"}` | `B` (Binary) | `int` |
| 3 | Values are `{"True", "False"}` | `B` (Binary) | `int` |
| 4 | All chars in `0-9` only | `I` (Integer) | `int` |
| 5 | All chars in `+-.0123456789e` | `N` (Numeric) | `float` |
| 6 | Otherwise | `T` (Text) | `str` |

**Feature columns:** Numeric (`N`) vs Text (`T`) only — no Binary/Integer distinction.

**Watch out:**
- `"3", "4", "5"` → Integer (`I`), but `"3.0", "4.0"` → Numeric (`N`). Target type affects how predictions are compared.
- `floatconversion` silently converts unparseable strings (`"N/A"`, `""`) to `0.0`.

## Deduplication

Both CSV and list\[dict\] flows deduplicate by hashing all **feature values** (not the target). If two rows have identical features but different targets, the second is dropped with a log message. This is intentional — conflicting training data degrades SAT clause quality.

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

## Performance & Scaling

**Complexity:**
- **Training:** `O(n_layers * n_buckets * m * bucket_size²)` — dominated by SAT clause construction
- **Inference:** `O(n_layers * n_clauses_in_matched_bucket)` — fast for small buckets
- **Memory:** Entire population stored in memory; model JSON includes full training data

**Scaling guidance:**

| Dataset Size | Recommendation |
|-------------|----------------|
| < 500 samples | `bucket=250` works fine, single bucket per layer |
| 500–5,000 | `bucket=250` creates 2–20 buckets per layer, good performance |
| 5,000+ | Training gets slow — consider `n_layers=3-5`, `bucket=500` |
| 10,000+ | Segment by a certain field, train per-segment models |

**Inference latency** (from benchmarks):

| Configuration | Latency |
|--------------|---------|
| 150 samples, 5 layers | ~0.2ms |
| 1,797 samples, 50 layers | ~12ms |
| 8,693 samples, 50 layers | ~7.4ms |

## Common Patterns

### Threshold-based routing

```python
model = Snake("model.json")
prob = model.get_probability(X)
confidence = max(prob.values())
prediction = model.get_prediction(X)

if confidence >= 0.51:
    return prediction          # Snake is confident
else:
    return fuzzy_fallback(X)   # Fall back to another matcher
```

### Batch prediction

```python
model = Snake("model.json")
results = []
for row in batch:
    features = {k: v for k, v in row.items() if k != "target"}
    results.append({
        "prediction": model.get_prediction(features),
        "probability": model.get_probability(features),
    })
```

### Audit for RAG / LLM consumption

```python
audit = model.get_audit(X)
# Multi-line string with:
#   - Lookalike summary with examples per class
#   - Probability distribution
#   - Per-layer: Routing AND + Lookalike AND explanations
#   - Final prediction
# Feed to an LLM for explanation generation.
```

## Meta Error Classifier

Meta learns WHERE a base Snake model fails. It generates cross-validated error labels for every training sample, then trains an error-type Snake classifier on those labels.

**Binary targets (2 classes):** Labels are `TP` / `TN` / `FP` / `FN` / `NS` (not stable).
**Multiclass targets (3+ classes):** Labels are `R1`-`R5` (rank of correct class) / `W` (wrong, not in top 5) / `NS`.

```python
from algorithmeai import Meta

# Train a meta error classifier
meta = Meta(data, target_index="survived", n_layers=7, bucket=400,
            n_splits=25, n_runs=2, error_layers=7, error_bucket=50)

# Predict error type for a new sample
meta.get_prediction({"pclass": 3, "sex": "male", "age": 22})   # "FN"
meta.get_probability({"pclass": 3, "sex": "male", "age": 22})  # {"TP": 0.1, "TN": 0.2, "FP": 0.05, "FN": 0.6, "NS": 0.05}

# Export augmented dataset
meta.to_csv("train_extended.csv")   # Original data + error_type column

# Save & reload
meta.to_json("meta.json")           # Writes meta.json + meta_error_model.json
meta = Meta("meta.json")            # Load without re-running labeling

# Inspect
print(meta.summary())               # Human-readable distribution
print(meta.label_counts)            # Counter({'TN': 313, 'TP': 152, ...})
print(meta.agreement_rate)          # 0.93
```

### Constructor

```python
Meta(Knowledge, target_index=0, excluded_features_index=(),
     n_layers=5, bucket=250, noise=0.25, workers=1,
     n_splits=25, n_runs=2, split_ratio=0.8,
     error_layers=7, error_bucket=50,
     vocal=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Knowledge` | str / list / DataFrame | — | Same input formats as Snake, or `.json` path to load saved Meta |
| `target_index` | int / str | `0` | Target column (passed through to Snake) |
| `n_layers` | int | `5` | Layers for ephemeral split models |
| `bucket` | int | `250` | Bucket size for split models |
| `noise` | float | `0.25` | Cross-bucket noise for split models |
| `workers` | int | `1` | Parallel workers for split model training |
| `n_splits` | int | `25` | Number of 80/20 splits per labeling run |
| `n_runs` | int | `2` | Independent runs; agreement required, else NS |
| `split_ratio` | float | `0.8` | Train/test split ratio |
| `error_layers` | int | `7` | Layers for the error classifier |
| `error_bucket` | int | `50` | Bucket size for the error classifier |
| `vocal` | bool | `False` | Print progress |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_prediction(X)` | str | Predicted error type (e.g. `"FN"`, `"R2"`) |
| `get_probability(X)` | dict | `{error_type: probability}` for all labels |
| `to_list()` | list[dict] | Population with `error_type` column added |
| `to_csv(path)` | None | Write augmented population to CSV |
| `to_json(path)` | None | Save Meta + error model (two JSON files) |
| `summary()` | str | Human-readable label distribution |

### Use Case: Targeted FN Flipping

```python
from algorithmeai import Snake, Meta

# Train base model
base = Snake(data, target_index="survived", n_layers=7, bucket=400)

# Train meta error classifier
meta = Meta(data, target_index="survived", n_layers=7, bucket=400,
            n_splits=25, n_runs=2, error_layers=7, error_bucket=50)

# At prediction time
X = {"pclass": 3, "sex": "male", "age": 22}
base_pred = base.get_prediction(X)
error_prob = meta.get_probability(X)

# If meta predicts FN with high confidence, flip the prediction
if error_prob.get("FN", 0) > 0.70:
    base_pred = positive_class  # Flip to positive
```

## Gotchas

1. **Target type must match at prediction time.** If training targets were `int` (e.g., `0`, `1`), predictions return `int`. If `str` (e.g., `"cat"`), predictions return `str`. Mixing types silently fails to match.

2. **Feature types are fixed at training time.** A feature detected as Numeric (`N`) does numeric comparison. Passing a string at prediction time for that feature causes a `TypeError`.

3. **CSV must be pandas-formatted.** The CSV parser handles quoted fields with commas but expects `pandas.DataFrame.to_csv()` format. Non-pandas CSVs may parse incorrectly.

4. **No incremental training.** To add data, rebuild from scratch. `make_validation` only prunes layers.

5. **Model JSON contains the full training population.** A model trained on 5,000 rows produces a large JSON file. Be mindful of disk/memory.

6. **`excluded_features_index` only works with CSV flow.** For list\[dict\] and DataFrame, filter your data before passing it in.

7. **Binary True/False targets become int 0/1.** Compare predictions against `0`/`1` (int), not `"True"`/`"False"` (str).

## Error Handling

Snake has minimal error handling by design:

| Situation | What Happens | How to Avoid |
|-----------|-------------|-------------|
| Empty list | `ValueError` | Check before constructing |
| Empty DataFrame | `ValueError` | Check `len(df) > 0` |
| Wrong file extension | Crashes | Use `.csv` or `.json` extensions |
| File not found | `FileNotFoundError` | Check path exists |
| Malformed JSON | `json.JSONDecodeError` | Validate JSON before loading |
| `target_index` not found | `ValueError` / `IndexError` | Check column name/index exists |
| Only 1 unique target | Trains but trivially predicts that class | Need at least 2 classes |
| Prediction with `{}` | Uniform probability | Pass at least some features |
| Unknown keys in prediction | Ignored silently | Use same key names as training |

**No exceptions during prediction.** `get_prediction`, `get_probability`, `get_lookalikes`, `get_audit` all handle edge cases gracefully (worst case: uniform probability, empty lookalikes).

## Stochastic Behavior

Snake training is **non-deterministic**. The `oppose()` function uses `random.choice` extensively. Two runs on the same data produce different models with different clause sets. This is by design — the power comes from stochastic clause generation + deterministic selection.

There is no `random.seed()` call in the Snake code. If you need reproducibility:

```python
import random
random.seed(42)
model = Snake(data, n_layers=5)
```

Test assertions are probabilistic (e.g., "at least 50% of training data has >50% confidence") rather than exact value checks.

## Logging

Snake uses Python's `logging` module. Each instance gets its own logger (`snake.<id>`).

- **Buffer handler** — always attached at DEBUG level, captures everything to `self.log`. This is how `to_json()` persists the training log.
- **Console handler** — attached only when `vocal`:
  - `vocal=True`: INFO level (training progress)
  - `vocal=2`: DEBUG level (per-target SAT progress)
  - `vocal=False`: no console output

The banner only prints when `vocal=True`.

## Optional: Cython Acceleration

Snake includes optional Cython-accelerated hot paths for `apply_literal`, `apply_clause`, and `traverse_chain`. When compiled, these provide significant speedups for both training and inference.

```bash
# Install with Cython support
pip install -e ".[fast]"
python setup.py build_ext --inplace
```

Without Cython, Snake runs in pure Python with identical behavior. The Cython extension is auto-detected at import time.

## Testing

```bash
pytest                                # all 192 tests
pytest tests/test_snake.py            # input modes, save/load, augmented, vocal, dedup, parallel training
pytest tests/test_buckets.py          # bucket chain, noise, routing, audit, dedup
pytest tests/test_core_algorithm.py   # oppose, construct_clause, construct_sat
pytest tests/test_validation.py       # make_validation / pruning
pytest tests/test_edge_cases.py       # errors, type detection, extreme params
pytest tests/test_cli.py              # CLI train/predict/info via subprocess
pytest tests/test_logging.py          # logging buffer, JSON persistence, banner
pytest tests/test_audit.py            # Routing AND, Lookalike AND, audit end-to-end
pytest tests/test_stress.py           # stress tests, batch equivalence
pytest tests/test_ultimate_stress.py  # extended stress tests
pytest tests/test_meta.py             # Meta error classifier
```

192 tests across 11 files. Tests use `tests/fixtures/sample.csv` (15 rows, 3 classes) with small `n_layers` (1–3) and `bucket` (3–5) for speed.

## Changelog

### v4.4.3 (Feb 2026)

- **Meta error classifier**: New `Meta` class that learns WHERE a base Snake model fails. Cross-validated error labeling (binary: TP/TN/FP/FN/NS, multiclass: R1-R5/W/NS), trains an error-type Snake classifier, supports save/load and CSV export
- **192 tests**: Extended from 174 across 11 files (added 18 Meta tests)

### v4.4.2 (Feb 2026)

- **Perfected audit system**: Two clean AND statements per layer — Routing AND (explains bucket routing) and Lookalike AND (per-sample clause negation). Replaces the stub from v4.3.x
- **Parallel training**: `workers=N` enables multiprocessing for layer construction. Each worker gets a unique RNG seed
- **Progress tracking**: `progress_file` parameter writes JSON progress updates during training
- **Cython batch acceleration**: `batch_get_lookalikes_fast` amortizes routing by grouping queries per bucket per layer
- **174 tests**: Extended from 151 across 10 files (added audit tests, parallel training, batch equivalence)

### v4.3.3 (Feb 2026)

- **Cython infinite loop fix**: Fixed `oppose()` infinite loop in Cython hot paths
- **`oppose()` canaries**: Added safety canaries to detect stuck loops
- **151 tests**: Extended test suite from 92 to 151 tests across 9 files (added stress + ultimate stress tests)

### v4.3.2 (Feb 2026)

- **Logging migration**: Replaced `print()` + string accumulation with Python `logging`. Per-instance logger, buffer handler always captures, StreamHandler to stdout only when `vocal`
- **Extensive test suite**: 92 tests across 7 files (was ~41). New: core algorithm, validation, edge cases, CLI, logging
- **Cython training acceleration**: 4 new functions in `_accel.pyx` for `construct_clause`, `build_condition`, `_construct_local_sat`
- **Bug fix — Binary True/False targets**: `floatconversion("True")` returned `0.0`, collapsing all True/False targets to 0. Fixed in both flows
- **Spaceship Titanic benchmark**: 78.4% test accuracy (Kaggle top ~80–81%)
- **`__main__.py`**: Enables `python -m algorithmeai` as CLI entry point

## License

Proprietary. Source code is available for viewing and reference only.

See [LICENSE](LICENSE) for details. For licensing inquiries: contact@monce.ai
