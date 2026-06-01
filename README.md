
[![GitHub](https://img.shields.io/github/stars/Monce-AI/algorithmeai-snake?style=social)](https://github.com/Monce-AI/algorithmeai-snake)
[![GitHub last commit](https://img.shields.io/github/last-commit/Monce-AI/algorithmeai-snake?logo=github)](https://github.com/Monce-AI/algorithmeai-snake/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/Monce-AI/algorithmeai-snake)](https://github.com/Monce-AI/algorithmeai-snake)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAxTDMgNXY2YzcgNCA4LjUgOC40IDkgMTIuOEM5LjUgMjAuNCA4IDE2IDggMTFWNmw0LTIuNUwxNiA2djVjMCA1LTEuNSA5LjQtNCAxa)](LICENSE)
[![Version](https://img.shields.io/badge/v5.4.8-Parallel_Batch_+_Stripped_Serving-blueviolet.svg?logo=semanticrelease)](https://github.com/Monce-AI/algorithmeai-snake/releases/tag/v5.4.8)
[![Build](https://img.shields.io/badge/Build-295_tests_passing-brightgreen.svg?logo=githubactions&logoColor=white)](#testing)
[![Profiles](https://img.shields.io/badge/Profiles-8_oppose_strategies-orange.svg?logo=probot&logoColor=white)](#oppose-profiles-v520)

[![Production](https://img.shields.io/badge/Production-Live_on_AWS-FF9900.svg?logo=amazonaws&logoColor=white)](https://snake.aws.monce.ai)
[![API](https://img.shields.io/badge/API-snake.aws.monce.ai-009688.svg?logo=fastapi&logoColor=white)](https://snake.aws.monce.ai/health)
[![AUROC](https://img.shields.io/badge/AUROC-0.999_Breast_Cancer-brightgreen.svg?logo=target)](#snake-vs-rfgb)
[![Titanic](https://img.shields.io/badge/Titanic-0.924_AUROC-blue.svg?logo=speedtest)](#snake-vs-rfgb)
[![Regression](https://img.shields.io/badge/Regression-R²_+7pp_vs_classification-2E7D32.svg?logo=quicklook)](#regression-via-distribution-candles-v546)
[![Candles](https://img.shields.io/badge/Candles-Distribution_per_Prediction-D32F2F.svg?logo=tradingview&logoColor=white)](#regression-via-distribution-candles-v546)

[![Algorithm](https://img.shields.io/badge/Algorithm-SAT--based_Lookalikes-7B2D8B.svg?logo=probot&logoColor=white)](#what-is-snake)
[![XAI](https://img.shields.io/badge/XAI-Fully_Explainable-FF6F00.svg?logo=opensourceinitiative&logoColor=white)](#prediction-api)
[![Literals](https://img.shields.io/badge/Literals-29_boolean_test_types-lightgrey.svg?logo=wolframmathematica)](#oppose-type-formalism)
[![Cython](https://img.shields.io/badge/Cython-Optional_3x_speedup-E91E63.svg?logo=stackblitz&logoColor=white)](#optional-cython-acceleration)

[![Shannon](https://img.shields.io/badge/Shannon-MI--Weighted_Feature_Selection-0078D4.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQyIDAtOC0zLjU4LTgtOHMzLjU4LTggOC04IDggMy41OCA4IDgtMy41OCA4LTggNHoiLz48L3N2Zz4=)](#mi-weighted-feature-selection-v530)
[![Lookahead](https://img.shields.io/badge/Lookahead-Best_of_K_Literals-00C853.svg?logo=bullseye)](#lookahead-literal-selection-v530)
[![Madelon](https://img.shields.io/badge/Madelon_500f-+20pp_AUROC_vs_uniform-gold.svg?logo=target)](#mi-weighted-feature-selection-v530)

**Author:** Charles Dana · [Monce SAS](https://monce.ai) · Co-authored with [Claude](https://claude.ai)

> *I(X;Y) = Σ P(x,y) · log P(x,y) / P(x)P(y)* — the Shannon signature that decides which features matter.

# Snake

SAT-based explainable multiclass classifier. Zero dependencies. Pure Python.

> ## 🐍 New in v5.4.8 — parallel batch inference + stripped models for serving
>
> Every prediction method now takes a **list** as easily as a single dict — and uses every core when you do.
>
> ```python
> # Same method. Pass one dict → one result. Pass a list → a list, in order.
> model.get_prediction(X)        # "versicolor"
> model.get_prediction(batch)    # ["versicolor", "setosa", ...]  — fanned across CPUs
> ```
>
> - **Pass a list, use every core** — Snake divides a batch across `min(cpu_count, len(batch))` processes. **5.4× on a 10-core box** for a 20k-row batch, scaling with core count. No threads (GIL-bound), no new API.
> - **pandas, seamlessly** — train from a DataFrame, predict on a DataFrame or a single `pd.Series`, then `df['pred'] = model.get_prediction(X)`. Snake never imports pandas — it duck-types, so zero dependencies still holds.
> - **Exact, not approximate** — the batch result is element-for-element identical to looping the single-dict call. Layer independence + non-dedup vote merge make splitting safe.
> - **Stripped models for serving** — `to_json(stripped=True)` drops the training population (the bulk of the file) and keeps only what inference needs. Lean artifact → many workers fit in RAM → you stay CPU-bound. Audit/augmented raise a clear error on a stripped model instead of guessing.
>
> → [Batch inference](#batch-inference--pass-a-list-use-every-core-v548) · [pandas](#pandas-seamlessly-v548) · [Stripped models](#stripped-models-for-serving-v548)

> ### Previously — v5.4.7: `get_synthetic`, audit a whole dataset
>
> Snake already explains *every* prediction; v5.4.7 explains the **whole dataset** — and tells you, honestly, when there's nothing to explain. A permutation-free χ² significance test on debiased mutual information flags `is_noise=true` on random data and `strong` on real signal, reports calibration vs the true base rate, and runs zero-dependency offline (the plain-language narration is one optional cloud call). → [Full section](#synthetic-audit-at-scale-v547)

## What Is Snake

Snake is a **SAT-based lookalike voting classifier**. For each prediction, it finds training samples that "look alike" via Boolean clause matching, then votes by their labels. The result: a fully explainable classifier where every prediction comes with a human-readable audit trail.

```
Input X  →  Match SAT clauses  →  Find lookalikes  →  Vote  →  Prediction + Audit
```

> **Predicted outcome:** `setosa` (93.3%)
> Because: `"petal_length" <= 2.45` AND `"petal_width" <= 0.8`
> Matched 15 lookalikes, all class `setosa`.

**v5.4.8** makes every prediction method **polymorphic** — pass a list of dicts (or a **pandas DataFrame**, or a single `pd.Series`) and Snake divides the batch across your CPU cores (processes, not threads), returning results in input order, **exactly** matching the sequential call (**5.4× on 10 cores**, 20k rows). Train from a DataFrame, predict on a DataFrame, `df['pred'] = model.get_prediction(X)` — all duck-typed, so the library still imports nothing. Adds **stripped serialization** (`to_json(stripped=True)`) — drop the training population to serve the hot path from a lean, RAM-light artifact, the basis for a multi-core worker pool.

**v5.4.7** adds **`get_synthetic`** — a synthetic audit at *dataset* scale. Snake scans a batch of points locally (0 tokens, sub-ms each) and reports deterministic diagnostics: prediction spread, calibration vs the true base rate, debiased feature mutual-information, and a **label-free noise detector** that tells you honestly when your data looks like random luck. An optional one-call cloud narration (via the [monceai SDK](https://github.com/Monce-AI/monceai-sdk)) explains the finding as a tiny science experiment — *hypothesis / experiment / result*. The local path stays zero-dependency; the cloud voice is imported lazily.

**v5.4.6** adds **distribution candles** and **regression** — Snake now does both classification and regression from the same model. Every prediction yields a candle (high/q3/median/q1/low/mean/iqr_mean/std/n) summarising the lookalike distribution; the IQR-trimmed mean is the regression estimate. On a synthetic linear regression (n=800 train), **+7.4pp R²** over `get_prediction` and **2.3× faster** in batch.

**v5.4.0** added **Shannon MI-weighted feature selection** and **lookahead literal selection** — the first principled feature importance mechanism in Snake. On 500-feature datasets with 20 informative features, **+12.6pp AUROC** over v5.2.1. Zero regressions on classical datasets.

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

## Regression via Distribution Candles (v5.4.6)

Snake votes by lookalikes. When the target is a class, votes give a probability dict. When the target is a **float**, votes give a **distribution** — and a distribution is exactly what a trader reads off a chart: a high, a low, a body, a wick.

For each prediction, Snake assembles the lookalike y values into a `Candle`:

```
        ▲ high       ── max(y over lookalikes)
        │
        │
       ╔╧╗ q3        ── 75th percentile
       ║ ║
       ║─║ median    ── 50th percentile
       ║ ║
       ╚╤╝ q1        ── 25th percentile
        │
        │
        ▼ low        ── min(y over lookalikes)
```

The candle is a pure-distribution object — there is no order on a set of lookalikes, so OHLC is replaced by a five-number summary plus mean, IQR-trimmed mean, std, and n.

```python
from algorithmeai import Snake

# Train on a continuous target — y is a float, no binning required.
data = [{"x1": ..., "x2": ..., "y": 12.34}, ...]
model = Snake(data, target_index="y", n_layers=20, bucket=80, noise=0.5, workers=4)

# Single-point candle
c = model.get_candle({"x1": 4.3, "x2": 1.4})
c.high, c.q3, c.median, c.q1, c.low      # five-number summary
c.mean, c.iqr_mean, c.std, c.n           # point estimates + dispersion
c.to_dict()                               # JSON-friendly

# Regression — the IQR-trimmed mean of the lookalike set
y_hat = model.get_regression({"x1": 4.3, "x2": 1.4})

# Batch (uses the Cython batch lookalike fast-path when available)
candles = model.get_batch_candles(X_test)
y_hats  = model.get_batch_regression(X_test)
```

### Why IQR-trimmed mean

`get_prediction` is mode-like — it picks the most-frequent lookalike value, which is brittle on continuous targets. `iqr_mean` averages the *middle 50%* of the lookalike distribution, discarding the wicks. It's robust like the median and smooth like the mean.

| Method | What it does | R² | Time (200 preds) |
|--------|--------------|-----|---------------|
| `get_prediction` (classification path) | most-frequent lookalike y | 0.8734 | 248 ms |
| `get_regression` / `iqr_mean` | average of lookalikes within [Q1, Q3] | **0.9477** | **106 ms** |

Synthetic regression: `y = 3·x₁ − 2·x₂ + 1.5·x₃ + 𝒩(0, 1)`, n=800 train / 200 test, `Snake(n_layers=20, bucket=80, noise=0.5)`. Batch is faster because all four candle/regression methods share a single Cython lookalike fetch.

### Why the wicks are information

Wide wicks → "model is unsure, downside/upside tail real." Tight body around the median → "high consensus." The candle gives the trader (or risk function) a confidence interval *for free*, on every prediction, without bootstrapping or sampling.

```python
c = model.get_candle(X)
range_width = c.high - c.low          # uncertainty proxy
body_width  = c.q3 - c.q1             # consensus width
skew        = (c.q3 - c.median) - (c.median - c.q1)  # >0 = upside skew
```

### API

| Method | Returns | Description |
|--------|---------|-------------|
| `get_candle(X)` | `Candle` | Distribution of lookalike y values |
| `get_batch_candles(Xs)` | `list[Candle]` | Batched, shares the Cython lookalike path |
| `get_regression(X)` | `float` | IQR-trimmed mean of the candle |
| `get_batch_regression(Xs)` | `list[float]` | Batched regression |

`Candle` is a `dataclass` exposing `high, q3, median, q1, low, mean, iqr_mean, std, n` plus `to_dict()`.

The candle path applies to any continuous target — financial returns, lab measurements, sensor readings, latencies, prices. The model is unchanged; only the post-processing of lookalikes differs.

## MI-Weighted Feature Selection (v5.4.0)

The Shannon signature. On high-dimensional data, oppose() used to pick features **uniformly at random**. With 20 informative features out of 500, that's a **4% hit rate on signal**. The rest is noise.

v5.4.0 precomputes **mutual information** `MI(feature; target)` for each feature — the exact quantity decision trees maximize at each split (Quinlan 1986). oppose() now picks features **proportionally to MI weight**:

```python
# Before (v5.2.1): uniform random
index = choice(candidates)

# After (v5.4.0): MI-weighted
index = choices(candidates, weights=MI_weights)[0]
```

**The math:**
```
MI(X; Y) = Σ P(x,y) · log₂ P(x,y) / (P(x) · P(y))
```

Computed once in O(n×m), passed to all workers. Numeric features are quantile-binned into 20 bins. Text features use raw values (capped at 200 unique).

**Impact on feature selection (Hard Madelon, 500 features):**

| Informative features | Uniform P(signal) | MI-weighted P(signal) | Lift |
|---------------------|-------------------|----------------------|------|
| 5 / 500 | 1.0% | 3.1% | 3.1× |
| 20 / 500 | 4.0% | 10.8% | 2.7× |
| 50 / 500 | 10.0% | 25.0% | 2.5× |

## Lookahead Literal Selection (v5.4.0)

MI weights decide **which features** to try. Lookahead decides **which literal to keep**.

Instead of taking the first literal oppose() generates, `construct_clause` now generates **K=5 candidates** and picks the one that **covers the most Ts** (training positives). Higher coverage = tighter clause = fewer literals needed = better generalization.

```python
# Before: one shot
lit = oppose(choice(Ts), F)

# After: best of K
candidates = [oppose(choice(Ts), F) for _ in range(K)]
lit = max(candidates, key=lambda l: coverage(l, Ts))
```

On classical data, this produces **fewer, tighter clauses**:

| Dataset | v5.2.1 clauses | v5.4.0 clauses | Reduction |
|---------|---------------|---------------|-----------|
| Breast Cancer | 872 | 783 | -10.2% |
| Digits | 10,195 | 9,061 | -11.1% |
| Wine | 396 | 343 | -13.4% |

```python
# Control lookahead: K=1 disables it, K=10 for maximum quality
model = Snake(data, lookahead=5)   # default
model = Snake(data, lookahead=1)   # v5.2.1 behavior
model = Snake(data, lookahead=10)  # slower, tighter clauses
```

## A/B: v5.4.0 vs v5.2.1

Tested on Hard Madelon (500 features, weak Gaussian signal in n informative features, noise in the rest) and classical sklearn datasets. Same seeds, same splits.

**High-dimensional (the MI sweet spot):**

| Dataset (500 features) | v5.2.1 AUROC | v5.4.0 AUROC | Delta |
|------------------------|-------------|-------------|-------|
| 5 informative, signal=1.2 | 0.526 | **0.727** | **+20.1pp** |
| 20 informative, signal=0.8 | 0.587 | **0.712** | **+12.6pp** |
| 50 informative, signal=0.8 | 0.711 | **0.850** | **+13.9pp** |
| 10 informative, signal=0.8 | 0.562 | **0.634** | **+7.2pp** |

**Classical (no regression):**

| Dataset | Features | v5.2.1 AUROC | v5.4.0 AUROC | Delta |
|---------|----------|-------------|-------------|-------|
| Breast Cancer | 30 | 0.996 | **0.998** | +0.2pp |
| Iris | 4 | 0.993 | **0.998** | +0.5pp |
| Wine | 13 | 0.999 | **1.000** | +0.1pp |
| Digits | 64 | 0.997 | 0.996 | -0.1pp |

Run `python benchmark_mi.py` to reproduce.

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
DataFrames work for **inference** too — `model.get_prediction(df)` predicts every
row in parallel and returns a list in row order. See [DataFrames & pandas](#dataframes--pandas-v548).

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
Snake(Knowledge, target_index=0, excluded_features_index=(), n_layers=5, bucket=250, noise=0.25, vocal=False, saved=False, progress_file=None, workers=1, oppose_profile="auto", lookahead=5)
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
| `oppose_profile` | str | `"auto"` | Literal generation strategy: `auto`, `balanced`, `linguistic`, `industrial`, `cryptographic`, `scientific`, `categorical` |
| `lookahead` | int | `5` | Literal candidates per oppose call. `1` = v5.2.1 behavior. Higher = tighter clauses, slower |

## Prediction API

| Method | Returns | Description |
|--------|---------|-------------|
| `get_prediction(X)` | value | Most probable class |
| `get_probability(X)` | dict | `{class: probability}` for all classes |
| `get_lookalikes(X)` | list | `[[index, class, condition], ...]` matched training samples |
| `get_lookalikes_labeled(X)` | list | `[[index, class, condition, origin], ...]` with `"c"` (core) or `"n"` (noise) |
| `get_augmented(X)` | dict | Input enriched with Lookalikes, Probability, Prediction, Audit |
| `get_audit(X)` | str | Full human-readable reasoning trace |
| `get_candle(X)` | `Candle` | Distribution of lookalike y values — high/q3/median/q1/low/mean/iqr_mean/std/n |
| `get_batch_candles(Xs)` | `list[Candle]` | Batched candles, shares Cython lookalike fast-path |
| `get_regression(X)` | `float` | IQR-trimmed mean of the candle (continuous targets) |
| `get_batch_regression(Xs)` | `list[float]` | Batched regression |
| `get_synthetic(Xs, interpret=True)` | dict | Dataset-scale synthetic audit (v5.4.7) — deterministic summary + optional cloud narration |

Every `X` above can also be a **list of dicts** — Snake then parallelizes across
your CPU cores and returns a list of results in input order. See
[Batch inference](#batch-inference--pass-a-list-use-every-core-v548).

```python
X = {"petal_length": 4.3, "petal_width": 1.4}

model.get_prediction(X)    # "versicolor"
model.get_probability(X)   # {"setosa": 0.0, "versicolor": 0.87, "virginica": 0.13}
model.get_lookalikes(X)    # [[42, "versicolor", [0, 5]], [87, "versicolor", [3]]]
model.get_augmented(X)     # {**X, "Lookalikes": ..., "Probability": ..., "Prediction": ..., "Audit": ...}
```

### Batch inference — pass a list, use every core (v5.4.8)

Every prediction method is **polymorphic**: give it a single dict and you get a
single result; give it a **list of dicts** and Snake fans the work across a
proportionate number of CPU processes and returns a list in the same order.
Same method, same name — you don't learn a new API to go fast.

```python
# Single datapoint — one result (unchanged)
model.get_prediction({"petal_length": 4.3, "petal_width": 1.4})   # "versicolor"

# A list — Snake divides it across your cores, returns a list in input order
batch = [{"petal_length": 4.3, "petal_width": 1.4}, ...]   # 20,000 rows
model.get_prediction(batch)     # ["versicolor", "setosa", ...]  (len == 20,000)
model.get_probability(batch)    # [ {...}, {...}, ... ]
model.get_lookalikes(batch)     # [ [...], [...], ... ]
model.get_regression(batch)     # [ 3.14, 2.71, ... ]
```

**The result is exact, not approximate.** Each datapoint is scored independently
by the very same single-dict code path, so the batch result is element-for-element
identical to a plain `[model.get_prediction(x) for x in batch]` — just faster.
Snake's layers are independent and votes are merged without dedup, so splitting a
batch across processes can't change an answer.

**It uses processes, not threads** (pure-Python inference is GIL-bound, so threads
wouldn't help). Snake sizes the pool to `min(cpu_count, len(batch))` — one balanced
chunk per worker — so it scales with the hardware you actually have. Measured **5.4×
on a 10-core machine** for a 20k-row batch; the speedup tracks core count.

Best practices:

- **Hand Snake the whole list at once.** Don't loop in Python calling the
  single-dict method — that's the slow path you're trying to avoid. One call with
  the full list lets Snake schedule the cores.
- **Batch has a small fixed cost** (forking workers). Below `model._parallel_threshold`
  (default 64) Snake runs inline automatically — tiny batches don't pay for a pool.
- **Cap the pool** with `model._max_workers = N` if you're sharing the box with
  other work (e.g. a web server's own workers). Defaults to all cores.

The same polymorphism extends to **pandas** — pass a DataFrame or a `pd.Series`
and Snake handles it natively. See [DataFrames & pandas](#dataframes--pandas-v548)
for the full workflow.

**Audit output** (Routing AND + Lookalike AND):
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

## DataFrames & pandas (v5.4.8)

Snake speaks **pandas** end to end — train from a `DataFrame`, predict on a
`DataFrame`, score a single `Series` — without you ever converting a thing.

**The whole loop, four lines:**

```python
import pandas as pd
from algorithmeai import Snake

df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")

snake = Snake(df_train, target_index="target")     # fit
df_test["target"] = snake.get_prediction(df_test)  # predict the whole test set, in row order
df_test.to_csv("submission.csv", index=False)      # ship it
```

That's the entire train → predict → submit cycle. No `.values`, no `.to_dict()`,
no batching loop, no glue — read a CSV, fit, assign the column back. The
`get_prediction(df_test)` call fans every row across your CPU cores under the hood
and hands back a list in row order, so the assignment just lines up.

> **Zero-dependency, by design.** Snake **never imports pandas**. It *duck-types*:
> a DataFrame is "anything with a callable `.to_dict()` and a `.columns` attribute",
> a Series is "`.to_dict()` but no `.columns`". So pandas flows through only because
> *you* brought it — install Snake without pandas and nothing changes.

### Train from a DataFrame

The constructor already accepts a DataFrame; `target_index` is just the label
column's name (or position). Everything else — type detection, dedup, profiles —
works exactly as with `list[dict]`.

```python
import pandas as pd
from algorithmeai import Snake

df = pd.read_csv("orders.csv")     # columns: family, denomination, factory, thickness, qty
model = Snake(df, target_index="family", n_layers=12, oppose_profile="industrial")
```

### Predict on a DataFrame — and read off columns

Every prediction method takes the feature DataFrame and returns a **plain list in
row order**, so assignment back onto the frame is positional and clean. Under the
hood it's the v5.4.8 parallel batch path — the rows fan out across your CPU cores.

```python
X = df.drop(columns=["family"])

df["prediction"]  = model.get_prediction(X)                       # ["LAMINATED", "MONOLITHIC", ...]
df["confidence"]  = [max(p.values()) for p in model.get_probability(X)]
df["forecast"]    = model.get_regression(X)                       # continuous targets (candles)

# Then it's just pandas — group, filter, route on Snake's own output:
df.groupby("factory")["confidence"].mean()
low_conf = df[df["confidence"] < 0.80]                            # send these to a fallback
```

### An *explainable* prediction column — audit + lookalikes per row

This is what makes Snake different from a black-box classifier on a DataFrame:
every prediction comes with its **reasoning** and its **evidence**, and both are
just columns. `get_audit(df)` returns one human-readable trace per row; the
labeled lookalikes are the actual training rows that voted for each answer.

```python
df["prediction"] = model.get_prediction(X)
df["audit"]      = model.get_audit(X)                             # one reasoning trace per row (str)
df["lookalikes"] = model.get_lookalikes_labeled(X)               # [[idx, class, condition, "c"|"n"], ...] per row

# Why was this row called LAMINATED? Read it.
print(df.loc[0, "audit"])

# How many training rows backed it, and were they core or noise?
las  = df.loc[0, "lookalikes"]
core = [la for la in las if la[3] == "c"]
print(f"{len(las)} lookalikes, {len(core)} core")
```

### One call, the whole bundle — `get_augmented` → enriched DataFrame

`get_augmented(df)` returns a list of dicts that each carry the **original row
(your metadata)** plus `Prediction`, `Probability`, `Audit`, and `Lookalikes`.
Wrap it in `pd.DataFrame(...)` and you get a fully enriched frame in one line —
your `id` and feature columns ride along untouched, next to everything the model
knows about each row.

```python
enriched = pd.DataFrame(model.get_augmented(X))
# columns: <your original columns...> + Prediction + Probability + Audit + Lookalikes

enriched[["id", "Prediction"]]                                   # metadata + answer
enriched["n_lookalikes"] = enriched["Lookalikes"].str.len()      # evidence count, per row
enriched.query("Prediction == 'LAMINATED'")["Audit"].iloc[0]     # full reasoning for a class
```

Predict, explain, and keep your keys — all in a single pass over the data. Filter
on confidence, drill into any row's audit, count its supporting lookalikes, route
the uncertain ones to a fallback. The submission isn't just labels; it's a
queryable record of *why*.

### Score a single row (`pd.Series`)

A single row off a DataFrame is a `pd.Series` — pass it directly. It behaves
exactly like the equivalent dict, all the way through the full audit trail.

```python
row = X.iloc[0]                         # a pd.Series
model.get_prediction(row)               # "LAMINATED"
model.get_probability(row)              # {"LAMINATED": 1.0, "MONOLITHIC": 0.0}
print(model.get_audit(row))             # human-readable reasoning, no conversion
model.get_lookalikes_labeled(row)       # the training rows that voted, with core/noise origin
```

### What returns what

| You pass | You get back | Notes |
|----------|--------------|-------|
| `dict` | single result | unchanged — the classic path |
| `pd.Series` (one row) | single result | treated as a dict |
| `list[dict]` | `list` of results, input order | parallel across cores |
| `pd.DataFrame` | `list` of results, **row order** | parallel across cores; assign with `df["col"] = ...` |

**One guarantee worth stating plainly:** the result is *identical* across all three
batch call styles. `model.get_prediction(df)`, `model.get_prediction(df.to_dict("records"))`,
and `[model.get_prediction(r) for r in rows]` return the same list — DataFrame
support is pure ergonomics, never a different code path. (Covered by the test suite.)

## Synthetic Audit at Scale (v5.4.7)

`get_audit(X)` explains **one** prediction. `get_synthetic(Xs)` explains a **whole dataset**: it scans a batch, aggregates deterministic diagnostics locally (0 tokens, sub-ms per point), and — optionally — makes a single cloud call to narrate the finding in plain language.

The design has two layers, and the split is deliberate:

- **Local core (`interpret=False`)** — 100% standard library, zero network, zero tokens. This is what the public free tier always gets.
- **Cloud voice (`interpret=True`, default)** — one call to the [monceai SDK](https://github.com/Monce-AI/monceai-sdk) `Json`, imported **lazily** so Snake stays zero-dependency. `monceai` depends on `algorithmeai`, never the reverse — no circular dependency.

```python
model = Snake(train, target_index="sensitive", n_layers=20, bucket=200, oppose_profile="industrial")

# Pure local — deterministic, offline, free. Pass labeled rows to also get held-out metrics.
summary = model.get_synthetic(test, interpret=False)["summary"]
```
```python
{
  "n_points": 95, "n_classes": 2, "task": "binary", "n_layers": 20,
  "prediction_distribution_pct": {"0": 57.9, "1": 42.1},
  "true_base_rate_pct":          {"0": 50.4, "1": 49.6},
  "calibration_gap_pts": 7.5,            # predicted vs true majority rate
  "mean_confidence": 0.881,
  "low_confidence_rate_pct": 11.6,
  "coverage_rate_pct": 100.0,            # share of points a bucket actually matched
  "top_features_mi": [["subtype", 0.61], ["hotspot_mutations", 0.60], ...],
  "signal_strength": "strong",           # none / weak / moderate / strong
  "is_noise": false,                     # honest: true when data looks like chance
  "n_labeled": 95,
  "holdout_accuracy_pct": 82.1,          # only when X carries the target column
  "holdout_auroc": 0.856                 # no retrain — uses the labels you passed
}
```

```python
# With the cloud voice — one Json call narrates it as a science experiment.
out = model.get_synthetic(test)
print(out["hypothese"])    # "We bet the genetic clues could sort cell lines into responders vs not."
print(out["experience"])   # "Snake ran a 20-layer logic check over all 95 points, spending zero AI tokens."
print(out["resultat"])     # "The bet held: 82.1% correct (AUROC 0.856), strongest clue = subtype —
                           #  though it's a touch over-optimistic, so trust accuracy over confidence."
```

If `monceai` is not installed, `interpret=True` raises a clear error pointing at the install **and** the `interpret=False` escape hatch — never a bare `ImportError`.

### The honest noise detector

The headline property: **Snake refuses to manufacture signal from noise.** Two safeguards back the `is_noise` flag:

- **Effect size** — mutual information per feature, **Miller-Madow bias-corrected**. On finite data, an unrelated feature still shows MI > 0 because empty (feature, class) cells never appear; we subtract the analytic null `(bins−1)(classes−1)/(2n·ln2)`.
- **Significance** — under independence, `2n·MI·ln2 ∼ χ²` with `df = (bins−1)(classes−1)`. `is_noise` fires only when **no** feature clears ≈3σ above its null. This is a permutation-free significance test, not a tuned threshold.

| Data | `signal_strength` | `is_noise` | held-out AUROC |
|------|-------------------|-----------|----------------|
| Random labels (coin-flip) | `none` | `true` | ≈ 0.50 |
| Real signal `σ(1.8a − 1.3b)` | `strong` | `false` | 0.84 |

Same pipeline, opposite verdicts — exactly what you want from an explainable classifier: silent when there's nothing, confident when there is.

### Inputs & returns

- `Xs` — a list of dicts (a lone dict is treated as a batch of one). Include the target column to receive held-out `accuracy`/`AUROC`; omit it for an unlabeled scan (everything except those two fields still works).
- Returns `{"summary": {...}}`, plus `"hypothese"`/`"experience"`/`"resultat"` when `interpret=True`.

## Lookalike Origins — Core vs Noise (v5.0.0)

Each lookalike carries an origin label: **core `(c)`** = routed to the bucket by condition, **noise `(n)`** = randomly injected from the full population for regularization. This splits Snake's probability into independent signals.

```python
model = Snake(data, target_index="label", n_layers=77, bucket=150, noise=0.40, workers=10)

# Labeled lookalikes — each entry: [global_idx, target_value, condition, origin]
lookalikes = model.get_lookalikes_labeled(X)
for idx, target, cond, origin in lookalikes:
    print(f"#{idx} [{target}] ({origin})")  # e.g. "#42 [1] (c)", "#8 [0] (n)"

# Weighted probability — trust core more than noise
def weighted_prob(lookalikes, target_class, w_c=2, w_n=1):
    total = sum(w_c if la[3] == "c" else w_n for la in lookalikes)
    hits = sum((w_c if la[3] == "c" else w_n) for la in lookalikes if str(la[1]) == str(target_class))
    return hits / total if total > 0 else 0.5

# Split signals
core_only = [la for la in lookalikes if la[3] == "c"]
noise_only = [la for la in lookalikes if la[3] == "n"]
```

**Key finding:** The optimal weight ratio `(w_c, w_n)` depends on `n_layers`. At low layer counts, core dominates — noise is a distraction. At high layer counts, noise becomes a genuine complementary signal because full-population diversity compounds across stochastic layers.

| Config | Core AUROC | Noise AUROC | Divergence winner | Best weighting |
|--------|-----------|-------------|-------------------|----------------|
| 7 layers, noise=0.25 | 0.895 | 0.768 | Core 81% | Pure core |
| 77 layers, noise=0.40 | 0.891 | 0.877 | Noise 59% | Noise-heavy |

Backwards compatible — old models without origins default to `"c"` for all lookalikes.

## Oppose Profiles (v5.2.0)

Snake supports **7 oppose profiles** — each a tuned literal generation strategy for a data archetype. The `oppose()` function is untouched; profiles are substitute functions that control which literal types get generated and at what probability.

```python
# Auto (default) — Snake scans your data and picks the best profile
model = Snake(data, oppose_profile="auto")

# Explicit — you know your data
model = Snake(data, oppose_profile="cryptographic")
model = Snake(data, oppose_profile="linguistic")
```

| Profile | Best for | Key literal types | Speed |
|---------|----------|-------------------|-------|
| `auto` | Any data — scans population, picks one | Depends on detection | — |
| `balanced` | Unknown data, mixed types | Equal weight across all 24 types | Medium |
| `linguistic` | NLP, free text, author attribution | LEV (edit distance), JAC (bigram similarity), PFX/SFX | Slower |
| `industrial` | Product codes, SKUs, short labels | T (substring), TN/TLN (structural), splits | Fast |
| `cryptographic` | Hashes, IDs, encoded data | ENT (entropy), HEX (hex ratio), CFC (char freq), REP (repeat) | Medium |
| `scientific` | Measurements, sensors, lab data | NZ (z-score), NL (log-scale), NMG (magnitude) | Fast |
| `categorical` | Surveys, tags, enums | TWS/TPS/TSS (splits) + T (substring) at 60% combined | Fast |

**30 literal types** (was 7): the original 7 (`T`, `TN`, `TLN`, `TWS`, `TPS`, `TSS`, `N`) plus 23 new types across distance, positional, charclass, crypto, and scientific families. Each literal is still `[index, value, negat, tag]` — same format, same `apply_literal`.

**Auto-detection** scans text features for avg length, length variance, digit ratio, uppercase ratio, special char ratio, and delimiter density. Pure numeric data → `scientific`. Long varied text → `linguistic`. Short codes with digits → `industrial`. High special chars → `cryptographic`. Many delimiters → `categorical`. Mixed or unclear → `balanced`.

Backwards compatible — old models without `oppose_profile` default to the original `oppose()`. New literal types return `False` for unknown tags (graceful degradation).

## Snake vs RF/GB

Same data, same split (80/20, seed=42), **zero preprocessing**. Snake uses 15 layers, bucket=250, workers=10. RF/GB use 100 estimators (sklearn defaults). No feature engineering on either side.

**Pure numeric (float matrices):**

| Dataset | Features | Classes | Random Forest | GradBoost | Snake (best profile) | Profile |
|---------|----------|---------|--------------|-----------|---------------------|---------|
| Iris | 4 | 3 | 100.0% | 100.0% | **100.0%** | all tie |
| Wine | 13 | 3 | 100.0% | 94.4% | **100.0%** | original |
| Breast Cancer | 30 | 2 | AUROC 0.997 | AUROC 0.995 | **AUROC 0.999** | original |
| Digits | 64 | 10 | **97.2%** | 96.9% | 96.4% | original |

**Mixed text + numeric (the Snake sweet spot):**

| Dataset | Features | Classes | Snake AUROC | Snake Acc | Best profile | vs original |
|---------|----------|---------|------------|-----------|-------------|-------------|
| Classic Titanic (w/ Names) | 8 | 2 | **0.924** | **87.2%** | cryptographic | +3.8pp |
| Spaceship Titanic | 12 | 2 | **0.840** | **78.6%** | balanced | +3.7pp |

Snake **beats RF and GB on Breast Cancer AUROC** (0.999 vs 0.997 vs 0.995). Ties on Iris/Wine. Within 0.8pp on Digits. On mixed text+numeric data, profiles add up to **+3.8pp AUROC** over the original oppose — no preprocessing required.

## Save & Load

```python
# Save (full model — keeps everything, the default)
model.to_json("model.json")

# Load (auto-detected by .json extension)
model = Snake("model.json")
```

Backwards compatible — v0.1 flat JSON files (with `clauses` + `lookalikes` at top level) are automatically wrapped into the bucketed format on load.

### Stripped models for serving (v5.4.8)

A saved model includes the full training **population** — typically the large
majority of the file. Inference doesn't need it: prediction, probability,
lookalikes, candle, and regression run entirely off the SAT layers. Save a
**stripped** model to drop the population and keep only what serving needs:

```python
# One-time: produce a lean artifact for production
model.to_json("model.stripped.json", stripped=True)

# In the server: load the stripped model — far less RAM per instance
served = Snake("model.stripped.json")
served.get_prediction(X)        # works
served.get_probability(batch)   # works — and parallelizes
```

**Why it matters for a worker pool.** To use every core you run several Snake
workers, each holding the model. A stripped model is small, so you can fit many
copies in RAM and stay **CPU-bound** (every core busy doing useful clause work)
instead of RAM-bound. This is the same split the production Snake API uses to
serve at scale.

**The trade-off is explicit and safe.** `get_audit` and `get_augmented` render
real training examples ("e.g. *44.2 LowE*"), so they need the population. Call
them on a stripped model and you get a clear error telling you to load the full
model — never a silent wrong answer:

```python
served.get_audit(X)
# RuntimeError: get_audit() needs the training population, but this model was
# loaded stripped (no population). ... Re-save with to_json(stripped=False) ...
```

Rule of thumb: **keep one full model** (for audits / RAG / debugging) and
**serve from the stripped one** (for the high-throughput hot path).

## Validation & Pruning

```python
# Score each layer on validation data, keep the top N
model.make_validation(val_data, pruning_coef=0.5)

# pruning_coef=0.5 keeps the best 50% of layers
# Save the pruned model
model.to_json("model_pruned.json")
```

`val_data` is a list of dicts (same format as training data, must include the target field).

## Architecture

```
Input X
   │
   ▼
┌─────────────────────────────────────────┐
│     MI-Weighted Feature Selection       │
│                                          │
│  _precompute_feature_mi()               │
│  MI(feature; target) for each column    │
│  oppose() picks features ∝ MI weight    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Lookahead Literal Selection        │
│                                          │
│  Generate K=5 candidate literals        │
│  Pick the one covering the most Ts      │
│  → Tighter clauses, fewer per formula   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Bucket Chain (IF/ELIF/ELSE)         │
│                                          │
│  IF  condition_0(X):  → Bucket 0        │
│  ELIF condition_1(X): → Bucket 1        │
│  ELSE:                → Bucket N        │
│                                          │
│  Each condition = AND of SAT literals   │
│  Each bucket ≤ 250 samples              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Local SAT (per bucket)            │
│                                          │
│  For each target class:                 │
│    Build minimal clauses separating     │
│    positive from negative samples       │
│                                          │
│  30 literal types (7 families):         │
│    T/TN/TLN — substring/structural      │
│    TWS/TPS/TSS — split counts           │
│    N — numeric threshold                │
│    LEV/JAC — edit distance/bigrams      │
│    PFX/SFX — prefix/suffix length       │
│    ENT/HEX/REP/CFC — crypto features   │
│    NZ/NL/NMG — z-score/log/magnitude    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        Lookalike Voting                 │
│                                          │
│  Find training samples matching X       │
│  via SAT clause satisfaction            │
│                                          │
│  Vote by target labels → probability    │
│  Return max probability class           │
└─────────────────────────────────────────┘
```

Repeated across `n_layers` independent layers. Final prediction aggregates all lookalikes across all layers.

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

## Performance & Scaling

**Complexity:**
- **Training:** `O(n_layers * n_buckets * m * bucket_size²)` — dominated by SAT clause construction
- **Inference:** `O(n_layers * n_clauses_in_matched_bucket)` — fast for small buckets
- **MI precompute:** `O(n * m)` once — negligible vs training
- **Lookahead:** `O(K * |Ts|)` per literal — offset by fewer literals per clause
- **Memory:** Entire population stored in memory; model JSON includes full training data

**Scaling guidance:**

| Dataset Size | Recommendation |
|-------------|----------------|
| < 500 samples | `bucket=250` works fine, single bucket per layer |
| 500–5,000 | `bucket=250` creates 2–20 buckets per layer, good performance |
| 5,000+ | Training gets slow — consider `n_layers=3-5`, `bucket=500` |
| 10,000+ | Segment by a certain field, train per-segment models |

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

## Oppose Type Formalism

Snake's boolean test language is fully specified in [`oppose_types.snake`](oppose_types.snake). Every literal is:

```
MEASURE(field) > threshold    (negat flips to <=)
```

The file defines:
- **§1 Measures** — 20 primitive functions (identity, len, entropy, levenshtein, zscore, ...)
- **§2 Literals** — 30 boolean test types with oppose rules, eval rules, and format templates
- **§3 Profiles** — weight vectors over literal types

New literal types are added as cartridges: define the measure, the oppose rule (how to compute threshold from T and F), and the eval rule (how to test a new field). The `.snake` format is human-readable and serves as the single source of truth for Snake's discriminator library.

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

Snake includes optional Cython-accelerated hot paths for `apply_literal`, `apply_clause`, and `traverse_chain`. When compiled, these provide ~3× speedups for both training and inference. The lookahead coverage check in v5.4.0 automatically benefits from `apply_literal_fast`.

```bash
# Install with Cython support
pip install -e ".[fast]"
python setup.py build_ext --inplace
```

Without Cython, Snake runs in pure Python with identical behavior. The Cython extension is auto-detected at import time.

## Testing

```bash
pytest                                # all 266 tests
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
pytest tests/test_oppose_profiles.py  # oppose profiles, new literal types, auto-detection, JSON roundtrip
pytest tests/test_feature_mi.py       # MI precompute, weighted sampling, lookahead, JSON roundtrip
pytest tests/test_candle.py           # distribution candles, regression, IQR-trimmed mean
```

266 tests across 13 files. Tests use `tests/fixtures/sample.csv` (15 rows, 3 classes) with small `n_layers` (1–3) and `bucket` (3–5) for speed.

## Changelog

### v5.4.6 (May 2026)

- **Distribution candles**: `get_candle(X)` and `get_batch_candles(Xs)` return a `Candle` dataclass summarising the lookalike y values — `high, q3, median, q1, low, mean, iqr_mean, std, n` plus `to_dict()`. Pure-distribution object: no temporal order, so OHLC is replaced with quartiles.
- **Regression**: `get_regression(X)` and `get_batch_regression(Xs)` return floats — the IQR-trimmed mean of the lookalike distribution. On a synthetic linear regression (`y = 3·x₁ − 2·x₂ + 1.5·x₃ + 𝒩(0,1)`, n=800 train), R² = **0.9477** vs `get_prediction` R² = 0.8734 (**+7.4pp**) at **2.3× the throughput** (batch shares the Cython lookalike fast path).
- **No new params**: works on any `Snake` model trained with continuous targets. The model is unchanged; only post-processing of the lookalike set is new.
- **10 new tests** in `tests/test_candle.py` (266 tests total)
- **New exports**: `Candle`, `compute_candle` from `algorithmeai`

### v5.4.5 (Mar 2026)

- **Enforced datatypes parameter**: optional `enforced_datatypes` constructor argument lets callers pin the target/feature types instead of relying on auto-detection. Useful when training data is too small to disambiguate `T` (text) from `N` (numeric) and the caller knows the intent.

### v5.4.4 (Mar 2026)

- **New literal type: SIM** — subsequence match ratio with N-style midpoint thresholding. Combines T's substring generation with N's continuous logic. `_text_score(field, ref)` returns the fraction of `ref`'s characters found in `field` in order. O(len(field)), single pass, 8× substring cost. Snake's SAT construction picks the ref string and threshold — the literal stores `[ref, threshold]` and evaluates `score >= threshold`
- **New oppose profile: HEF** — SIM-dominant profile for part number matching. Short alphanumeric strings (5-30 chars) where customers send variant refs for the same catalog article. Weights: SIM:35 TEQ:15 PFX:13 SFX:13 JAC:10 T:8. No LEV — SIM handles fuzzy matching at 30× less cost
- **LEV exact DP raised to 256 chars** — was 32. Bag-of-chars fallback only kicks in above 256 chars. All practical strings now use exact Wagner-Fischer DP
- **Tested on HEF production data**: Pernat (9 articles) `"S700376REC"` → `ARAA03-700376-A2` at **90% confidence** (was 20% with industrial profile). The SIM literal catches that "700376" is a subsequence of "S700376REC"
- **256 tests passing**, zero regressions

### v5.4.0 (Mar 2026)

- **MI-weighted feature selection**: `_precompute_feature_mi()` computes Shannon mutual information `MI(feature; target)` for every feature in O(n×m). All 7 oppose profiles + the original `oppose()` now pick features proportionally to MI weight instead of uniformly. On 500-feature datasets: P(oppose picks informative feature) jumps from 4% to 10.8% (2.7× lift)
- **Lookahead literal selection**: `_oppose_lookahead()` generates K=5 candidate literals per position in `construct_clause`, keeps the one covering the most Ts. Produces 10-13% fewer clauses on classical datasets — tighter clauses generalize better
- **+20pp AUROC on hard high-dimensional data**: 5 informative features out of 500 at signal=1.2: v5.2.1 AUROC 0.526 → v5.4.0 AUROC **0.727**. 50/500 at signal=0.8: 0.711 → **0.850**
- **Zero classical regressions**: Breast Cancer +0.2pp, Iris +0.5pp, Wine +0.1pp (perfect 1.000), Digits -0.1pp (noise)
- **New parameter `lookahead=5`**: Controls literal candidates per oppose call. `1` = v5.2.1 behavior. Stored in JSON config, backwards compatible (defaults to 5 for old models)
- **Worker pipeline**: MI weights and lookahead passed to parallel workers via `_init_worker`
- **256 tests** across 12 files (20 new MI/lookahead tests)
- **Benchmark script**: `benchmark_mi.py` — A/B comparison of v5.2.1 vs v5.4.0 on Hard Madelon + classical datasets

### v5.2.1 (Mar 2026)

- **O(n) string distance**: Levenshtein uses exact DP for strings ≤32 chars, O(n) char-frequency distance for longer strings. No truncation — signal preserved, compute bounded
- **Dataset-specific profiles**: 8 `.snake` profile files in `profiles/` with empirical annotations (PIMA, Breast Cancer, Titanic, Spaceship, Wine Quality, Mushroom, Digits, Adult Income)
- **Snake vs RF/GB benchmark**: Snake beats Random Forest on Breast Cancer AUROC (0.999 vs 0.997). Ties on Iris/Wine. Within 0.8pp on Digits. Same data, zero preprocessing
- **Classic Titanic with Names**: cryptographic profile achieves 0.924 AUROC / 87.2% accuracy (+3.8pp over original)
- **Meta classifier removed** from codebase
- **Linguistic profile deprecated**: never won where expected, blocks training on long text. Pure NLP is out of Snake's scope
- **Cython bool-safe**: `str(field)` casts in `_accel.pyx` handle bool/None values without TypeError

### v5.2.0 (Mar 2026)

- **7 oppose profiles**: `auto`, `balanced`, `linguistic`, `industrial`, `cryptographic`, `scientific`, `categorical`. Each profile is a tuned literal generation strategy — weighted random draws across 6 text families + 5 numeric families. `oppose()` itself is untouched
- **23 new literal types** (30 total): distance (LEV, JAC), positional (PFX, SFX), charclass (TUC, TDC, TSC), crypto (ENT, HEX, REP, CFC), numeric extended (ND, NZ, NL, NMG), exact match (TEQ), affix (TSW, TEW), zero test (NZR), range (NRG), vowel ratio (TVR)
- **FA/TA single-char matching**: `_gen_text_substring` now includes character-level discrimination (chars unique to T or F), matching original `oppose()`'s most powerful pattern
- **Oppose type formalism**: `oppose_types.snake` — complete specification of all 30 literal types + 7 profiles in a human-readable DSL. Defines measures, oppose rules, eval rules, and format templates
- **Auto-detection**: Scans population text features (avg length, variance, digit ratio, special ratio, delimiter density). Pure numeric → scientific, long varied text → linguistic, short codes → industrial
- **Spaceship Titanic**: industrial profile achieves 78.0% optimal accuracy (vs original 77.2%), 0.8038 AUROC. Balanced achieves best AUROC (0.8093). Breast Cancer: scientific hits 98.2% / 0.9987 AUROC (vs original 96.5%)
- **Meta classifier removed** — was experimental, unused in production
- **Cython support**: All 30 literal types in `_accel.pyx` with C-level helpers. Bool-safe `str(field)` casts
- **236 tests** across 11 files (62 new profile tests, 20 Meta tests removed)

### v5.0.0 (Feb 2026)

- **Lookalike origin labeling**: Every lookalike now carries `"c"` (core) or `"n"` (noise) origin. New `get_lookalikes_labeled(X)` method returns `[index, class, condition, origin]` per match. Enables weighted probability with `(w_c, w_n)` tuning
- **Full-population noise**: Noise sourced from the entire population minus core (was: remaining minus core). Deep-chain buckets now access global diversity
- **Origins in JSON**: Each bucket stores `"origins"` parallel to `"members"`. Backwards compatible — old models default to all-core
- **Regime discovery**: Core vs noise signal quality depends on `n_layers`. Low layers = trust core. High layers = blend both

### v4.4.2 (Feb 2026)

- **Perfected audit system**: Two clean AND statements per layer — Routing AND (explains bucket routing) and Lookalike AND (per-sample clause negation). Replaces the stub from v4.3.x
- **Parallel training**: `workers=N` enables multiprocessing for layer construction. Each worker gets a unique RNG seed
- **Progress tracking**: `progress_file` parameter writes JSON progress updates during training
- **Cython batch acceleration**: `batch_get_lookalikes_fast` amortizes routing by grouping queries per bucket per layer

### v4.3.2 (Feb 2026)

- **Logging migration**: Replaced `print()` + string accumulation with Python `logging`. Per-instance logger, buffer handler always captures, StreamHandler to stdout only when `vocal`
- **Extensive test suite**: 92 tests across 7 files. New: core algorithm, validation, edge cases, CLI, logging
- **Cython training acceleration**: 4 new functions in `_accel.pyx` for `construct_clause`, `build_condition`, `_construct_local_sat`
- **Bug fix — Binary True/False targets**: `floatconversion("True")` returned `0.0`, collapsing all True/False targets to 0. Fixed in both flows
- **Spaceship Titanic benchmark**: 78.4% test accuracy (Kaggle top ~80–81%)

## License

Proprietary. Source code is available for viewing and reference only.

See [LICENSE](LICENSE) for details. For licensing inquiries: contact@monce.ai
