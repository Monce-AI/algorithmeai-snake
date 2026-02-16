# Snake — Guide for Claude

This is the `algorithmeai-snake` repo: a SAT-based explainable multiclass classifier.
Public at https://github.com/Monce-AI/algorithmeai-snake. Proprietary license (view-only).

Author: Charles Dana / Monce SAS. Charles is the user you're talking to.

## File Layout

```
algorithmeai/
  __init__.py          # Exports: Snake, floatconversion, __version__ = "4.3.0"
  snake.py             # The entire classifier (~1200 lines, zero dependencies)
  _accel.pyx           # Optional Cython hot paths (apply_literal, apply_clause, traverse_chain)
  cli.py               # CLI: snake train / predict / info
tests/
  test_snake.py        # 5 input modes, save/load, backwards compat
  test_buckets.py      # Bucket chain, noise, routing, audit, dedup
  fixtures/sample.csv  # 15-row toy dataset (color, size, shape -> label A/B/C)
benchmarks.py          # Benchmark script (requires pandas + scikit-learn)
pyproject.toml         # Build config, hatchling, proprietary license
README.md              # Public-facing docs
LICENSE                # Proprietary
```

## How to Use Snake

### The Constructor — 5 Input Modes

```python
from algorithmeai import Snake
```

Snake's constructor accepts ONE positional argument `Knowledge` and dispatches based on type:

| Input | Detection Rule | `target_index` Behavior |
|-------|---------------|------------------------|
| `"file.json"` | `str` ending `.json` | Ignored — loads saved model, skips training |
| `"file.csv"` | `str` ending `.csv` | `int` column index (default `0`) |
| `list[dict]` | `list` where `first` is `dict` | `str` key name or `int` (default `0` = first key) |
| `list[tuple\|list]` | `list` where `first` is `tuple`/`list` | Ignored — first element is always target, headers auto-generated as `target, f1, f2, ...` |
| `list[str\|int\|float]` | `list` of scalars | Ignored — self-classing mode, target = value itself |
| DataFrame | Has `.to_dict()` method (duck-typed) | `str` column name or `int` index |

**IMPORTANT:** For `list[dict]`, the **first key** of the first dict becomes the target by default. Key order matters. If you need a different target, pass `target_index="column_name"`.

### Constructor Signature

```python
Snake(Knowledge, target_index=0, excluded_features_index=(), n_layers=5, bucket=250, noise=0.25, vocal=False, saved=False)
```

**Defaults that matter:**
- `n_layers=5` — NOT 100 (old README was wrong). More layers = better accuracy but slower.
- `bucket=250` — max samples per bucket before the chain splits.
- `noise=0.25` — 25% cross-bucket noise for regularization. Set to 0 for strict partitioning.
- `vocal=False` — set to `True` to see training progress and the banner header.
- `saved=False` — only used in CSV flow. If `True`, auto-saves to `snakeclassifier.json` after training.

### Production Pattern (list[dict])

This is how snake-api uses it:

```python
# Build training data as list of dicts
data = [
    {"article": "60442", "denomination": "44.2 LowE", "client": "Riou"},
    {"article": "1004",  "denomination": "Float 4mm", "client": "VIT"},
    # ... hundreds/thousands of rows
]

# Train
model = Snake(data, n_layers=5, bucket=250, noise=0.25, vocal=False)

# Save for production serving
model.to_json("model.json")

# Later: load without retraining
model = Snake("model.json")

# Predict
result = model.get_prediction({"denomination": "44.2 LowE", "client": "Riou"})
confidence = model.get_probability({"denomination": "44.2 LowE", "client": "Riou"})
```

### Prediction Methods

All take a dict `X` with feature keys (NOT the target key):

| Method | Returns | Notes |
|--------|---------|-------|
| `get_prediction(X)` | The target value with highest probability | If no lookalikes found, returns random class (uniform distribution) |
| `get_probability(X)` | `{class: float}` dict, sums to 1.0 | If 0 lookalikes: uniform `1/n_classes` for each class |
| `get_lookalikes(X)` | `[[global_idx, target_value, condition], ...]` | Deduped by population index across layers |
| `get_augmented(X)` | `{**X, "Lookalikes": ..., "Probability": ..., "Prediction": ..., "Audit": ...}` | Calls all 4 methods (4x the work) |
| `get_audit(X)` | Multi-line string with per-layer bucket routing + vote breakdown | Human-readable, good for RAG/LLM consumption |

**Critical behavior:** `get_prediction` does NOT raise on unknown keys. Missing keys in `X` cause `apply_literal` to return `False` for that literal — the sample just won't match those clauses. This is by design for partial-input robustness.

### Save & Load

```python
# Save — always writes v4.3.0 bucketed format
model.to_json("model.json")       # or any path
model.to_json()                    # defaults to "snakeclassifier.json"

# Load — auto-detected by .json extension in constructor
model = Snake("model.json")       # skips training entirely
```

**Backwards compatibility:** If the JSON has `clauses` + `lookalikes` at top level but no `layers` key, it's v0.1 flat format. `from_json` wraps it into a single ELSE bucket automatically. No migration needed.

**JSON structure (v4.3.0):**
```json
{
  "version": "4.3.0",
  "population": [...],           // list of dicts (training data)
  "header": ["target", "f1", ...],
  "target": "target",            // target column name
  "targets": [...],              // target values array (parallel to population)
  "datatypes": ["T", "N", ...],  // B=binary, I=integer, N=numeric, T=text, J=complex JSON (dict/list)
  "config": {"n_layers": 5, "bucket": 250, "noise": 0.25, "vocal": false},
  "layers": [...],               // bucketed layer data
  "log": "..."                   // training log string
}
```

### Validation & Pruning

```python
# val_data must include the target field
model.make_validation(val_data, pruning_coef=0.5)
# Keeps the best 50% of layers by validation accuracy
# Modifies model in-place (self.layers, self.n_layers)
model.to_json("pruned.json")
```

`make_validation` silently skips any sample in `Xs` where `self.target` is not a key. Make sure your validation dicts include the target.

## Data Type Detection

Snake auto-detects types for each column. The logic (in order of priority):

**Target column (`_detect_target_type`):**
1. Any value is `dict` or `list` → Complex JSON (`J`), stored as-is (raw Python objects)
2. Values are exactly `{"0", "1"}` → Binary (`B`), stored as `int`
3. Values are exactly `{"True", "False"}` or `{"TRUE", "FALSE"}` → Binary (`B`), stored as `int`
4. All characters in `0-9` only → Integer multiclass (`I`), stored as `int`
5. All characters in `+-.0123456789e` → Numeric multiclass (`N`), stored as `float`
6. Otherwise → Text multiclass (`T`), stored as `str`

**Feature columns:** Same numeric vs text test (step 4 vs 5 above). No Binary/Integer distinction for features.

**Gotcha:** A column with values like `"3.0", "4.0", "5.0"` is detected as Numeric, NOT Integer. A column with `"3", "4", "5"` is Integer. The target type affects how `self.targets` is stored and compared — this matters for prediction equality checks.

**Gotcha:** `floatconversion` silently converts unparseable strings to `0.0`. If your numeric column has `"N/A"` or `""`, those become `0.0` without warning.

## Deduplication

Both CSV and list[dict] flows deduplicate by hashing all **feature values** (not the target). If two rows have identical features but different targets, the second is dropped with a log message. This is intentional — conflicting training data degrades SAT clause quality.

The hash is a string concatenation of all feature values (not a proper hash function). It's fast but depends on string representation stability.

## The Banner Print

As of v4.3.0, the banner only prints when `vocal=True`. Previously it printed unconditionally.

## Error Handling

Snake has minimal error handling by design. Here's what can go wrong:

| Situation | What Happens | How to Avoid |
|-----------|-------------|-------------|
| Empty list | `ValueError("Knowledge must be a non-empty list...")` | Check before constructing |
| Empty DataFrame | `ValueError("Empty DataFrame")` | Check `len(df) > 0` |
| Wrong file extension | Falls through to `_init_from_data` on a string, crashes | Use `.csv` or `.json` extensions |
| CSV file not found | `FileNotFoundError` from `open()` | Check path exists |
| JSON file not found | `FileNotFoundError` from `open()` | Check path exists |
| Malformed JSON | `json.JSONDecodeError` | Validate JSON before loading |
| `target_index` string not in keys | `ValueError` from `list.index()` | Check column name exists |
| `target_index` int out of range | `IndexError` | Check column count |
| Only 1 unique target value | Trains but predictions are trivially that class | Need at least 2 classes |
| Prediction with empty dict `{}` | Returns uniform probability (all classes equal) | Pass at least some features |
| Prediction with unknown keys | Unknown keys ignored, known keys matched | Use the same key names as training |
| Numeric feature passed as string | `apply_literal` does string comparison, not numeric | Ensure types match training data |
| `None` values in data | Converted to `"None"` string or `0.0` float | Clean your data |

**No exceptions are raised during prediction.** `get_prediction`, `get_probability`, `get_lookalikes`, `get_audit` all handle edge cases gracefully (worst case: uniform probability, empty lookalikes).

## Performance Characteristics

- **Training:** `O(n_layers * n_buckets * m * bucket_size^2)` per layer. Dominated by SAT clause construction.
- **Inference:** `O(n_layers * n_clauses_in_matched_bucket)` per prediction. Fast for small bucket sizes.
- **Memory:** Entire population is stored in memory (list of dicts). Model JSON includes full training data.

**Scaling guidance:**
- < 500 samples: `bucket=250` works fine, single bucket per layer
- 500-5000 samples: `bucket=250` creates 2-20 buckets per layer, good performance
- 5000+ samples: Training gets slow. Consider `n_layers=3-5` and `bucket=500`
- 10000+ samples: Use Divide & Conquer (segment by a certain field, train per-segment models)

**Inference latency (from benchmarks):**
- 150 samples, 5 layers: ~0.2ms per prediction
- 1797 samples, 50 layers: ~12ms per prediction

## CLI

Installed as `snake` command via the `[project.scripts]` entry in `pyproject.toml`:

```toml
[project.scripts]
snake = "algorithmeai.cli:main"
```

After `pip install -e .`, the `snake` command is available system-wide. Entry point: `algorithmeai/cli.py:main()`.

### `snake train`

Trains a Snake model from a CSV file and saves it to JSON.

```bash
snake train data.csv [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `csv` (positional) | str | required | Path to CSV file. Must end in `.csv`. |
| `--target` | int | `0` | Target column index in the CSV |
| `--layers` | int | `5` | Number of SAT layers to build |
| `--bucket` | int | `250` | Max samples per bucket |
| `--noise` | float | `0.25` | Cross-bucket noise ratio |
| `--output`, `-o` | str | `snakeclassifier.json` | Output JSON path |
| `--vocal` | flag | off | Print training progress to stdout |

```bash
# Minimal
snake train iris.csv -o iris.json

# Full control
snake train titanic.csv --target 0 --layers 15 --bucket 500 --noise 0.1 -o titanic.json --vocal
```

**Note:** `train` always passes `saved=True` to Snake, so it also auto-saves to the default path. Then it explicitly calls `to_json(args.output)` again. The output at `args.output` is the one you want.

**Note:** `train` only works with CSV files. To train from list[dict] or DataFrame, use the Python API.

### `snake predict`

Predicts a single datapoint using a saved model.

```bash
snake predict model.json -q '{"feature": "value"}' [--audit]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `model` (positional) | str | required | Path to saved JSON model |
| `--query`, `-q` | str (JSON) | required | JSON dict to classify |
| `--audit` | flag | off | Print full audit trace after prediction |

```bash
# Basic prediction
snake predict model.json -q '{"color": "red", "size": 10}'
# Output:
#   Prediction: A
#   Confidence: 93.3%

# With audit
snake predict model.json -q '{"color": "red", "size": 10}' --audit
# Output:
#   Prediction: A
#   Confidence: 93.3%
#   ==================================================
#     LAYER 0
#   ...
```

**Note:** The `--query` value must be valid JSON parseable by `json.loads()`. Use single quotes around the JSON and double quotes inside (shell-safe). Feature values must match the types from training — pass numbers as numbers, strings as strings.

**Note:** The banner header is always printed to stdout when loading the model. The prediction output comes after it.

### `snake info`

Shows model metadata without loading the full model into Snake.

```bash
snake info model.json
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `model` (positional) | str | required | Path to saved JSON model |

```bash
snake info model.json
# Output:
#   Snake model v4.3.0
#   Target: species
#   Population: 150
#   Layers: 5
```

**Note:** `info` reads the JSON directly with `json.load()` — it does NOT instantiate a `Snake` object, so the banner is not printed and there's no overhead.

### CLI Limitations

- No `validate` / `prune` command — use the Python API for `make_validation()`
- No batch prediction — one query at a time via `--query`
- No `list[dict]` or DataFrame input — CSV only for `train`
- No `--seed` flag — Snake is non-deterministic by design
- `predict` loads the full model each time — for batch serving, use the Python API

## Testing

```bash
pytest                    # runs all tests
pytest tests/test_snake.py     # input modes + save/load
pytest tests/test_buckets.py   # bucket chain + audit
```

Tests use `tests/fixtures/sample.csv` (15 rows, 3 classes). All tests use small `n_layers` (1-3) and `bucket` (3-5) for speed.

## Stochastic Behavior

Snake training is **non-deterministic**. The `oppose()` function uses `random.choice` extensively. Two runs on the same data produce different models with different clause sets. This is by design — the power comes from stochastic clause generation + deterministic selection.

**Consequence:** Test assertions must be probabilistic (e.g., "at least 50% of training data has >50% confidence") rather than exact value checks. Benchmark numbers are approximate.

There is no `random.seed()` call in the Snake code. If you need reproducibility, set the seed externally before constructing:

```python
import random
random.seed(42)
model = Snake(data, n_layers=5)
```

## Optional Cython Acceleration

Snake v4.3.0 includes optional Cython-accelerated hot paths in `algorithmeai/_accel.pyx`:

```python
try:
    from ._accel import apply_literal_fast, apply_clause_fast, traverse_chain_fast
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False
```

When the compiled extension is available, `apply_literal`, `apply_clause`, and `get_lookalikes` delegate to the Cython versions. Install with:

```bash
pip install -e ".[fast]" && python setup.py build_ext --inplace
```

Without Cython, pure Python runs identically. The `_HAS_ACCEL` flag is checked at function call time.

## Common Patterns

### Threshold-based routing (production hybrid)

```python
model = Snake("model.json")
prob = model.get_probability(X)
confidence = max(prob.values())
prediction = model.get_prediction(X)

if confidence >= 0.51:
    return prediction          # Snake is confident
else:
    return fuzzy_fallback(X)   # Fall back to fuzzy matcher
```

### Adding a synonym (retrain)

```python
# Append new synonym to training data
data.append({"article": "60442", "denomination": "44.2 feuilleté"})
# Retrain from scratch — Snake has no incremental add
model = Snake(data, n_layers=5, bucket=250)
model.to_json("model.json")
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

### Using the audit for RAG / LLM consumption

```python
audit = model.get_audit(X)
# audit is a multi-line string with:
#   - Per-layer bucket routing (IF/ELIF/ELSE)
#   - Per-bucket lookalike vote breakdown
#   - Global summary with probabilities
#   - Final prediction
# Feed this to an LLM for explanation generation
```

## Things That Will Bite You

1. **The banner prints to stdout with `vocal=True` only.** As of v4.3.0, set `vocal=False` to suppress the banner.

2. **`get_augmented` is now efficient.** As of v4.3.0, it calls `get_lookalikes` once and derives probability, prediction, and audit from that single pass. No longer 4x overhead.

3. **Target type must match at prediction time.** If training targets were `int` (e.g., `0`, `1`), your prediction comparison is against `int`. If they were `str` (e.g., `"cat"`, `"dog"`), comparison is against `str`. Mixing types silently fails to match.

4. **Feature types are fixed at training time.** If a feature was detected as `"N"` (numeric), `apply_literal` does numeric comparison. If you pass a string at prediction time for that feature, you'll get a `TypeError`. The training data determines the contract.

5. **CSV must be pandas-formatted.** The CSV parser (`make_bloc_from_line`) handles quoted fields with commas but expects `pandas.DataFrame.to_csv()` format. Non-pandas CSVs may parse incorrectly.

6. **No incremental training.** To add data, you rebuild from scratch. `make_validation` only prunes layers — it doesn't add new ones.

7. **Model JSON contains the full training population.** A model trained on 5000 rows with 20 features produces a large JSON file. Be mindful of disk/memory.

8. **`exclude_features_index` only works with CSV flow.** The list[dict] and DataFrame flows don't support it — filter your data before passing it in.
