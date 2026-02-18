# Snake — Guide for Claude

This is the `algorithmeai-snake` repo: a SAT-based explainable multiclass classifier.
Public at https://github.com/Monce-AI/algorithmeai-snake. Proprietary license (view-only).

Author: Charles Dana / Monce SAS. Charles is the user you're talking to.

## File Layout

```
algorithmeai/
  __init__.py          # Exports: Snake, floatconversion, __version__ = "4.4.2"
  snake.py             # The entire classifier (~1500 lines, zero dependencies)
  _accel.pyx           # Optional Cython hot paths (inference + training acceleration)
  cli.py               # CLI: snake train / predict / info
  __main__.py          # Enables `python -m algorithmeai` for CLI
tests/
  test_snake.py        # 5 input modes, save/load, backwards compat, augmented, vocal, dedup, parallel training
  test_buckets.py      # Bucket chain, noise, routing, audit, dedup
  test_core_algorithm.py  # oppose, construct_clause, construct_sat
  test_validation.py   # make_validation / pruning
  test_edge_cases.py   # Errors, type detection, extreme params, prediction edges
  test_cli.py          # CLI train/predict/info via subprocess
  test_logging.py      # Logging migration validation
  test_audit.py        # Routing AND, Lookalike AND, plain text assertions, audit end-to-end
  test_stress.py       # Stress tests, batch equivalence
  test_ultimate_stress.py  # Extended stress tests
  fixtures/sample.csv  # 15-row toy dataset (color, size, shape -> label A/B/C)
benchmarks.py          # Benchmark script (sklearn + Spaceship Titanic)
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
Snake(Knowledge, target_index=0, excluded_features_index=(), n_layers=5, bucket=250, noise=0.25, vocal=False, saved=False, progress_file=None, workers=1)
```

**Defaults that matter:**
- `n_layers=5` — NOT 100 (old README was wrong). More layers = better accuracy but slower.
- `bucket=250` — max samples per bucket before the chain splits.
- `noise=0.25` — 25% cross-bucket noise for regularization. Set to 0 for strict partitioning.
- `vocal=False` — set to `True` to see training progress and the banner header.
- `saved=False` — only used in CSV flow. If `True`, auto-saves to `snakeclassifier.json` after training.
- `progress_file=None` — if set to a file path, writes JSON progress updates during training (layer/ETA info). Useful for UI progress bars.
- `workers=1` — number of parallel workers for layer construction. `workers > 1` uses `multiprocessing.Pool` for parallel training. Each worker builds one layer with a unique RNG seed.

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
| `get_augmented(X)` | `{**X, "Lookalikes": ..., "Probability": ..., "Prediction": ..., "Audit": ...}` | Single pass — calls get_lookalikes once, derives all else |
| `get_audit(X)` | Multi-line string with Routing AND + Lookalike AND per layer | Human-readable, good for RAG/LLM consumption |

**Critical behavior:** `get_prediction` does NOT raise on unknown keys. Missing keys in `X` cause `apply_literal` to return `False` for that literal — the sample just won't match those clauses. This is by design for partial-input robustness.

### Save & Load

```python
# Save — always writes v4.4.2 bucketed format
model.to_json("model.json")       # or any path
model.to_json()                    # defaults to "snakeclassifier.json"

# Load — auto-detected by .json extension in constructor
model = Snake("model.json")       # skips training entirely
```

**Backwards compatibility:** If the JSON has `clauses` + `lookalikes` at top level but no `layers` key, it's v0.1 flat format. `from_json` wraps it into a single ELSE bucket automatically. No migration needed.

**JSON structure (v4.4.2):**
```json
{
  "version": "4.4.2",
  "population": [...],           // list of dicts (training data)
  "header": ["target", "f1", ...],
  "target": "target",            // target column name
  "targets": [...],              // target values array (parallel to population)
  "datatypes": ["T", "N", ...],  // B=binary, I=integer, N=numeric, T=text, J=complex JSON (dict/list)
  "config": {"n_layers": 5, "bucket": 250, "noise": 0.25, "vocal": false, "workers": 1},
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

## Logging

Snake uses Python's `logging` module internally. Each `Snake` instance gets its own logger (`snake.<id>`).

**Architecture:**
- **Buffer handler** — always attached at DEBUG level, captures everything to `self.log`. This is how `to_json()` persists the training log.
- **Console handler** — attached to `sys.stdout` only when `vocal`:
  - `vocal=True` (or `vocal=1`): StreamHandler at INFO level
  - `vocal >= 2`: StreamHandler at DEBUG level (shows per-target SAT progress)
  - `vocal=False`: no console handler, silent training

**The `qprint` method** is the only logging interface. All 82+ call sites use `self.qprint(msg)` or `self.qprint(msg, level=2)`:
- `level=1` (default) → `logger.info()`
- `level=2` → `logger.debug()`

**The `log` property** reads/writes `self._buffer_handler.buffer`. The banner is always in the buffer (initialized in `__init__`). JSON serialization reads `self.log`, deserialization sets `self.log`.

**The banner** prints directly via `print(_BANNER)` when `vocal=True`, preserving exact legacy behavior. It is NOT sent through the logger to avoid duplicate output.

## The Banner Print

As of v4.3.3+, the banner only prints when `vocal=True`. Previously it printed unconditionally.

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
- 8693 samples (Spaceship Titanic), 50 layers: ~7.4ms per prediction

## CLI

Installed as `snake` command via the `[project.scripts]` entry in `pyproject.toml`:

```toml
[project.scripts]
snake = "algorithmeai.cli:main"
```

After `pip install -e .`, the `snake` command is available system-wide. Entry point: `algorithmeai/cli.py:main()`.

Also runnable as `python -m algorithmeai` via `algorithmeai/__main__.py`.

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
#   Snake model v4.4.2
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
pytest                                # runs all 174 tests
pytest tests/test_snake.py            # input modes, save/load, augmented, vocal, dedup, parallel training
pytest tests/test_buckets.py          # bucket chain, noise, routing, audit, dedup
pytest tests/test_core_algorithm.py   # oppose, construct_clause, construct_sat
pytest tests/test_validation.py       # make_validation / pruning
pytest tests/test_edge_cases.py       # errors, type detection, extreme params, prediction edges
pytest tests/test_cli.py              # CLI train/predict/info via subprocess
pytest tests/test_logging.py          # logging buffer, JSON persistence, banner
pytest tests/test_audit.py            # Routing AND, Lookalike AND, plain text assertions, audit end-to-end
pytest tests/test_stress.py           # stress tests, batch equivalence
pytest tests/test_ultimate_stress.py  # extended stress tests
```

174 tests across 10 files. Tests use `tests/fixtures/sample.csv` (15 rows, 3 classes). All tests use small `n_layers` (1-3) and `bucket` (3-5) for speed.

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

Snake includes optional Cython-accelerated hot paths in `algorithmeai/_accel.pyx`:

**Inference functions (v4.3.0):**
- `apply_literal_fast` — single literal evaluation
- `apply_clause_fast` — OR over literals
- `traverse_chain_fast` — IF/ELIF/ELSE chain walk
- `get_lookalikes_fast` — full inference pipeline
- `batch_predict_fast` — batch inference

**Batch acceleration (v4.4.2):**
- `batch_get_lookalikes_fast` — batch lookalike computation with amortized routing (groups queries by bucket per layer)

**Training functions (v4.3.2):**
- `filter_ts_remainder_fast` — filter Ts by literal (used in construct_clause)
- `minimize_clause_fast` — full clause minimization loop in C
- `filter_indices_by_literal_fast` — index-based population filtering (used in build_condition)
- `filter_consequence_fast` — consequence + remaining Fs filtering (used in _construct_local_sat)

```python
try:
    from ._accel import (apply_literal_fast, apply_clause_fast,
                         traverse_chain_fast, get_lookalikes_fast,
                         batch_predict_fast, batch_get_lookalikes_fast,
                         filter_ts_remainder_fast, minimize_clause_fast,
                         filter_indices_by_literal_fast,
                         filter_consequence_fast)
    _HAS_ACCEL = True
except ImportError:
    _HAS_ACCEL = False
```

Install with:

```bash
pip install -e ".[fast]" && python setup.py build_ext --inplace
```

Without Cython, pure Python runs identically. The `_HAS_ACCEL` flag gates all fast paths — training and inference methods check it before dispatching.

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
#   - Lookalike summary with percentages and examples per class
#   - Probability distribution
#   - Per-layer detail:
#     - Routing AND: why X was routed to this bucket (negated skip literals + passing condition)
#     - Lookalike AND: per-lookalike explanation of matching clause negations
#   - Final prediction
# Feed this to an LLM for explanation generation
```

### Audit structure (v4.4.2)

The audit system produces two AND statements per layer:

**Routing AND:** Explains why X was routed to a particular bucket in the IF/ELIF/ELSE chain. For each skipped branch, it shows the negated first-failing literal. For the matching branch, it shows the passing condition literals.

**Lookalike AND:** For each matching lookalike in the bucket, shows the AND of negated clause literals that explain why the lookalike was matched. A lookalike matches when ALL its associated clauses evaluate to FALSE on X. Each clause is an OR of literals, so FALSE means every literal in the clause is FALSE. Negating each literal gives a human-readable description of what IS true.

```
### BEGIN AUDIT ###
  Prediction: ClassA
  Layers: 5, Lookalikes: 12

  LOOKALIKE SUMMARY
  ================================================
  ClassA               60.0% (3/5) ████████████░░░░░░░░
    e.g. 44.2 LowE
  ClassB               40.0% (2/5) ████████░░░░░░░░░░░░
    e.g. Float 4mm

  PROBABILITY
  ================================================
  P(ClassA) = 60.0% ████████████░░░░░░░░
  P(ClassB) = 40.0% ████████░░░░░░░░░░░░

  ================================================
  LAYER 0
  ================================================

  Routing AND (bucket 2/4, 15 members):
    "size" <= 10 AND "color" does NOT contain "r" AND "shape" contains "round"

  Lookalike AND (3 matches):
    Lookalike #42 [ClassA]: 44.2 LowE
      AND: "denomination" does NOT contain "float" AND "client" does NOT contain "VIT"
    Lookalike #17 [ClassA]: Stadip 44.2
      AND: "denomination" does NOT contain "float"

  >> PREDICTION: ClassA
### END AUDIT ###
```

## Things That Will Bite You

1. **The banner prints to stdout with `vocal=True` only.** Set `vocal=False` to suppress.

2. **`get_augmented` is efficient.** It calls `get_lookalikes` once and derives probability, prediction, and audit from that single pass. No longer 4x overhead.

3. **Target type must match at prediction time.** If training targets were `int` (e.g., `0`, `1`), your prediction comparison is against `int`. If they were `str` (e.g., `"cat"`, `"dog"`), comparison is against `str`. Mixing types silently fails to match.

4. **Feature types are fixed at training time.** If a feature was detected as `"N"` (numeric), `apply_literal` does numeric comparison. If you pass a string at prediction time for that feature, you'll get a `TypeError`. The training data determines the contract.

5. **CSV must be pandas-formatted.** The CSV parser (`make_bloc_from_line`) handles quoted fields with commas but expects `pandas.DataFrame.to_csv()` format. Non-pandas CSVs may parse incorrectly.

6. **No incremental training.** To add data, you rebuild from scratch. `make_validation` only prunes layers — it doesn't add new ones.

7. **Model JSON contains the full training population.** A model trained on 5000 rows with 20 features produces a large JSON file. Be mindful of disk/memory.

8. **`exclude_features_index` only works with CSV flow.** The list[dict] and DataFrame flows don't support it — filter your data before passing it in.

9. **Binary True/False targets are stored as int 0/1.** As of v4.3.2+, `"True"`/`"False"` strings are correctly converted in both CSV and list[dict] flows. Compare predictions against `0`/`1` (int), not `"True"`/`"False"` (str).

## Changelog

### v4.4.2 (Feb 2026)

- **Perfected audit system**: Two clean AND statements per layer — Routing AND (why X was routed to this bucket) and Lookalike AND (per-sample clause negation explanation). Replaces the v4.3.x stub that only showed clause indices.
- **Parallel training**: `workers=N` parameter enables multiprocessing.Pool for layer construction. Each worker builds one layer with a unique RNG seed. `workers=1` (default) preserves sequential behavior.
- **Progress file**: `progress_file` parameter writes JSON progress updates during training with layer/ETA info, useful for UI integration.
- **Cython batch acceleration**: `batch_get_lookalikes_fast` groups queries by routed bucket per layer, amortizing chain traversal. Used by `get_batch_prediction` when Cython is compiled.
- **Lookalike summary**: Audit now starts with a summary showing actual feature examples and percentages per class.
- **174 tests**: Extended test suite from 151 to 174 across 10 files (added audit tests, parallel training tests, batch equivalence tests).
- **New methods**: `_negate_literal`, `_first_failing_literal`, restored `get_plain_text_assertion` with bucket-aware clause resolution.

### v4.3.3 (Feb 2026)

- **Cython infinite loop fix**: Fixed `oppose()` infinite loop in Cython hot paths
- **`oppose()` canaries**: Added safety canaries to detect stuck loops
- **151 tests**: Extended test suite from 92 to 151 tests across 9 files (added stress + ultimate stress tests)

### v4.3.2 (Feb 2026)

- **Logging migration**: Replaced `print()` + string accumulation with Python `logging`. Per-instance logger, buffer handler always captures, StreamHandler to stdout only when `vocal`. All 82 `qprint` call sites unchanged.
- **Extensive test suite**: 92 tests across 7 files (was ~41). New: core algorithm, validation, edge cases, CLI, logging.
- **Cython training acceleration**: 4 new functions in `_accel.pyx` for `construct_clause`, `build_condition`, `_construct_local_sat`. Expected 3-5x training speedup when compiled.
- **Bug fix — Binary True/False targets**: `floatconversion("True")` returned `0.0`, collapsing all True/False targets to 0. Fixed in both `_init_from_data` and `make_population`.
- **Spaceship Titanic benchmark**: 78.4% test accuracy (Kaggle top ~80-81%), added to benchmarks.py and README.
- **`__main__.py`**: Enables `python -m algorithmeai` as CLI entry point.
