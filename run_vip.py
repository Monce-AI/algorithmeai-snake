"""
VIP Factory 4 — Snake v4.2.0 assessment
Train if model missing, otherwise load and evaluate.
"""
import os
import sys
from monce_db import connect
from algorithmeai import Snake
from time import time


def progress_bar(current, total, width=40, prefix=""):
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  {prefix}|{bar}| {current}/{total} ({pct*100:.1f}%)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")

MODEL_PATH = "vip_snake_v420.json"

# ── Fetch training data ──
print("Fetching VIP articles + snake_learning from monce_db...")
client = connect()
articles = client.fetch_articles(factory=4)
sl = client.fetch_snake_learning(factory=4)

# ── Build training rows ──
rows = []
seen = set()

for a in articles:
    if a.denomination:
        key = (a.num_article, a.denomination)
        if key not in seen:
            seen.add(key)
            rows.append({"num_article": a.num_article, "denomination": a.denomination})
    if a.synonymes:
        for syn in a.synonymes:
            key = (a.num_article, syn)
            if key not in seen and syn:
                seen.add(key)
                rows.append({"num_article": a.num_article, "denomination": syn})

for s in sl:
    key = (s.num_article, s.synonym)
    if key not in seen and s.synonym:
        seen.add(key)
        rows.append({"num_article": s.num_article, "denomination": s.synonym})

n_classes = len(set(r["num_article"] for r in rows))
print(f"Training rows: {len(rows)}, unique classes: {n_classes}")

# ── Train or load ──
if os.path.exists(MODEL_PATH):
    print(f"\nModel found at {MODEL_PATH}, loading...")
    t0 = time()
    s = Snake(MODEL_PATH, vocal=True)
    print(f"Loaded in {time() - t0:.2f}s")
else:
    print(f"\nNo model at {MODEL_PATH}, training from scratch...")
    t0 = time()
    s = Snake(rows, n_layers=5, bucket=250, noise=0.25, vocal=True)
    elapsed = time() - t0
    print(f"\nTrained in {elapsed:.1f}s")
    for i, chain in enumerate(s.layers):
        total_clauses = sum(len(b['clauses']) for b in chain)
        print(f"  Layer {i}: {len(chain)} buckets, {total_clauses} clauses")
    s.to_json(MODEL_PATH)

# ── 1) Perfect fit on training basis ──
print(r"""
╔══════════════════════════════════════════════════════════╗
║   ___         __        _     ___ _ _                   ║
║  | _ \___ _ _/ _|___ __| |_  | __(_) |_                 ║
║  |  _/ -_) '_|  _/ -_) _|  _|| _|| |  _|                ║
║  |_| \___|_| |_| \___\__|\__||_| |_|\__|                ║
║                                                          ║
║  Training basis accuracy assessment                      ║
╚══════════════════════════════════════════════════════════╝""")

correct = 0
wrong = 0
errors = []
t0 = time()

for i, row in enumerate(rows):
    X = {"denomination": row["denomination"]}
    pred = s.get_prediction(X)
    expected = row["num_article"]
    if str(pred) == str(expected):
        correct += 1
    else:
        wrong += 1
        if len(errors) < 20:
            errors.append((row["denomination"], expected, pred))
    if (i + 1) % 50 == 0 or i + 1 == len(rows):
        progress_bar(i + 1, len(rows), prefix="Fit ")

fit_time = time() - t0
total = correct + wrong
accuracy = correct / total if total else 0

print(f"  Samples:   {total}")
print(f"  Correct:   {correct}")
print(f"  Wrong:     {wrong}")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Time:      {fit_time:.2f}s")

if errors:
    print(f"\n  First {len(errors)} mismatches:")
    for denom, expected, pred in errors:
        print(f"    '{denom}' -> predicted {pred}, expected {expected}")

# ── 2) Inference time ──
print(r"""
╔══════════════════════════════════════════════════════════╗
║   ___       __                                          ║
║  |_ _|_ _  / _|___ _ _ ___ _ _  __ ___                  ║
║   | || ' \|  _/ -_) '_/ -_) ' \/ _/ -_)                 ║
║  |___|_||_|_| \___|_| \___|_||_\__\___|                 ║
║                                                          ║
║  Latency & throughput benchmarks                         ║
╚══════════════════════════════════════════════════════════╝""")

avg_ms = (fit_time / total * 1000) if total else 0
print(f"  Total inference: {fit_time:.2f}s for {total} samples")
print(f"  Avg per query:   {avg_ms:.2f}ms")
print(f"  Throughput:      {total / fit_time:.0f} queries/sec" if fit_time > 0 else "  Throughput:      N/A")

# ── 3) Interesting synonyms ──
print(r"""
╔══════════════════════════════════════════════════════════╗
║   ___                                                    ║
║  / __|_  _ _ _  ___ _ _ _  _ _ __  ___                   ║
║  \__ \ || | ' \/ _ \ ' \ || | '  \(_-<                   ║
║  |___/\_, |_||_\___/_||_\_, |_|_|_/__/                   ║
║       |__/              |__/                              ║
║                                                          ║
║  Glass industry term matching                            ║
╚══════════════════════════════════════════════════════════╝""")

test_synonyms = [
    # Classic glass variants — the black hole zone
    ("44.2", "Base feuilleté"),
    ("44.2 LowE", "LowE variant"),
    ("44.2 ITR", "ITR variant"),
    ("4mm clair", "Base 4mm"),
    ("4mm Planitherm", "Planitherm variant"),
    # OCR-degraded inputs
    ("Stopso1 gris", "OCR: Stopsol gris"),
    ("Parsol bronz", "OCR: Parsol bronze"),
    ("P1anitherm", "OCR: Planitherm"),
    # Abbreviations and jargon
    ("VT4", "Shorthand"),
    ("TGI noir", "Intercalaire"),
    ("16 THN", "Spacer"),
    # Full denominations
    ("Float 4mm clair", "Float glass"),
    ("Satinovo", "Satin glass"),
    ("Stadip 44.2", "Laminated"),
]

syn_results = []
for i, (query, label) in enumerate(test_synonyms):
    X = {"denomination": query}
    t_q = time()
    pred = s.get_prediction(X)
    prob = s.get_probability(X)
    t_q = time() - t_q
    confidence = max(prob.values()) * 100
    syn_results.append((label, query, pred, confidence, t_q))
    progress_bar(i + 1, len(test_synonyms), prefix="Syn ")

for label, query, pred, confidence, t_q in syn_results:
    print(f"  [{label:.<25s}] '{query}' -> article {pred} ({confidence:.1f}% conf, {t_q*1000:.1f}ms)")

print(r"""
╔══════════════════════════════════════════════════════════╗
║          ___ _  _   _   _  _____                         ║
║         / __| \| | /_\ | |/ / __|                        ║
║         \__ \ .` |/ _ \|   <| _|                         ║
║         |___/_|\_/_/ \_\_|\_\___|                         ║
║                                                          ║
║         v4.2.0 — Assessment complete                     ║
╚══════════════════════════════════════════════════════════╝""")
