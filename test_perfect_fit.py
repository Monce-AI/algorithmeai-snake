"""
Snake v5.4.3 — Perfect Fit Assessment
======================================

Tests that every training datapoint gets P(actual_class) = 1.0.


WHY CLAUSE TEST PASSES BUT LOOKALIKE TEST CAN FAIL
===================================================

The clause contract is verified PER CLAUSE during construction:

  construct_clause(F, Ts):
    loop until Ts_remain is empty
    --> clause is TRUE on ALL Ts, FALSE on F    ......... VERIFIED IN WORKER PROCESS

But workers run in separate processes (multiprocessing).
If the verification exit() fires, the WORKER dies silently.
The main process never sees it.

  WORKER PROCESS                    MAIN PROCESS
  +--------------------------+      +------------------+
  | construct_clause(F, Ts)  |      |                  |
  | clause = [lit1, lit2...] |      |                  |
  |                          |      |                  |
  | VERIFY: clause TRUE on T |      |                  |
  |   -> FATAL! exit()  ----X|  ??  | pool.imap()      |
  |   (worker dies, main     |      | "layer returned" |
  |    may not see it)       |      |                  |
  +--------------------------+      +------------------+

The LOOKALIKE test runs in the MAIN process after layer assembly.
It simulates the EXACT inference path:

  +--------------------------------------------------+
  | For each bucket in layer:                        |
  |   For each member X:                             |
  |     Evaluate ALL clauses on X (ONCE)             |
  |     negated = clauses that are FALSE on X        |
  |                                                  |
  |     For each member L (different class than X):  |
  |       If L's condition is fully in negated:      |
  |         --> WRONG-CLASS LOOKALIKE                 |
  |         --> Write /tmp/snake_fatal.json           |
  |         --> exit() (main process, not swallowed)  |
  +--------------------------------------------------+

This catches the ACTUAL behavior: if a clause built in a worker
was supposed to be TRUE on T but ISN'T (due to pickling, Cython
mismatch, or any other issue), the lookalike test catches it
because it runs the SAME apply_clause in the SAME process that
will later do inference.


THE PERFECT FIT INVARIANT
=========================

For training sample X of class C in bucket B:

  Class C SAT:                    Class D SAT (D != C):
  +------------------------+      +------------------------+
  | clause_a: FALSE on X   |      | clause_p: TRUE on X    |
  |   (X was an F, covered)|      |   (X was a T, covered) |
  | clause_b: maybe T/F    |      | clause_q: TRUE on X    |
  +------------------------+      +------------------------+
           |                                 |
           v                                 v
  X's condition = [a]             D-member's condition = [p]
  -> a in negated? YES            -> p in negated? NO (TRUE on X)
  -> X is own lookalike (C)       -> D-member NOT a lookalike

  Result: ALL lookalikes of X are class C --> P(C) = 1.0
"""

import json
import sys
import time
import os

DATA_PATH = "/Users/charlesdana/Documents/chess.aws.monce.ai/data/vectorize/charles_w/train_110.json"
MODEL_PATH = "/tmp/chess_13k_v543.json"


def train_and_save():
    from algorithmeai import Snake

    with open(DATA_PATH) as f:
        data = json.load(f)

    print(f"Training: {len(data)} rows, workers=10, layers=10")
    t0 = time.time()
    model = Snake(data, target_index="target", n_layers=10, bucket=250,
                  noise=0.25, workers=10, vocal=False)
    print(f"Done in {time.time() - t0:.1f}s — saving to {MODEL_PATH}")
    model.to_json(MODEL_PATH)
    return model, data


def assess(model, data):
    n = len(data)
    perfect_pred = 0
    perfect_prob = 0
    failures = []

    for i in range(n):
        features = {k: v for k, v in data[i].items() if k != "target"}
        actual = model.targets[i]

        pred = model.get_prediction(features)
        prob = model.get_probability(features)
        p_actual = prob.get(actual, 0.0)

        if pred == actual:
            perfect_pred += 1
        if p_actual == 1.0:
            perfect_prob += 1
        else:
            failures.append({
                "row": i,
                "actual": actual,
                "pred": pred,
                "p_actual": round(p_actual, 6),
                "p_pred": round(prob.get(pred, 0.0), 6),
            })

    print(f"\n{'='*60}")
    print(f"  PERFECT FIT ASSESSMENT — Snake v5.4.3")
    print(f"{'='*60}")
    print(f"  Samples:          {n}")
    print(f"  Correct pred:     {perfect_pred}/{n} ({100*perfect_pred/n:.2f}%)")
    print(f"  P(actual)=1.0:    {perfect_prob}/{n} ({100*perfect_prob/n:.2f}%)")
    print(f"  Failures:         {len(failures)}")
    print(f"{'='*60}")

    if failures:
        print(f"\n  First 10 failures:")
        for f in failures[:10]:
            print(f"    Row {f['row']}: actual={f['actual']} pred={f['pred']} "
                  f"P(actual)={f['p_actual']} P(pred)={f['p_pred']}")

    if perfect_prob == n:
        print(f"\n  PASS")
        return True
    else:
        print(f"\n  FAIL — {len(failures)} samples without P(actual)=1.0")
        return False


if __name__ == "__main__":
    from algorithmeai import Snake

    # Always train fresh — the _verify_layer check runs during training
    # and exits on first wrong-class lookalike with /tmp/snake_fatal.json
    if "--load" in sys.argv and os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = Snake(MODEL_PATH)
        with open(DATA_PATH) as f:
            data = json.load(f)
    else:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        model, data = train_and_save()

    ok = assess(model, data)
    sys.exit(0 if ok else 1)
