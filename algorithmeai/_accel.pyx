# cython: language_level=3
"""
Cython-accelerated hot paths for Snake classifier.

Build:  python setup.py build_ext --inplace
Install: pip install -e ".[fast]" && python setup.py build_ext --inplace

These are standalone functions (not methods) that mirror the pure-Python
logic in snake.py. They are conditionally imported at module level.
"""


def apply_literal_fast(dict X, list literal, list header):
    """Cython version of Snake.apply_literal — returns True/False."""
    cdef int index = literal[0]
    cdef object value = literal[1]
    cdef bint negat = literal[2]
    cdef str datat = literal[3]
    cdef str key = header[index]

    if key not in X:
        return False

    cdef object field = X[key]

    if datat == "TWS":
        if negat:
            return value <= len((<str>field).split(" "))
        return value > len((<str>field).split(" "))
    elif datat == "TPS":
        if negat:
            return value <= len((<str>field).split(","))
        return value > len((<str>field).split(","))
    elif datat == "TSS":
        if negat:
            return value <= len((<str>field).split("."))
        return value > len((<str>field).split("."))
    elif datat == "TLN":
        if negat:
            return value <= len(set(<str>field))
        return value > len(set(<str>field))
    elif datat == "TN":
        if negat:
            return value <= len(<str>field)
        return value > len(<str>field)
    elif datat == "T":
        if negat:
            return value not in <str>field
        return value in <str>field
    elif datat == "N":
        if negat:
            return value <= field
        return value > field
    return False


def apply_clause_fast(dict X, list clause, list header):
    """Cython version of Snake.apply_clause — OR over literals."""
    cdef list literal
    for literal in clause:
        if apply_literal_fast(X, literal, header):
            return True
    return False


def traverse_chain_fast(list chain, dict X, list header):
    """Cython version of traverse_chain — walk IF/ELIF/ELSE chain."""
    cdef dict entry
    cdef list condition
    cdef list lit
    for entry in chain:
        condition = entry["condition"]
        if condition is None:
            return entry
        cdef bint all_match = True
        for lit in condition:
            if not apply_literal_fast(X, lit, header):
                all_match = False
                break
        if all_match:
            return entry
    if chain:
        return chain[-1]
    return None


def get_lookalikes_fast(list layers, dict X, list header, list targets):
    """Cython version of Snake.get_lookalikes — full inference in C."""
    cdef list all_lookalikes = []
    cdef set seen = set()
    cdef list layer, clause_bool
    cdef dict bucket
    cdef set negated
    cdef int i, global_idx, c_idx
    cdef list condition
    cdef bint all_negated

    for layer in layers:
        bucket = traverse_chain_fast(layer, X, header)
        if bucket is None:
            continue
        clause_bool = [apply_clause_fast(X, c, header) for c in bucket["clauses"]]
        negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
        for l in bucket["lookalikes"]:
            for condition in bucket["lookalikes"][l]:
                all_negated = True
                for c_idx in condition:
                    if c_idx not in negated:
                        all_negated = False
                        break
                if all_negated:
                    global_idx = bucket["members"][int(l)]
                    if global_idx not in seen:
                        seen.add(global_idx)
                        all_lookalikes.append([global_idx, targets[global_idx], condition])
    return all_lookalikes


def batch_predict_fast(list layers, list Xs, list header, list targets, list unique_targets):
    """Cython-accelerated batch prediction. Returns list of (prediction, confidence, prob_dict)."""
    cdef list results = []
    cdef int n_classes = len(unique_targets)
    cdef double uniform = 1.0 / n_classes if n_classes > 0 else 0.0
    cdef dict X
    cdef list lookalikes
    cdef dict prob
    cdef double best_p, p
    cdef object best_tv, tv
    cdef int n_lk

    for X in Xs:
        lookalikes = get_lookalikes_fast(layers, X, header, targets)
        n_lk = len(lookalikes)

        if n_lk == 0:
            prob = {}
            for tv in unique_targets:
                try:
                    prob[tv] = uniform
                except TypeError:
                    import json
                    prob[json.dumps(tv, sort_keys=True)] = uniform
            best_tv = unique_targets[0]
            best_p = uniform
        else:
            prob = {}
            best_tv = unique_targets[0]
            best_p = -1.0
            for tv in unique_targets:
                count = 0
                for triple in lookalikes:
                    if triple[1] == tv:
                        count += 1
                p = <double>count / <double>n_lk
                try:
                    prob[tv] = p
                except TypeError:
                    import json
                    prob[json.dumps(tv, sort_keys=True)] = p
                if p > best_p:
                    best_p = p
                    best_tv = tv

        results.append({
            "prediction": best_tv,
            "probability": prob,
            "confidence": best_p,
        })
    return results
