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
    cdef bint all_match
    for entry in chain:
        condition = entry["condition"]
        if condition is None:
            return entry
        all_match = True
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


# ---------------------------------------------------------------------------
# Training acceleration functions
# ---------------------------------------------------------------------------


def filter_ts_remainder_fast(list Ts, list literal, list header):
    """Filter Ts to keep only those where apply_literal is False (remainder).
    Used in construct_clause to find Ts not yet covered by the last literal."""
    cdef list result = []
    cdef dict T
    for T in Ts:
        if not apply_literal_fast(T, literal, header):
            result.append(T)
    return result


def minimize_clause_fast(list clause, list Ts, list header):
    """Minimize a clause by removing redundant literals.
    A literal is redundant if removing it still leaves the clause True on all Ts."""
    cdef int i = 0
    cdef int j, n
    cdef list sub_clause
    cdef dict T
    cdef bint some_fail
    while i < len(clause):
        n = len(clause)
        sub_clause = [clause[j] for j in range(n) if j != i]
        some_fail = False
        for T in Ts:
            if not apply_clause_fast(T, sub_clause, header):
                some_fail = True
                break
        if some_fail:
            i += 1
        else:
            clause = sub_clause
    return clause


def filter_indices_by_literal_fast(list indices, list population, list literal, list header):
    """Filter population indices where apply_literal is True.
    Used in build_condition to filter matching indices by a literal."""
    cdef list result = []
    cdef int idx
    for idx in indices:
        if apply_literal_fast(population[idx], literal, header):
            result.append(idx)
    return result


def check_clause_covers_all_fast(list Ts, list clause, list header):
    """Check that a clause (OR of literals) is True on all samples in Ts."""
    cdef dict T
    for T in Ts:
        if not apply_clause_fast(T, clause, header):
            return False
    return True


def filter_consequence_fast(list local_pop, list local_targets, object target_value, list clause, list header):
    """Compute consequence indices and remaining Fs for a clause and target value.
    Returns (consequence_indices, remaining_Fs) where:
    - consequence_indices: indices where target matches AND clause is False (NOT eliminated)
    - remaining_Fs: list of Fs where clause is True (eliminated, need further clauses)
    """
    cdef list consequence = []
    cdef list remaining_fs = []
    cdef int i
    cdef int n = len(local_pop)
    for i in range(n):
        if local_targets[i] == target_value:
            if not apply_clause_fast(local_pop[i], clause, header):
                consequence.append(i)
            else:
                remaining_fs.append(local_pop[i])
    return consequence, remaining_fs


def batch_get_lookalikes_fast(list layers, list Xs, list header, list targets):
    """Batch lookalike computation: route all queries per layer, group by bucket.
    Returns list of lookalike-lists, one per query.
    Amortizes chain traversal and improves cache locality for clause evaluation."""
    cdef int n_queries = len(Xs)
    cdef int q_idx, i, global_idx, c_idx
    cdef list result = [[] for _ in range(n_queries)]
    cdef list seen_sets = [set() for _ in range(n_queries)]
    cdef dict X, bucket, grouped
    cdef list layer, clause_bool, condition
    cdef set negated
    cdef bint all_negated

    for layer in layers:
        # Group all queries by their routed bucket (using id for same-object grouping)
        grouped = {}  # id(bucket) -> list of (q_idx, X)
        for q_idx in range(n_queries):
            X = Xs[q_idx]
            bucket = traverse_chain_fast(layer, X, header)
            if bucket is None:
                continue
            bucket_id = id(bucket)
            if bucket_id not in grouped:
                grouped[bucket_id] = (bucket, [])
            grouped[bucket_id][1].append((q_idx, X))

        # Process each bucket group
        for bucket_id in grouped:
            bucket, queries = grouped[bucket_id]
            clauses = bucket["clauses"]
            members = bucket["members"]
            lookalikes_map = bucket["lookalikes"]

            for q_idx, X in queries:
                clause_bool = [apply_clause_fast(X, c, header) for c in clauses]
                negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
                for l in lookalikes_map:
                    for condition in lookalikes_map[l]:
                        all_negated = True
                        for c_idx in condition:
                            if c_idx not in negated:
                                all_negated = False
                                break
                        if all_negated:
                            global_idx = members[int(l)]
                            if global_idx not in seen_sets[q_idx]:
                                seen_sets[q_idx].add(global_idx)
                                result[q_idx].append([global_idx, targets[global_idx], condition])

    return result


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
