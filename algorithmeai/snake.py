import json
from random import choice, sample
from time import time

################################################################
#                                                              #
#    Algorithme.ai : Snake         Author : Charles Dana       #
#                                                              #
#    v4.2.0 — SAT-ensembled bucketed multiclass classifier     #
#                                                              #
################################################################

"""
When working with strings of floating point, handles the mistakes by replacing the value to 0.0 when floating parse error
"""
def floatconversion(txt):
    try:
        result = float(txt)
        return result
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Bucket helpers (pure functions, no self)
# ---------------------------------------------------------------------------

def build_condition(matching, population, targets, bucket, oppose_fn, apply_literal_fn, max_retries=50, log_fn=None):
    """AND of oppose literals that peels ~bucket elements from matching."""
    condition = []
    retries = 0
    t_start = time()
    n_literals = 0
    while len(matching) > 2 * bucket and retries < max_retries:
        target_vals = list(set(targets[i] for i in matching))
        if len(target_vals) < 2:
            if log_fn:
                log_fn(f"#     [condition] only 1 target left in {len(matching)} samples, stopping")
            break
        t_a = choice(target_vals)
        t_b = choice([t for t in target_vals if t != t_a])
        A = population[choice([i for i in matching if targets[i] == t_a])]
        B = population[choice([i for i in matching if targets[i] == t_b])]
        literal = oppose_fn(A, B)
        if literal is None:
            retries += 1
            continue
        satisfying = [i for i in matching if apply_literal_fn(population[i], literal)]
        if len(satisfying) < bucket:
            retries += 1
            continue
        n_literals += 1
        before = len(matching)
        condition.append(literal)
        matching = satisfying
        retries = 0
        if log_fn:
            log_fn(f"#     [condition] literal #{n_literals}: {before} -> {len(matching)} samples (type={literal[3]}, retries_left={max_retries})")
    elapsed = time() - t_start
    if log_fn:
        log_fn(f"#     [condition] built {len(condition)} literals, {len(matching)} samples remaining ({elapsed:.3f}s)")
    return condition, matching


def build_bucket_chain(population, targets, bucket, oppose_fn, apply_literal_fn, noise=0.25, log_fn=None):
    """Sequential IF/ELIF/ELSE peeling into buckets."""
    chain = []
    remaining = list(range(len(population)))
    t_chain_start = time()
    branch_idx = 0
    if log_fn:
        log_fn(f"#   [bucket_chain] START building chain: {len(population)} samples, bucket_size={bucket}, noise={noise}")
    while len(remaining) > 2 * bucket:
        t_branch = time()
        if log_fn:
            n_targets_remaining = len(set(targets[i] for i in remaining))
            log_fn(f"#   [bucket_chain] --- BRANCH {branch_idx} --- {len(remaining)} remaining, {n_targets_remaining} unique targets")
        condition, selected = build_condition(
            remaining, population, targets, bucket,
            oppose_fn, apply_literal_fn, log_fn=log_fn
        )
        if not condition:
            if log_fn:
                log_fn(f"#   [bucket_chain] BRANCH {branch_idx}: no condition found, stopping chain")
            break
        core_set = set(selected)
        rest = [i for i in remaining if i not in core_set]
        members = list(selected)
        noise_added = 0
        if noise > 0 and len(rest) > 0:
            noise_count = max(1, int(noise * len(selected)))
            noise_added = min(noise_count, len(rest))
            members += sample(rest, noise_added)
        chain.append({"condition": condition, "members": members})
        elapsed_branch = time() - t_branch
        if log_fn:
            log_fn(f"#   [bucket_chain] BRANCH {branch_idx}: IF({len(condition)} literals) -> {len(selected)} core + {noise_added} noise = {len(members)} members ({elapsed_branch:.3f}s)")
        remaining = rest
        branch_idx += 1
    if remaining:
        chain.append({"condition": None, "members": remaining})
        if log_fn:
            log_fn(f"#   [bucket_chain] ELSE bucket: {len(remaining)} remaining members")
    elapsed_chain = time() - t_chain_start
    if log_fn:
        sizes = [len(e["members"]) for e in chain]
        log_fn(f"#   [bucket_chain] DONE: {len(chain)} buckets, sizes={sizes}, total={elapsed_chain:.3f}s")
    return chain


def traverse_chain(chain, X, apply_literal_fn):
    """Walk the IF/ELIF/ELSE chain, return the first matching bucket."""
    for entry in chain:
        if entry["condition"] is None:
            return entry
        if all(apply_literal_fn(X, lit) for lit in entry["condition"]):
            return entry
    return chain[-1] if chain else None


"""
Snake() of data will provide insights
"""
class Snake():
    def __init__(self, Knowledge, target_index=0, excluded_features_index=(),
                 n_layers=5, bucket=250, noise=0.25, vocal=False, saved=False):
        self.log = """################################################################
#                                                              #
#    Algorithme.ai : Snake         Author : Charles Dana       #
#                                                              #
#    v4.2.0 — SAT-ensembled bucketed multiclass classifier     #
#                                                              #
################################################################
"""
        print(self.log)
        self.population = 0
        self.header = 0
        self.target = 0
        self.targets = 0
        self.datatypes = 0
        self.layers = []
        self.clauses = []
        self.lookalikes = 0
        self.n_layers = n_layers
        self.bucket = bucket
        self.noise = noise
        self.vocal = vocal

        # Detect input type and dispatch
        if isinstance(Knowledge, str) and Knowledge.endswith(".json"):
            self.from_json(Knowledge)
            return

        if isinstance(Knowledge, str) and Knowledge.endswith(".csv"):
            self._init_from_csv(Knowledge, target_index, list(excluded_features_index), saved)
        else:
            self._init_from_data(Knowledge, target_index)

    # ------------------------------------------------------------------
    # CSV flow (original, preserved)
    # ------------------------------------------------------------------
    def _init_from_csv(self, csv_path, target_index, excluded_features_index, saved):
        self.qprint(f"# Initiated Snake with {self.n_layers} layers and vocal mode {self.vocal} from csv {csv_path}")
        with open(csv_path, "r") as f:
            header = self.make_bloc_from_line(f.readlines()[0])
        with open(csv_path, "r") as f:
            rows = f.readlines()[1:]
        target_column = header[target_index]
        self.target = target_column
        train_columns = [header[i] for i in range(len(header)) if not i in (excluded_features_index + [target_index])]
        header_index = [target_index] + [i for i in range(len(header)) if not i in (excluded_features_index + [target_index])]
        self.header = [target_column] + train_columns
        self.qprint(f"# Analysis train columns {train_columns}")
        self.qprint(f"# Analysis header {self.header}")
        self.datatypes = []
        targets = [self.make_bloc_from_line(row)[target_index] for row in rows]
        self._detect_target_type(targets)
        occurences_vector = {targ : sum((trg == targ for trg in self.targets)) for targ in set(self.targets)}
        self.qprint(f"# Algorithme.ai : Occurence Vector {occurences_vector}")
        for t in range(1, len(self.header)):
            hi = header_index[t]
            dtt = "N"
            values = [self.make_bloc_from_line(row)[hi] for row in rows]
            universe = set("".join(values))
            if [c for c in universe if not c in "+-.0123456789e"] == []:
                dtt = "N"
            else:
                dtt = "T"
            if dtt == "N":
                h = header[hi]
                self.qprint(f"#\t[{h}] numeric field")
            if dtt == "T":
                h = header[hi]
                self.qprint(f"#\t[{h}] text field")
            self.datatypes += [dtt]
        self.qprint(f"# Analysis datatypes {self.datatypes}")
        pp = self.make_population(csv_path, drop=True)
        self.target = self.header[0]
        self.targets = [dp[self.target] for dp in pp]
        self.population = pp
        unique = len(set(self.targets))
        self.qprint(f"# Population ready: {len(pp)} samples, {unique} unique targets, {len(self.header)-1} features")
        self._train(saved)

    # ------------------------------------------------------------------
    # Universal data flow (list/dict/DataFrame)
    # ------------------------------------------------------------------
    def _init_from_data(self, Knowledge, target_index):
        header, rows, ti = self._normalize_input(Knowledge, target_index)
        self.header = [header[ti]] + [header[i] for i in range(len(header)) if i != ti]
        self.target = self.header[0]
        self.qprint(f"# Initiated Snake with {self.n_layers} layers from in-memory data ({len(rows)} rows)")
        self.qprint(f"# Analysis header {self.header}")

        targets = [row[ti] for row in rows]
        self.datatypes = []
        self._detect_target_type([str(t) for t in targets])

        # Detect feature types
        header_index = [ti] + [i for i in range(len(header)) if i != ti]
        for t in range(1, len(self.header)):
            hi = header_index[t]
            values = [str(row[hi]) for row in rows]
            universe = set("".join(values))
            if [c for c in universe if not c in "+-.0123456789e"] == []:
                dtt = "N"
            else:
                dtt = "T"
            self.qprint(f"#\t[{self.header[t]}] {'numeric' if dtt == 'N' else 'text'} field")
            self.datatypes += [dtt]
        self.qprint(f"# Analysis datatypes {self.datatypes}")

        # Build population dicts
        pp = []
        hashes = set()
        for row in rows:
            item = {}
            item_hash = ""
            reordered = [row[hi] for hi in header_index]
            for i in range(len(self.header)):
                h = self.header[i]
                dtt = self.datatypes[i]
                val = reordered[i]
                if dtt in "NIB":
                    item[h] = floatconversion(str(val)) if dtt == "N" else int(floatconversion(str(val)))
                else:
                    item[h] = str(val)
                if i > 0:
                    item_hash += str(item[h])
            if item_hash not in hashes:
                hashes.add(item_hash)
                pp.append(item)
            else:
                self.qprint(f"# Algorithme.ai : Dropped conflicting row {item}")

        self.population = pp
        self.target = self.header[0]
        self.targets = [dp[self.target] for dp in pp]
        unique = len(set(self.targets))
        self.qprint(f"# Population ready: {len(pp)} samples, {unique} unique targets, {len(self.header)-1} features")
        self.qprint(f"# Deduplication: {len(rows)} rows -> {len(pp)} unique ({len(rows) - len(pp)} dropped)")
        self._train(False)

    # ------------------------------------------------------------------
    # Normalization: any Knowledge → (header, rows, target_index)
    # ------------------------------------------------------------------
    def _normalize_input(self, Knowledge, target_index):
        # Duck-typed DataFrame
        if hasattr(Knowledge, 'to_dict') and callable(Knowledge.to_dict):
            records = Knowledge.to_dict('records')
            if not records:
                raise ValueError("Empty DataFrame")
            header = list(records[0].keys())
            rows = [list(r.values()) for r in records]
            if isinstance(target_index, str):
                ti = header.index(target_index)
            else:
                ti = target_index
            return header, rows, ti

        if not isinstance(Knowledge, list) or len(Knowledge) == 0:
            raise ValueError("Knowledge must be a non-empty list, CSV path, JSON path, or DataFrame")

        first = Knowledge[0]

        # list[dict]
        if isinstance(first, dict):
            header = list(first.keys())
            rows = [list(d.get(k, "") for k in header) for d in Knowledge]
            if isinstance(target_index, str):
                ti = header.index(target_index)
            else:
                ti = 0
            return header, rows, ti

        # list[tuple|list] — check if uniform or variable length
        if isinstance(first, (tuple, list)):
            max_len = max(len(r) for r in Knowledge)
            # Pad variable-length rows with defaults
            rows = []
            for r in Knowledge:
                padded = list(r) + [""] * (max_len - len(r))
                rows.append(padded)
            header = ["target"] + [f"f{i}" for i in range(1, max_len)]
            ti = 0
            return header, rows, ti

        # list[str|int|float] — self-classing
        header = ["target", "f1"]
        rows = [[v, v] for v in Knowledge]
        ti = 0
        return header, rows, ti

    # ------------------------------------------------------------------
    # Target type detection (shared by both flows)
    # ------------------------------------------------------------------
    def _detect_target_type(self, targets):
        """Detect target column type from string values. Appends to self.datatypes and sets self.targets."""
        universe = set("".join(targets))
        if sorted(list(set(targets))) == ["0", "1"]:
            self.datatypes = ["B"]
            self.targets = [int(trg) for trg in targets]
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a binary problem 0/1")
        elif sorted(list(set(targets))) in [["False", "True"], ["FALSE", "TRUE"]]:
            self.datatypes = ["B"]
            self.targets = [int("T" in trg or "t" in trg) for trg in targets]
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a binary problem True/False")
        elif [c for c in universe if not c in "0123456789"] == []:
            self.datatypes = ["I"]
            self.targets = [int("0" + trg) for trg in targets]
            unique_targets = sorted(list(set(targets)))
            label = "/".join(unique_targets)
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass integers problem {label}")
        elif [c for c in universe if not c in "+-.0123456789e"] == []:
            self.datatypes = ["N"]
            unique_targets = sorted(list(set(targets)))
            label = "/".join(unique_targets)
            self.targets = [floatconversion(trg) for trg in targets]
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass floating point problem {label}")
        else:
            unique_targets = sorted(list(set(targets)))
            label = "/".join(unique_targets)
            self.targets = list(targets)
            self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass text field problem {label}")
            self.datatypes = ["T"]

    # ------------------------------------------------------------------
    # Training (shared)
    # ------------------------------------------------------------------
    def _train(self, saved):
        self.layers = []
        self.clauses = []
        self.lookalikes = {str(l): [] for l in range(len(self.population))}

        unique_targets = sorted(list(set(self.targets)))
        target_counts = {t: self.targets.count(t) for t in unique_targets}

        self.qprint(f"#")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   TRAINING START")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   Population:    {len(self.population)} samples")
        self.qprint(f"#   Features:      {len(self.header) - 1} ({sum(1 for d in self.datatypes[1:] if d == 'T')} text, {sum(1 for d in self.datatypes[1:] if d == 'N')} numeric)")
        self.qprint(f"#   Target:        {self.target} ({self.datatypes[0]} type)")
        self.qprint(f"#   Classes:       {len(unique_targets)} unique values")
        self.qprint(f"#   Layers:        {self.n_layers}")
        self.qprint(f"#   Bucket size:   {self.bucket}")
        self.qprint(f"#   Noise:         {self.noise}")
        self.qprint(f"#   Vocal:         {self.vocal}")
        self.qprint(f"#")
        top_5 = sorted(target_counts.items(), key=lambda x: -x[1])[:5]
        self.qprint(f"#   Top classes:   {', '.join(f'{t}({c})' for t, c in top_5)}")
        min_class = min(target_counts.values())
        max_class = max(target_counts.values())
        self.qprint(f"#   Class range:   min={min_class}, max={max_class}, avg={len(self.population)/len(unique_targets):.1f}")
        self.qprint(f"# ============================================================")
        self.qprint(f"#")

        t_0 = time()
        for i in range(self.n_layers):
            t_layer_start = time()
            self.qprint(f"#")
            self.qprint(f"# >>> LAYER {i+1}/{self.n_layers} — starting construction...")
            self.construct_layer()
            t_layer_end = time()
            layer_time = t_layer_end - t_layer_start
            elapsed_total = t_layer_end - t_0
            layers_done = i + 1
            layers_left = self.n_layers - layers_done
            avg_per_layer = elapsed_total / layers_done
            eta = avg_per_layer * layers_left

            # Count total clauses and buckets in this layer
            layer = self.layers[-1]
            n_buckets = len(layer)
            n_clauses = sum(len(entry["clauses"]) for entry in layer)

            self.qprint(f"# <<< LAYER {i+1}/{self.n_layers} DONE in {layer_time:.2f}s — {n_buckets} buckets, {n_clauses} clauses")
            self.qprint(f"#     elapsed={elapsed_total:.2f}s, avg/layer={avg_per_layer:.2f}s, ETA={eta:.2f}s ({layers_left} layers left)")

        total_time = time() - t_0
        total_clauses = sum(len(entry["clauses"]) for layer in self.layers for entry in layer)
        total_buckets = sum(len(layer) for layer in self.layers)
        self.qprint(f"#")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   TRAINING COMPLETE")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   Total time:    {total_time:.2f}s")
        self.qprint(f"#   Total layers:  {self.n_layers}")
        self.qprint(f"#   Total buckets: {total_buckets}")
        self.qprint(f"#   Total clauses: {total_clauses}")
        self.qprint(f"#   Avg clauses/layer: {total_clauses/self.n_layers:.1f}")
        self.qprint(f"# ============================================================")

        if saved:
            self.to_json()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def qprint(self, txt, level=1):
        v = 1 if self.vocal is True else (self.vocal if self.vocal else 0)
        if v >= level:
            print(txt)
        self.log += str(txt) + "\n"

    """
    Will parse a .csv line properly,
    returning an array of string, handling triple quotes elegantly
    """
    def make_bloc_from_line(self, line):
        line = line.replace('\n', '')
        if '"' in line:
            quoted = False
            bloc = []
            txt = ''
            for c in line:
                if c == '"':
                    quoted = not quoted
                else:
                    if c == ',' and not quoted:
                        bloc += [txt]
                        txt = ''
                    else:
                        txt += c
            bloc += [txt]
            return bloc
        return line.split(',')

    """
    Will effectively parse any .csv properly formated by pandas
    """
    def read_csv(self, fname):
        if not '.csv' in fname:
            self.qprint("Algorithme.ai: Please input a .csv file")
            return 0, 0
        with open(fname, "r") as f:
            lines = f.readlines()
        header = self.make_bloc_from_line(lines[0])
        data = [self.make_bloc_from_line(lines[t]) for t in range(1, len(lines))]
        return header, data

    """
    Makes the population available to the user
    """
    def make_population(self, fname, drop=False):
        POPULATION = []
        data_header, data = self.read_csv(fname)
        mapping_table = {h : -1 for h in self.header}
        for h in mapping_table:
            if h in data_header:
                mapping_table[h] = min((t for t in range(len(data_header)) if data_header[t] == h))
        hashes = set()
        for row in data:
            item_hash = ""
            item = {}
            for i in range(len(self.header)):
                h = self.header[i]
                dtt = self.datatypes[i]
                if mapping_table[h] == -1:
                    if dtt in "NIB":
                        item[h] = 0
                    if dtt == "T":
                        item[h] = ""
                else:
                    if dtt in "IB":
                        item[h] = int(row[mapping_table[h]])
                    if dtt in "N":
                        item[h] = floatconversion(row[mapping_table[h]])
                    if dtt == "T":
                        item[h] = str(row[mapping_table[h]])
                if i > 0:
                    item_hash += str(item[h])
            if drop and item_hash in hashes:
                self.qprint(f"# Algorithme.ai : Dropped conflicting row {item}")
            if drop and not item_hash in hashes:
                hashes.add(item_hash)
                POPULATION += [item]
            if not drop:
                POPULATION += [item]
        return POPULATION

    # ------------------------------------------------------------------
    # Core SAT methods (unchanged)
    # ------------------------------------------------------------------

    """
    Will return
    - For text fields: words to be or not to be included
    - For numeric fields: splits to be greater or not to be greater
    """
    def oppose(self, T, F):
        candidates = [i for i in range(1, len(self.header)) if T[self.header[i]] != F[self.header[i]]]
        if not candidates:
            return None
        index = choice(candidates)
        h = self.header[index]
        if self.datatypes[index] == "T":
            if choice(["Do it", "Don't"]) == "Do it":
                possible = []
                length = (len(F[h]) != len(T[h]))
                if length:
                    possible += ["TN"]
                alphabet = (len(list(set(F[h]))) != len(list(set(T[h]))))
                if alphabet:
                    possible += ["TLN"]
                alphabet_false = len([c for c in list(set(F[h])) if not c in T[h]]) > 0
                if alphabet_false:
                    possible += ["FA"]
                alphabet_true = len([c for c in list(set(T[h])) if not c in F[h]]) > 0
                if alphabet_true:
                    possible += ["TA"]
                word_splits = (len(F[h].split(" ")) != len(T[h].split(" ")))
                if word_splits:
                    possible += ["TWS"]
                part_splits = (len(F[h].split(",")) != len(T[h].split(",")))
                if part_splits:
                    possible += ["TPS"]
                sent_splits = (len(F[h].split(".")) != len(T[h].split(".")))
                if sent_splits:
                    possible += ["TSS"]
                if len(possible):
                    todo = choice(possible)
                    if todo == "TN":
                        return [index, (len(F[h]) + len(T[h])) / 2, len(T[h]) > len(F[h]), "TN"]
                    if todo == "TLN":
                        return [index, (len(list(set(F[h]))) + len(list(set(T[h])))) / 2, len(list(set(T[h]))) > len(list(set(F[h]))), "TLN"]
                    if todo == "FA":
                        return [index, choice([c for c in list(set(F[h])) if not c in T[h]]), True, "T"]
                    if todo == "TA":
                        return [index, choice([c for c in list(set(T[h])) if not c in F[h]]), False, "T"]
                    if todo == "TWS":
                        return [index, (len(F[h].split(" ")) + len(T[h].split(" "))) / 2, len(T[h].split(" ")) > len(F[h].split(" ")), "TWS"]
                    if todo == "TPS":
                        return [index, (len(F[h].split(",")) + len(T[h].split(","))) / 2, len(T[h].split(",")) > len(F[h].split(",")), "TWS"]
                    if todo == "TSS":
                        return [index, (len(F[h].split(".")) + len(T[h].split("."))) / 2, len(T[h].split(".")) > len(F[h].split(".")), "TWS"]
            pros = set()
            cons = set()
            for sep in [" ", "/", ":", "-"]:
                for label in T[h].split(sep):
                    pros.add(label.split("\'")[0].split('\"')[0])
                for label in F[h].split(sep):
                    cons.add(label.split("\'")[0].split('\"')[0])
            clean_pros = [label for label in pros if len(label) and len(label) < max(2,len(T[h])) and not label in F[h]]
            clean_cons = [label for label in cons if len(label) and len(label) < max(2,len(F[h])) and not label in T[h]]
            possibilities = [[index, label, False, "T"] for label in clean_pros] + [[index, label, True, "T"] for label in clean_cons]
            if len(possibilities):
                return choice(possibilities)
            else:
                if T[h] != F[h] and not T[h] in F[h]:
                    return [index, T[h], False, "T"]
                if T[h] != F[h] and not F[h] in T[h]:
                    return [index, F[h], True, "T"]
        if self.datatypes[index] == "N":
            return [index, (F[h] + T[h]) / 2, T[h] > F[h], "N"]

    """
    Will return:
    - True if the datapoint satisfies the literal
    - False if the datapoint misses the header value or do not satisfy the literal
    Robust.
    """
    def apply_literal(self, X, literal):
        index = literal[0]
        value = literal[1]
        negat = literal[2]
        datat = literal[3]
        if not self.header[index] in X:
            return False
        if datat == "TWS":
            if negat == True:
                return value <= len(X[self.header[index]].split(" "))
            if negat == False:
                return value > len(X[self.header[index]].split(" "))
        if datat == "TPS":
            if negat == True:
                return value <= len(X[self.header[index]].split(","))
            if negat == False:
                return value > len(X[self.header[index]].split(","))
        if datat == "TSS":
            if negat == True:
                return value <= len(X[self.header[index]].split("."))
            if negat == False:
                return value > len(X[self.header[index]].split("."))
        if datat == "TLN":
            if negat == True:
                return value <= len(list(set(X[self.header[index]])))
            if negat == False:
                return value > len(list(set(X[self.header[index]])))
        if datat == "TN":
            if negat == True:
                return value <= len(X[self.header[index]])
            if negat == False:
                return value > len(X[self.header[index]])
        if datat == "T":
            if negat == True:
                return not value in X[self.header[index]]
            if negat == False:
                return value in X[self.header[index]]
        if datat == "N":
            if negat == True:
                return value <= X[self.header[index]]
            if negat == False:
                return value > X[self.header[index]]

    """
    Applies an or Statement on the literals
    """
    def apply_clause(self, X, clause):
        for literal in clause:
            if self.apply_literal(X, literal) == True:
                return True
        return False

    """
    Constructs a minimal clause to discriminate F relative to Ts
    - True on all Ts
    - False on at least F
    - Minimal
    """
    def construct_clause(self, F, Ts):
        clause = [self.oppose(choice(Ts), F)]
        Ts_remainder = [T for T in Ts if self.apply_literal(T, clause[-1]) == False]
        while len(Ts_remainder):
            clause += [self.oppose(choice(Ts_remainder), F)]
            Ts_remainder = [T for T in Ts_remainder if self.apply_literal(T, clause[-1]) == False]
        i = 0
        while i < len(clause):
            sub_clause = [clause[j] for j in range(len(clause)) if i != j]
            minimal_test = False
            for T in Ts:
                if self.apply_clause(T, sub_clause) == False:
                    minimal_test = True
                    break
            if minimal_test:
                i += 1
            if minimal_test == False:
                clause = sub_clause
        return clause

    """
    Constructs a minimal SAT Instance for a target value
    """
    def construct_sat(self, target_value):
        Fs = [self.population[i] for i in range(len(self.population)) if self.targets[i] == target_value]
        Ts = [self.population[i] for i in range(len(self.population)) if self.targets[i] != target_value]
        sat = []
        while len(Fs):
            F = choice(Fs)
            clause = self.construct_clause(F, Ts)
            consequence = [i for i in range(len(self.population)) if self.targets[i] == target_value and self.apply_clause(self.population[i], clause) == False]
            Fs = [F for F in Fs if self.apply_clause(F, clause) == True]
            sat += [[clause, consequence]]
        return sat

    # ------------------------------------------------------------------
    # Bucketed layer construction
    # ------------------------------------------------------------------

    def _construct_local_sat(self, member_indices):
        """Run construct_sat scoped to a bucket's member indices. Returns (clauses, lookalikes) with 0-based local indexing."""
        local_pop = [self.population[i] for i in member_indices]
        local_targets = [self.targets[i] for i in member_indices]
        target_values = sorted(list(set(local_targets)))
        local_clauses = []
        local_lookalikes = {str(l): [] for l in range(len(local_pop))}

        n_local = len(local_pop)
        m = len(self.header) - 1
        self.qprint(f"#     [SAT] local SAT: {n_local} samples, {len(target_values)} targets, O(m*n^2)={m * n_local * n_local:,}")

        t_sat_all = time()
        for tv_idx, target_value in enumerate(target_values):
            t_target = time()
            Fs = [local_pop[i] for i in range(len(local_pop)) if local_targets[i] == target_value]
            Ts = [local_pop[i] for i in range(len(local_pop)) if local_targets[i] != target_value]
            if not Ts:
                self.qprint(f"#     [SAT] target {tv_idx+1}/{len(target_values)} [{target_value}]: skipped (no negatives)", level=2)
                continue
            n_fs_start = len(Fs)
            sat = []
            while len(Fs):
                F = choice(Fs)
                clause = self.construct_clause(F, Ts)
                consequence = [i for i in range(len(local_pop)) if local_targets[i] == target_value and self.apply_clause(local_pop[i], clause) == False]
                Fs = [F for F in Fs if self.apply_clause(F, clause) == True]
                sat += [[clause, consequence]]

            lookalikes_for_target = {str(l): [] for l in range(len(local_pop)) if local_targets[l] == target_value}
            for pair in sat:
                local_clauses.append(pair[0])
                for l in pair[1]:
                    lookalikes_for_target[str(l)].append(len(local_clauses) - 1)
            for l in lookalikes_for_target:
                local_lookalikes[str(l)].append(lookalikes_for_target[str(l)])

            target_time = time() - t_target
            targets_done = tv_idx + 1
            targets_left = len(target_values) - targets_done
            elapsed_sat = time() - t_sat_all
            if targets_done > 0:
                avg_per_target = elapsed_sat / targets_done
                eta_targets = avg_per_target * targets_left
            else:
                eta_targets = 0
            self.qprint(f"#     [SAT] target {targets_done}/{len(target_values)} [{target_value}]: {n_fs_start} positives -> {len(sat)} clauses in {target_time:.2f}s — ETA {eta_targets:.2f}s", level=2)

        total_sat_time = time() - t_sat_all
        self.qprint(f"#     [SAT] local SAT complete: {len(local_clauses)} total clauses in {total_sat_time:.2f}s")

        return local_clauses, local_lookalikes

    """
    Constructs a logical layer of lookalikes (bucketed)
    """
    def construct_layer(self):
        t_layer = time()
        n = len(self.population)
        m = len(self.header) - 1
        k = len(set(self.targets))
        self.qprint(f"#   [layer] Building bucket chain... O(n={n}, m={m}, k={k})")

        chain = build_bucket_chain(
            self.population, self.targets, self.bucket,
            self.oppose, self.apply_literal, self.noise,
            log_fn=self.qprint
        )

        self.qprint(f"#   [layer] Bucket chain ready: {len(chain)} buckets. Now constructing SAT per bucket...")
        t_sat_start = time()
        for b_idx, entry in enumerate(chain):
            t_bucket = time()
            n_b = len(entry["members"])
            k_b = len(set(self.targets[i] for i in entry["members"]))
            # O(m * n_b^2) per bucket SAT construction
            complexity = m * n_b * n_b
            cond_type = f"IF({len(entry['condition'])} lit)" if entry["condition"] else "ELSE"
            self.qprint(f"#   [layer] BUCKET {b_idx}/{len(chain)} ({cond_type}): {n_b} members, {k_b} classes, complexity O(m*n^2)={complexity:,}")

            entry["clauses"], entry["lookalikes"] = self._construct_local_sat(entry["members"])

            bucket_time = time() - t_bucket
            buckets_done = b_idx + 1
            buckets_left = len(chain) - buckets_done
            if buckets_done > 0:
                avg_bucket = (time() - t_sat_start) / buckets_done
                eta_buckets = avg_bucket * buckets_left
            else:
                eta_buckets = 0
            self.qprint(f"#   [layer] BUCKET {b_idx} DONE: {len(entry['clauses'])} clauses in {bucket_time:.2f}s — ETA remaining buckets: {eta_buckets:.2f}s")

        layer_time = time() - t_layer
        self.qprint(f"#   [layer] Layer construction total: {layer_time:.2f}s")
        self.layers.append(chain)

    # ------------------------------------------------------------------
    # Prediction pipeline (bucketed)
    # ------------------------------------------------------------------

    """
    Predict the probability vector for a given X
    """
    def get_lookalikes(self, X):
        all_lookalikes = []
        seen = set()
        for layer in self.layers:
            bucket = traverse_chain(layer, X, self.apply_literal)
            if bucket is None:
                continue
            clause_bool = [self.apply_clause(X, c) for c in bucket["clauses"]]
            negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
            for l in bucket["lookalikes"]:
                for condition in bucket["lookalikes"][l]:
                    if all(c_idx in negated for c_idx in condition):
                        global_idx = bucket["members"][int(l)]
                        if global_idx not in seen:
                            seen.add(global_idx)
                            all_lookalikes.append([global_idx, self.targets[global_idx], condition])
        return all_lookalikes

    """
    Gives the probability vector associated
    """
    def get_probability(self, X):
        target_values = sorted(list(set(self.targets)))
        lookalikes = self.get_lookalikes(X)
        if len(lookalikes) == 0:
            probability = {target_value : 1/len(target_values) for target_value in target_values}
        else:
            probability = {target_value : sum((triple[1] == target_value for triple in lookalikes)) / len(lookalikes) for target_value in target_values}
        return probability

    """
    Predicts the outcome for a datapoint
    """
    def get_prediction(self, X):
        probability = self.get_probability(X)
        pr_max = max((probability[target] for target in probability))
        prediction = [target for target in probability if probability[target] == pr_max][0]
        return prediction

    """
    Augments a datapoint with every available information
    """
    def get_augmented(self, X):
        Y = X.copy()
        Y["Lookalikes"] = self.get_lookalikes(X)
        Y["Probability"] = self.get_probability(X)
        Y["Prediction"] = self.get_prediction(X)
        Y["Audit"] = self.get_audit(X)
        return Y

    # ------------------------------------------------------------------
    # Enhanced Audit
    # ------------------------------------------------------------------

    def _format_literal_text(self, literal):
        """Human-readable description of a single literal."""
        index, value, negat, datat = literal
        h = self.header[index]
        if datat == "T":
            if negat:
                return f'"{h}" does NOT contain "{value}"'
            return f'"{h}" contains "{value}"'
        if datat == "TN":
            if negat:
                return f'len("{h}") >= {value}'
            return f'len("{h}") < {value}'
        if datat == "TLN":
            if negat:
                return f'alphabet("{h}") >= {value}'
            return f'alphabet("{h}") < {value}'
        if datat == "TWS":
            if negat:
                return f'words("{h}") >= {value}'
            return f'words("{h}") < {value}'
        if datat == "TPS":
            if negat:
                return f'parts("{h}") >= {value}'
            return f'parts("{h}") < {value}'
        if datat == "TSS":
            if negat:
                return f'sentences("{h}") >= {value}'
            return f'sentences("{h}") < {value}'
        if datat == "N":
            if negat:
                return f'"{h}" <= {value}'
            return f'"{h}" > {value}'
        return str(literal)

    def _format_bar(self, pct, width=20):
        filled = int(pct / 100 * width)
        return "\u2588" * filled + "\u2591" * (width - filled)

    def get_plain_text_assertion(self, condition, l):
        """Partial audit for a given lookalike (backwards compat)."""
        plain_text_assertion = f"""
        # Datapoint is a lookalike to #{l} of class [{self.targets[int(l)]}]
        - {self.population[int(l)]}

        Because of the following AND statement that applies to both
        """
        # In bucketed mode, condition is local clause indices — we can't resolve global clauses
        # Just note the condition indices
        plain_text_assertion += f"\n\u2022 Matched via local clause condition {condition}"
        return plain_text_assertion

    """
    Audit for a given datapoint — enhanced per-layer ASCII
    R.A.G.
    """
    def get_audit(self, X):
        probability = self.get_probability(X)
        prediction = self.get_prediction(X)
        audit = ""

        for layer_idx, chain in enumerate(self.layers):
            audit += f"\n{'='*50}\n  LAYER {layer_idx}\n{'='*50}\n"
            matched_bucket_idx = None
            for b_idx, entry in enumerate(chain):
                cond = entry["condition"]
                n_members = len(entry["members"])
                if cond is None:
                    label = "ELSE"
                    prefix = "        ELSE:"
                else:
                    parts = [self._format_literal_text(lit) for lit in cond]
                    gate = " AND ".join(parts)
                    if b_idx == 0:
                        prefix = f"  >>>   IF   {gate}:"
                    else:
                        prefix = f"        ELIF {gate}:"

                # Check if X routes here
                if matched_bucket_idx is None:
                    if cond is None:
                        routes_here = True
                    else:
                        routes_here = all(self.apply_literal(X, lit) for lit in cond)
                else:
                    routes_here = False

                if routes_here and matched_bucket_idx is None:
                    matched_bucket_idx = b_idx
                    audit += f"{prefix}\n  >>>     -> BUCKET {b_idx} ({n_members} members)\n"
                else:
                    audit += f"{prefix}\n            -> BUCKET {b_idx} ({n_members} members)\n"

            # Show local lookalikes for matched bucket
            if matched_bucket_idx is not None:
                bucket = chain[matched_bucket_idx]
                clause_bool = [self.apply_clause(X, c) for c in bucket["clauses"]]
                negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
                local_lookalikes = []
                for l in bucket["lookalikes"]:
                    for condition in bucket["lookalikes"][l]:
                        if all(c_idx in negated for c_idx in condition):
                            global_idx = bucket["members"][int(l)]
                            local_lookalikes.append(self.targets[global_idx])

                audit += f"\n  Within BUCKET {matched_bucket_idx}:\n"
                audit += f"    {len(local_lookalikes)} lookalikes found\n"
                if local_lookalikes:
                    counts = {}
                    for t in local_lookalikes:
                        counts[t] = counts.get(t, 0) + 1
                    for t in sorted(counts, key=lambda k: -counts[k]):
                        pct = 100 * counts[t] / len(local_lookalikes)
                        bar = self._format_bar(pct)
                        audit += f"    P({t}){' '*(20-len(str(t)))}= {pct:5.1f}% {bar}\n"

        audit += f"\n{'='*50}\n  GLOBAL SUMMARY ({len(self.layers)} layers)\n{'='*50}\n"
        lookalikes = self.get_lookalikes(X)
        audit += f"  Total lookalikes: {len(lookalikes)}\n"
        for t in sorted(probability, key=lambda k: -probability[k]):
            if probability[t] > 0:
                audit += f"  P({t}) = {100*probability[t]:.1f}%\n"
        audit += f"  >> PREDICTION: {prediction}\n"
        audit += "### END AUDIT ###\n"
        return audit

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, fout="snakeclassifier.json"):
        snake_classifier = {
            "version": "4.2.0",
            "population": self.population,
            "header": self.header,
            "target": self.target,
            "targets": self.targets,
            "datatypes": self.datatypes,
            "config": {
                "n_layers": self.n_layers,
                "bucket": self.bucket,
                "noise": self.noise,
                "vocal": self.vocal,
            },
            "layers": self.layers,
            "log": self.log
        }
        with open(fout, "w") as f:
            f.write(json.dumps(snake_classifier, indent=2))
        self.qprint(f"Safely saved to {fout}")

    def from_json(self, filepath="snakeclassifier.json"):
        with open(filepath, "r") as f:
            loaded_module = json.load(f)
        self.population = loaded_module["population"]
        self.header = loaded_module["header"]
        self.target = loaded_module["target"]
        self.targets = loaded_module["targets"]
        self.datatypes = loaded_module["datatypes"]

        # Backwards compat: flat v0.1 format
        if "clauses" in loaded_module and "layers" not in loaded_module:
            self._load_flat(loaded_module)
        elif "layers" in loaded_module:
            self._load_bucketed(loaded_module)

        if "config" in loaded_module:
            cfg = loaded_module["config"]
            self.n_layers = cfg.get("n_layers", self.n_layers)
            self.bucket = cfg.get("bucket", self.bucket)
            self.noise = cfg.get("noise", self.noise)
            self.vocal = cfg.get("vocal", self.vocal)
        elif "n_layers" in loaded_module:
            self.n_layers = loaded_module["n_layers"]
            self.vocal = loaded_module.get("vocal", self.vocal)

        self.log = loaded_module.get("log", self.log)
        self.qprint(f"# Algorithme.ai : Successful load from {filepath}")

    def _load_flat(self, loaded_module):
        """Load v0.1 flat format (clauses + lookalikes at top level)."""
        self.clauses = loaded_module["clauses"]
        self.lookalikes = loaded_module["lookalikes"]
        # Wrap the flat model into a single ELSE bucket in a single layer
        members = list(range(len(self.population)))
        self.layers = [[{
            "condition": None,
            "members": members,
            "clauses": self.clauses,
            "lookalikes": self.lookalikes
        }]]

    def _load_bucketed(self, loaded_module):
        """Load v4.2.0 bucketed format."""
        self.layers = loaded_module["layers"]
        self.clauses = []
        self.lookalikes = {str(l): [] for l in range(len(self.population))}

    # ------------------------------------------------------------------
    # Validation (adapted for bucketed layers)
    # ------------------------------------------------------------------

    """
    Validation process of the lookalikes table on the premise of a targeted sample
    """
    def make_validation(self, Xs, pruning_coef=0.5):
        new_n_layers = max(1, int(len(self.layers) * pruning_coef))
        self.qprint(f"#")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   VALIDATION START")
        self.qprint(f"# ============================================================")
        self.qprint(f"#   Validation samples: {len(Xs)}")
        self.qprint(f"#   Pruning:            {len(self.layers)} layers -> {new_n_layers} (coef={pruning_coef})")
        self.qprint(f"#   Complexity:         O({len(self.layers)} layers * {len(Xs)} samples)")
        self.qprint(f"# ============================================================")

        # Score each layer by accuracy on validation set
        layer_scores = []
        t_val_start = time()
        for layer_idx, chain in enumerate(self.layers):
            t_layer = time()
            correct = 0
            total = 0
            n_buckets = len(chain)
            n_clauses = sum(len(e["clauses"]) for e in chain)
            for X in Xs:
                if self.target not in X:
                    continue
                target = X[self.target]
                bucket = traverse_chain(chain, X, self.apply_literal)
                if bucket is None:
                    continue
                clause_bool = [self.apply_clause(X, c) for c in bucket["clauses"]]
                negated = {i for i in range(len(clause_bool)) if not clause_bool[i]}
                votes = []
                for l in bucket["lookalikes"]:
                    for condition in bucket["lookalikes"][l]:
                        if all(c_idx in negated for c_idx in condition):
                            global_idx = bucket["members"][int(l)]
                            votes.append(self.targets[global_idx])
                if votes:
                    counts = {}
                    for v in votes:
                        counts[v] = counts.get(v, 0) + 1
                    pred = max(counts, key=counts.get)
                    if pred == target:
                        correct += 1
                total += 1
            accuracy = correct / total if total > 0 else 0
            layer_scores.append((layer_idx, accuracy))

            layer_time = time() - t_layer
            layers_done = layer_idx + 1
            layers_left = len(self.layers) - layers_done
            elapsed_val = time() - t_val_start
            avg_per = elapsed_val / layers_done
            eta_val = avg_per * layers_left
            self.qprint(f"#   [val] Layer {layer_idx}: accuracy={accuracy:.3f} ({correct}/{total}), {n_buckets} buckets, {n_clauses} clauses, {layer_time:.2f}s — ETA {eta_val:.2f}s")

        # Keep top layers
        layer_scores.sort(key=lambda x: -x[1])
        kept_indices = sorted([ls[0] for ls in layer_scores[:new_n_layers]])
        dropped_indices = [ls[0] for ls in layer_scores[new_n_layers:]]

        total_val_time = time() - t_val_start
        self.qprint(f"#")
        self.qprint(f"#   Scoring complete in {total_val_time:.2f}s")
        self.qprint(f"#   Best layers:  {[(idx, f'{acc:.3f}') for idx, acc in layer_scores[:new_n_layers]]}")
        self.qprint(f"#   Dropped:      {dropped_indices}")
        self.qprint(f"#   Kept:         {kept_indices}")

        self.layers = [self.layers[i] for i in kept_indices]
        self.n_layers = len(self.layers)
        self.qprint(f"# ============================================================")
        self.qprint(f"#   VALIDATION COMPLETE — {self.n_layers} layers retained")
        self.qprint(f"# ============================================================")
