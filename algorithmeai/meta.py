"""
Meta error classifier — learns WHERE a base Snake model fails.

Generates cross-validated error labels (TP/TN/FP/FN/NS for binary,
R1-R5/W/NS for multiclass), then trains a Snake error classifier on
those labels.  The error model can predict error type for new samples,
enabling targeted corrections (e.g. flip high-confidence FN predictions).
"""

import csv
import json
import os
import random
from collections import Counter
from time import time

from .snake import Snake, floatconversion


class Meta:
    """Cross-validated error-type classifier built on top of Snake."""

    def __init__(self, Knowledge, target_index=0, excluded_features_index=(),
                 n_layers=5, bucket=250, noise=0.25, workers=1,
                 n_splits=25, n_runs=2, split_ratio=0.8,
                 error_layers=7, error_bucket=50,
                 vocal=False):
        self.n_layers = n_layers
        self.bucket = bucket
        self.noise = noise
        self.workers = workers
        self.n_splits = n_splits
        self.n_runs = n_runs
        self.split_ratio = split_ratio
        self.error_layers = error_layers
        self.error_bucket = error_bucket
        self.vocal = vocal

        self.population = []
        self.target = None
        self.labels = []
        self.label_counts = Counter()
        self.is_binary = False
        self.positive_class = None
        self.agreement_rate = 0.0
        self.error_model = None

        # JSON load path
        if isinstance(Knowledge, str) and Knowledge.endswith(".json"):
            self._from_json(Knowledge)
            return

        # Probe Snake to normalize any input into population + target
        self.population, self.target, self._type_map = self._normalize_knowledge(
            Knowledge, target_index, excluded_features_index
        )

        # Detect binary vs multiclass
        unique_targets = set(row[self.target] for row in self.population)
        self.is_binary = len(unique_targets) == 2

        if self.is_binary:
            self.positive_class = self._determine_positive_class()

        # Generate labels
        total_splits = self.n_runs * self.n_splits
        if self.vocal:
            mode = "binary" if self.is_binary else "multiclass"
            print(f"Meta: {mode} mode, {len(self.population)} samples")
            print(f"Meta: generating error labels ({self.n_runs} run{'s' if self.n_runs > 1 else ''} x {self.n_splits} splits = {total_splits} models to train)")

        self._t0 = time()
        self._splits_done = 0
        self._total_splits = total_splits
        self.labels, self.label_counts, self.agreement_rate = self._generate_labels()

        if self.vocal:
            elapsed = time() - self._t0
            print(f"Meta: labeling done in {elapsed:.1f}s")
            print(f"Meta: agreement rate {self.agreement_rate:.1%}")
            print(f"Meta: label distribution: {dict(self.label_counts)}")
            print(f"Meta: training error model ({self.error_layers} layers, bucket={self.error_bucket})...")

        # Train error model
        t_err = time()
        self.error_model = self._train_error_model()

        if self.vocal:
            total = time() - self._t0
            print(f"Meta: error model trained in {time() - t_err:.1f}s")
            print(f"Meta: total {total:.1f}s")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_prediction(self, X):
        """Predict error type for a feature dict X."""
        return self.error_model.get_prediction(self._cast_features(X))

    def get_probability(self, X):
        """Error type probabilities for a feature dict X."""
        return self.error_model.get_probability(self._cast_features(X))

    def to_list(self):
        """Population with error_type column added."""
        result = []
        for row, label in zip(self.population, self.labels):
            extended = dict(row)
            extended["error_type"] = label
            result.append(extended)
        return result

    def to_csv(self, path):
        """Write augmented population to CSV."""
        rows = self.to_list()
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def to_json(self, path="meta.json"):
        """Save Meta metadata + error model to JSON files."""
        # Save error model as sibling file
        base, ext = os.path.splitext(path)
        error_model_path = base + "_error_model" + ext
        self.error_model.to_json(error_model_path)

        meta_data = {
            "version": "4.4.3",
            "meta_version": 1,
            "target": self.target,
            "is_binary": self.is_binary,
            "positive_class": self.positive_class,
            "labels": self.labels,
            "label_counts": dict(self.label_counts),
            "agreement_rate": self.agreement_rate,
            "population": self.population,
            "config": {
                "n_layers": self.n_layers,
                "bucket": self.bucket,
                "noise": self.noise,
                "workers": self.workers,
                "n_splits": self.n_splits,
                "n_runs": self.n_runs,
                "split_ratio": self.split_ratio,
                "error_layers": self.error_layers,
                "error_bucket": self.error_bucket,
            },
            "error_model": os.path.basename(error_model_path),
        }
        with open(path, "w") as f:
            json.dump(meta_data, f, indent=2)

    def summary(self):
        """Human-readable label distribution."""
        mode = "binary" if self.is_binary else "multiclass"
        lines = [f"Meta error classifier ({mode}, {len(self.population)} samples)"]
        lines.append(f"Target: {self.target}")
        lines.append(f"Agreement rate: {self.agreement_rate:.1%}")
        lines.append("Label distribution:")
        total = len(self.population)
        for label, count in self.label_counts.most_common():
            pct = 100 * count / total if total else 0
            bar = "#" * int(pct / 2)
            lines.append(f"  {label:>3}: {count:>4} ({pct:5.1f}%) {bar}")
        return "\n".join(lines)

    def __repr__(self):
        mode = "binary" if self.is_binary else "multiclass"
        return f"Meta({mode}, {len(self.population)} samples, {len(self.label_counts)} labels)"

    def _cast_features(self, X):
        """Cast feature values to match the error model's expected types."""
        em = self.error_model
        result = {}
        for col, dt in zip(em.header, em.datatypes):
            if col == em.target:
                continue
            if col not in X:
                continue
            val = X[col]
            if dt == "N":
                result[col] = floatconversion(str(val))
            elif dt in ("I", "B"):
                result[col] = int(floatconversion(str(val)))
            else:
                result[col] = val
        return result

    # ------------------------------------------------------------------
    # Internal: knowledge normalization
    # ------------------------------------------------------------------

    def _normalize_knowledge(self, Knowledge, target_index, excluded_features_index):
        """Train a 1-layer probe Snake to normalize any input format."""
        probe = Snake(Knowledge, target_index=target_index,
                      excluded_features_index=excluded_features_index,
                      n_layers=1, bucket=max(len(Knowledge) if not isinstance(Knowledge, str) else 9999, 250),
                      noise=0, vocal=False, workers=1)
        population = list(probe.population)
        target = probe.target
        type_map = self._build_type_map(probe)
        return population, target, type_map

    def _build_type_map(self, model):
        """Extract {feature: datatype} from a Snake model."""
        tm = {}
        for col, dt in zip(model.header, model.datatypes):
            if col != model.target:
                tm[col] = dt
        return tm

    def _convert_features(self, row, type_map):
        """Strip target key and cast types for prediction."""
        features = {}
        for k, v in row.items():
            if k == self.target:
                continue
            if type_map.get(k) == "N":
                features[k] = floatconversion(str(v))
            else:
                features[k] = v
        return features

    # ------------------------------------------------------------------
    # Internal: labeling
    # ------------------------------------------------------------------

    def _label_binary(self, pred, actual):
        """Classify a single binary prediction as TP/TN/FP/FN."""
        if pred == self.positive_class and actual == self.positive_class:
            return "TP"
        elif pred != self.positive_class and actual != self.positive_class:
            return "TN"
        elif pred == self.positive_class and actual != self.positive_class:
            return "FP"
        else:
            return "FN"

    def _label_multiclass(self, prob_dict, actual):
        """Rank classes by probability, return R1-R5 or W."""
        ranked = sorted(prob_dict, key=prob_dict.get, reverse=True)
        for i, cls in enumerate(ranked[:5]):
            if cls == actual:
                return f"R{i + 1}"
        return "W"

    def _label_one_run(self):
        """One labeling run: n_splits random splits -> majority label per sample."""
        n = len(self.population)
        labels_per_sample = [[] for _ in range(n)]
        indexed = list(enumerate(self.population))

        for s in range(self.n_splits):
            random.shuffle(indexed)
            split = int(n * self.split_ratio)
            train_part = indexed[:split]
            test_part = indexed[split:]

            train_rows = [row for _, row in train_part]
            model = Snake(train_rows, target_index=self.target,
                          n_layers=self.n_layers, bucket=self.bucket,
                          noise=self.noise, vocal=False, workers=self.workers)
            type_map = self._build_type_map(model)

            test_features = [self._convert_features(row, type_map)
                             for _, row in test_part]
            batch = model.get_batch_prediction(test_features)

            for (orig_idx, row), res in zip(test_part, batch):
                actual = row[self.target]
                if self.is_binary:
                    pred = res["prediction"]
                    labels_per_sample[orig_idx].append(
                        self._label_binary(pred, actual)
                    )
                else:
                    labels_per_sample[orig_idx].append(
                        self._label_multiclass(res["probability"], actual)
                    )

            self._splits_done += 1
            if self.vocal:
                elapsed = time() - self._t0
                avg = elapsed / self._splits_done
                remaining = avg * (self._total_splits - self._splits_done)
                mins, secs = divmod(int(remaining), 60)
                eta = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
                print(f"    split {s + 1}/{self.n_splits} ({self._splits_done}/{self._total_splits} total, ETA {eta})")

        # Majority vote per sample
        run_labels = []
        for sample_labels in labels_per_sample:
            if not sample_labels:
                run_labels.append("NS")
                continue
            counter = Counter(sample_labels)
            best, _ = counter.most_common(1)[0]
            run_labels.append(best)
        return run_labels

    def _generate_labels(self):
        """n_runs independent runs -> agreement filter -> final labels."""
        runs = []
        for r in range(self.n_runs):
            if self.vocal:
                print(f"  Run {r + 1}/{self.n_runs} ({self.n_splits} splits):")
            runs.append(self._label_one_run())

        if self.n_runs == 1:
            labels = runs[0]
        else:
            labels = []
            for i in range(len(self.population)):
                votes = [run[i] for run in runs]
                if len(set(votes)) == 1:
                    labels.append(votes[0])
                else:
                    labels.append("NS")

        agree = sum(1 for lbl in labels if lbl != "NS")
        agreement_rate = agree / len(labels) if labels else 0.0
        label_counts = Counter(labels)
        return labels, label_counts, agreement_rate

    def _determine_positive_class(self):
        """Minority class for binary; tiebreaker = second sorted value."""
        counts = Counter(row[self.target] for row in self.population)
        sorted_classes = sorted(counts.keys(), key=lambda c: (counts[c], c))
        return sorted_classes[0]

    # ------------------------------------------------------------------
    # Internal: error model training
    # ------------------------------------------------------------------

    def _train_error_model(self):
        """Train Snake on population + error_type labels."""
        training_data = self.to_list()
        model = Snake(training_data, target_index="error_type",
                      n_layers=self.error_layers, bucket=self.error_bucket,
                      noise=self.noise, vocal=False, workers=self.workers)
        return model

    # ------------------------------------------------------------------
    # Internal: JSON load
    # ------------------------------------------------------------------

    def _from_json(self, path):
        """Load a saved Meta from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        if data.get("meta_version") is None:
            raise ValueError(f"{path} is not a valid Meta JSON file")

        self.target = data["target"]
        self.is_binary = data["is_binary"]
        self.positive_class = data["positive_class"]
        self.labels = data["labels"]
        self.label_counts = Counter(data["label_counts"])
        self.agreement_rate = data["agreement_rate"]
        self.population = data["population"]

        cfg = data.get("config", {})
        self.n_layers = cfg.get("n_layers", self.n_layers)
        self.bucket = cfg.get("bucket", self.bucket)
        self.noise = cfg.get("noise", self.noise)
        self.workers = cfg.get("workers", self.workers)
        self.n_splits = cfg.get("n_splits", self.n_splits)
        self.n_runs = cfg.get("n_runs", self.n_runs)
        self.split_ratio = cfg.get("split_ratio", self.split_ratio)
        self.error_layers = cfg.get("error_layers", self.error_layers)
        self.error_bucket = cfg.get("error_bucket", self.error_bucket)

        # Load error model from sibling file
        error_model_filename = data["error_model"]
        error_model_path = os.path.join(os.path.dirname(path), error_model_filename)
        self.error_model = Snake(error_model_path)

        # Rebuild type map from error model (features minus error_type and target)
        self._type_map = {}
        for col, dt in zip(self.error_model.header, self.error_model.datatypes):
            if col not in (self.target, "error_type"):
                self._type_map[col] = dt
