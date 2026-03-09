"""
Snake v5.2.0 — Oppose Profile Benchmark
========================================
500 train / 500 test per dataset archetype.
Measures accuracy + training time + inference time per profile.
Designed to run under 500ms per model (lightweight: 3 layers, small buckets).
"""
import random
import string
import time
from algorithmeai import Snake

random.seed(42)

# ---------------------------------------------------------------------------
# Data generators — 1000 rows each, split 500/500
# ---------------------------------------------------------------------------

def _gen_cryptographic(n=1000):
    """Hashes, hex IDs, base64-ish blobs. 3 classes by structure."""
    data = []
    for _ in range(n):
        r = random.random()
        if r < 0.33:
            # hex hash — lowercase hex, fixed length
            val = ''.join(random.choices("0123456789abcdef", k=32))
            label = "hex_hash"
        elif r < 0.66:
            # structured ID — XXXX-9999-X pattern
            val = (
                ''.join(random.choices(string.ascii_uppercase, k=4)) + "-" +
                ''.join(random.choices(string.digits, k=4)) + "-" +
                random.choice(string.ascii_uppercase)
            )
            label = "struct_id"
        else:
            # base64 blob — mixed case + digits + /+=
            val = ''.join(random.choices(
                string.ascii_letters + string.digits + "+/=", k=random.randint(20, 40)
            ))
            label = "base64"
        data.append({"label": label, "token": val})
    return data


def _gen_linguistic(n=1000):
    """Short paragraphs in 3 'styles'. Signal is in word choice, length, punctuation."""
    formal_words = ["pursuant", "hereinafter", "notwithstanding", "aforementioned",
                    "stipulate", "constitute", "whereas", "thereof", "accordingly",
                    "provision", "obligation", "jurisdiction", "compliance", "statute"]
    casual_words = ["hey", "gonna", "wanna", "kinda", "lol", "yeah", "nah", "dude",
                    "awesome", "cool", "stuff", "thing", "totally", "literally", "like"]
    tech_words = ["algorithm", "latency", "throughput", "pipeline", "kubernetes",
                  "microservice", "TCP", "mutex", "deadlock", "garbage", "collector",
                  "polymorphism", "recursion", "bytecode", "serialization"]
    data = []
    for _ in range(n):
        r = random.random()
        if r < 0.33:
            words = random.choices(formal_words, k=random.randint(8, 20))
            text = " ".join(words) + "."
            label = "formal"
        elif r < 0.66:
            words = random.choices(casual_words, k=random.randint(5, 15))
            text = " ".join(words)
            if random.random() > 0.5:
                text += " lol"
            label = "casual"
        else:
            words = random.choices(tech_words, k=random.randint(6, 14))
            text = " ".join(words)
            if random.random() > 0.5:
                text = text.replace(" ", "_", random.randint(1, 3))
            label = "technical"
        data.append({"label": label, "text": text})
    return data


def _gen_mixed(n=1000):
    """Mixed text + numeric. 4 classes. Tests balanced/auto routing."""
    data = []
    categories = ["electronics", "clothing", "food", "furniture"]
    for _ in range(n):
        cat = random.choice(categories)
        if cat == "electronics":
            price = round(random.uniform(50, 2000), 2)
            weight = round(random.uniform(0.1, 15), 1)
            name = random.choice(["Samsung", "Apple", "Sony", "LG"]) + " " + \
                   random.choice(["TV", "Phone", "Tablet", "Laptop"]) + " " + \
                   str(random.randint(2020, 2026))
        elif cat == "clothing":
            price = round(random.uniform(5, 200), 2)
            weight = round(random.uniform(0.05, 2), 1)
            name = random.choice(["Nike", "Adidas", "Zara", "H&M"]) + " " + \
                   random.choice(["Shirt", "Pants", "Jacket", "Shoes"]) + " " + \
                   random.choice(["S", "M", "L", "XL"])
        elif cat == "food":
            price = round(random.uniform(0.5, 30), 2)
            weight = round(random.uniform(0.01, 5), 1)
            name = random.choice(["Organic", "Fresh", "Frozen", "Canned"]) + " " + \
                   random.choice(["Chicken", "Rice", "Pasta", "Vegetables"])
        else:
            price = round(random.uniform(100, 5000), 2)
            weight = round(random.uniform(5, 100), 1)
            name = random.choice(["IKEA", "Ashley", "Wayfair"]) + " " + \
                   random.choice(["Sofa", "Table", "Chair", "Bed"]) + " " + \
                   random.choice(["Oak", "Pine", "Metal", "Glass"])
        data.append({"label": cat, "name": name, "price": price, "weight": weight})
    return data


def _gen_scientific(n=1000):
    """Pure numeric. 3 classes based on measurement clusters."""
    data = []
    for _ in range(n):
        r = random.random()
        if r < 0.33:
            # Cluster A: high temp, low pressure, mid humidity
            temp = round(random.gauss(95, 5), 2)
            pressure = round(random.gauss(980, 15), 2)
            humidity = round(random.gauss(50, 10), 2)
            conductivity = round(random.gauss(0.8, 0.1), 4)
            label = "state_A"
        elif r < 0.66:
            # Cluster B: low temp, high pressure, high humidity
            temp = round(random.gauss(25, 8), 2)
            pressure = round(random.gauss(1050, 20), 2)
            humidity = round(random.gauss(85, 8), 2)
            conductivity = round(random.gauss(0.3, 0.05), 4)
            label = "state_B"
        else:
            # Cluster C: mid temp, mid pressure, low humidity
            temp = round(random.gauss(60, 12), 2)
            pressure = round(random.gauss(1010, 10), 2)
            humidity = round(random.gauss(30, 7), 2)
            conductivity = round(random.gauss(0.55, 0.08), 4)
            label = "state_C"
        data.append({"label": label, "temp": temp, "pressure": pressure,
                     "humidity": humidity, "conductivity": conductivity})
    return data


def _gen_categorical(n=1000):
    """Delimiter-separated tags. 3 classes by tag combinations."""
    action_tags = ["Action", "Adventure", "Sci-Fi", "Thriller", "War"]
    drama_tags = ["Drama", "Romance", "Biography", "History", "Mystery"]
    comedy_tags = ["Comedy", "Animation", "Family", "Musical", "Fantasy"]
    data = []
    for _ in range(n):
        r = random.random()
        if r < 0.33:
            tags = ",".join(random.sample(action_tags, k=random.randint(2, 4)))
            rating = round(random.uniform(5, 9), 1)
            label = "action"
        elif r < 0.66:
            tags = ",".join(random.sample(drama_tags, k=random.randint(2, 4)))
            rating = round(random.uniform(6, 10), 1)
            label = "drama"
        else:
            tags = ",".join(random.sample(comedy_tags, k=random.randint(2, 4)))
            rating = round(random.uniform(4, 8), 1)
            label = "comedy"
        # Add some cross-contamination
        if random.random() < 0.15:
            extra = random.choice(action_tags + drama_tags + comedy_tags)
            tags += "," + extra
        data.append({"label": label, "genres": tags, "rating": rating})
    return data


def _gen_industrial(n=1000):
    """Short product codes, SKUs. Monce-style data."""
    prefixes = {"glass": ["44.2", "33.1", "55.2", "66.2", "10.10"],
                "metal": ["ST-", "AL-", "CU-", "FE-", "TI-"],
                "plastic": ["PP-", "PE-", "PVC-", "ABS-", "PC-"]}
    data = []
    for _ in range(n):
        cat = random.choice(list(prefixes.keys()))
        prefix = random.choice(prefixes[cat])
        if cat == "glass":
            code = prefix + " " + random.choice(["LowE", "Bronze", "Clair", "Opal", "Float"])
            thickness = round(random.uniform(2, 12), 1)
        elif cat == "metal":
            code = prefix + str(random.randint(1000, 9999))
            thickness = round(random.uniform(0.5, 25), 1)
        else:
            code = prefix + str(random.randint(100, 999)) + random.choice(["A", "B", "C", ""])
            thickness = round(random.uniform(0.1, 8), 1)
        data.append({"label": cat, "code": code, "thickness": thickness})
    return data


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

DATASETS = {
    "cryptographic": _gen_cryptographic,
    "linguistic":    _gen_linguistic,
    "scientific":    _gen_scientific,
    "categorical":   _gen_categorical,
    "industrial":    _gen_industrial,
    "mixed":         _gen_mixed,
}

PROFILES = ["auto", "balanced", "linguistic", "industrial",
            "cryptographic", "scientific", "categorical"]

# Lightweight config: fast training, fair comparison
N_LAYERS = 3
BUCKET = 50
NOISE = 0.25


def run_benchmark(dataset_name, gen_fn, profile, seed=42):
    random.seed(seed)
    data = gen_fn(1000)
    random.shuffle(data)
    train = data[:500]
    test = data[500:]

    # Train
    t0 = time.time()
    model = Snake(train, target_index="label", n_layers=N_LAYERS,
                  bucket=BUCKET, noise=NOISE, oppose_profile=profile)
    train_ms = (time.time() - t0) * 1000

    # Inference
    test_features = [{k: v for k, v in row.items() if k != "label"} for row in test]
    test_labels = [row["label"] for row in test]

    t0 = time.time()
    predictions = [model.get_prediction(x) for x in test_features]
    infer_ms = (time.time() - t0) * 1000

    correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
    accuracy = correct / len(test_labels) * 100

    return {
        "dataset": dataset_name,
        "profile": profile,
        "resolved_profile": model.oppose_profile,
        "accuracy": accuracy,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "infer_per_sample_ms": infer_ms / len(test_labels),
    }


def main():
    print("=" * 90)
    print(f"  Snake v5.2.0 — Oppose Profile Benchmark")
    print(f"  Config: {N_LAYERS} layers, bucket={BUCKET}, noise={NOISE}, 500 train / 500 test")
    print("=" * 90)
    print()

    all_results = []

    for ds_name, gen_fn in DATASETS.items():
        print(f"--- {ds_name.upper()} dataset ---")
        print(f"{'Profile':<16} {'Resolved':<16} {'Accuracy':>8} {'Train(ms)':>10} {'Infer(ms)':>10} {'Per-sample':>10}")
        print("-" * 72)

        ds_results = []
        for profile in PROFILES:
            r = run_benchmark(ds_name, gen_fn, profile)
            all_results.append(r)
            ds_results.append(r)
            resolved = r["resolved_profile"] if profile == "auto" else ""
            print(f"{r['profile']:<16} {resolved:<16} {r['accuracy']:>7.1f}% {r['train_ms']:>9.1f} {r['infer_ms']:>9.1f} {r['infer_per_sample_ms']:>9.3f}")

        # Best profile for this dataset
        best = max(ds_results, key=lambda x: x["accuracy"])
        print(f"  >> Best: {best['profile']} ({best['accuracy']:.1f}%)")
        print()

    # Summary table: best profile per dataset
    print("=" * 90)
    print("  SUMMARY — Best profile per dataset")
    print("=" * 90)
    print(f"{'Dataset':<16} {'Best Profile':<16} {'Accuracy':>8} {'Auto Accuracy':>14} {'Auto Picks':>12}")
    print("-" * 68)

    for ds_name in DATASETS:
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        best = max(ds_results, key=lambda x: x["accuracy"])
        auto = next(r for r in ds_results if r["profile"] == "auto")
        print(f"{ds_name:<16} {best['profile']:<16} {best['accuracy']:>7.1f}% {auto['accuracy']:>13.1f}% {auto['resolved_profile']:>12}")

    # Auto routing accuracy
    print()
    print("=" * 90)
    print("  AUTO ROUTING — Did auto pick the best or close?")
    print("=" * 90)
    for ds_name in DATASETS:
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        best = max(ds_results, key=lambda x: x["accuracy"])
        auto = next(r for r in ds_results if r["profile"] == "auto")
        delta = auto["accuracy"] - best["accuracy"]
        match = auto["resolved_profile"]
        status = "OPTIMAL" if abs(delta) < 0.5 else ("CLOSE" if abs(delta) < 3 else "MISS")
        print(f"  {ds_name:<16} auto->{match:<14} {auto['accuracy']:.1f}% vs best {best['accuracy']:.1f}% ({best['profile']})  [{status}] delta={delta:+.1f}%")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
