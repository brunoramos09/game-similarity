import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

from resolve import ROOT, load_golden

RAW = ROOT / "data" / "raw"
K = 5
POOL = 200
MIN_EXPECTED = 3
SAMPLE_PER_SPLIT = 3000
BOOTSTRAP_SAMPLES = 2000
POPULARITY_WEIGHTS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


def normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def tokens(name):
    return re.findall(r"[a-z0-9]+", name.casefold())


def contains_tokens(longer, shorter):
    size = len(shorter)
    return size > 0 and any(longer[i : i + size] == shorter for i in range(len(longer) - size + 1))


def shares_name(first, second):
    return contains_tokens(first, second) or contains_tokens(second, first)


class Catalog:
    def __init__(self):
        games = pd.read_parquet(RAW / "games.parquet").reset_index(drop=True)
        signals = pd.read_parquet(RAW / "signals.parquet").set_index("game_id")
        self.games = games
        self.position = pd.Series(games.index, index=games["game_id"])
        self.production = normalize(np.array(games["embedding"].tolist(), dtype=np.float32))
        self.franchise = games["franchise"].where(games["franchise"].notna(), None).tolist()
        self.version_group = signals.loc[games["game_id"], "version_group"].to_numpy()
        self.igdb_id = signals.loc[games["game_id"], "igdb_id"].to_numpy()
        activity = signals.loc[games["game_id"], "checkpointd_activity"].to_numpy(dtype=float)
        self.popularity = np.log1p(activity) / np.log1p(activity.max())
        self.name_tokens = [tokens(name) for name in games["name"]]
        self.variants = {"production": self.production}
        without_name = RAW / "embeddings_without_name.parquet"
        if without_name.exists():
            frame = pd.read_parquet(without_name)
            matrix = np.array(frame["embedding"].tolist(), dtype=np.float32)
            order = self.position[frame["game_id"]].to_numpy()
            aligned = np.zeros_like(matrix)
            aligned[order] = matrix
            self.variants["without_name"] = normalize(aligned)

    def candidates(self, matrix, queries):
        results = {}
        for start in range(0, len(queries), 256):
            batch = queries[start : start + 256]
            scores = matrix[batch] @ matrix.T
            top = np.argpartition(-scores, POOL, axis=1)[:, :POOL]
            for row, query in enumerate(batch):
                indexes = top[row]
                results[query] = (indexes, scores[row, indexes])
        return results


def rank(catalog, query, indexes, scores, name_filter, franchise_cap, popularity_weight):
    blended = scores + popularity_weight * catalog.popularity[indexes]
    chosen = []
    seen_franchises = set()
    query_tokens = catalog.name_tokens[query]
    query_franchise = catalog.franchise[query]
    for position in np.argsort(-blended, kind="stable"):
        index = indexes[position]
        if index == query or catalog.version_group[index] == catalog.version_group[query]:
            continue
        franchise = catalog.franchise[index]
        if query_franchise is not None and franchise == query_franchise:
            continue
        if name_filter and shares_name(query_tokens, catalog.name_tokens[index]):
            continue
        if franchise_cap and franchise is not None:
            if franchise in seen_franchises:
                continue
            seen_franchises.add(franchise)
        chosen.append(index)
        if len(chosen) == K:
            break
    return chosen


def build_igdb_splits(catalog, excluded):
    pairs = pd.read_parquet(RAW / "igdb_similar.parquet")
    by_igdb = pd.Series(catalog.games.index, index=catalog.igdb_id)
    pairs = pairs[pairs["similar_igdb_id"].isin(by_igdb.index)]
    expected = {}
    for igdb_id, rows in pairs.groupby("igdb_id"):
        query = int(by_igdb[igdb_id])
        if query in excluded:
            continue
        franchise = catalog.franchise[query]
        kept = {
            int(by_igdb[other])
            for other in rows["similar_igdb_id"]
            if catalog.version_group[by_igdb[other]] != catalog.version_group[query]
            and (franchise is None or catalog.franchise[by_igdb[other]] != franchise)
        }
        if len(kept) >= MIN_EXPECTED:
            expected[query] = kept
    splits = {"dev": [], "test": []}
    for query in sorted(expected):
        digest = hashlib.sha256(catalog.games.at[query, "game_id"].encode()).digest()[0]
        splits["dev" if digest % 2 == 0 else "test"].append(query)
    generator = np.random.default_rng(7)
    return expected, {
        name: sorted(generator.choice(queries, min(SAMPLE_PER_SPLIT, len(queries)), replace=False).tolist())
        for name, queries in splits.items()
    }


def score(catalog, pool, queries, expected, config):
    variant, name_filter, franchise_cap, weight = config
    values = []
    for query in queries:
        indexes, scores = pool[variant][query]
        top = rank(catalog, query, indexes, scores, name_filter, franchise_cap, weight)
        values.append(len(set(top) & expected[query]) / K)
    return np.array(values)


def interval(values, seed=42):
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[samples].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def label(config):
    variant, name_filter, franchise_cap, weight = config
    return f"{variant} name_filter={name_filter} franchise_cap={franchise_cap} popularity={weight}"


def main():
    catalog = Catalog()
    golden = load_golden(catalog.games)
    golden_queries = [int(catalog.position[q]) for q in golden["query_id"].unique()]
    golden_expected = {
        int(catalog.position[query]): {int(catalog.position[e]) for e in rows["expected_id"]}
        for query, rows in golden.groupby("query_id")
    }
    excluded = set(golden_queries)
    expected, splits = build_igdb_splits(catalog, excluded)
    print({name: len(queries) for name, queries in splits.items()}, "eligible", len(expected), flush=True)

    all_queries = sorted(set(splits["dev"]) | set(splits["test"]) | excluded)
    pool = {name: catalog.candidates(matrix, all_queries) for name, matrix in catalog.variants.items()}

    baseline = ("production", False, False, 0.0)
    configs = [
        (variant, name_filter, cap, weight)
        for variant in catalog.variants
        for name_filter in (False, True)
        for cap in (False, True)
        for weight in POPULARITY_WEIGHTS
    ]
    dev = sorted(
        ((score(catalog, pool, splits["dev"], expected, config).mean(), config) for config in configs),
        key=lambda item: -item[0],
    )
    print("--- dev, top 12")
    for value, config in dev[:12]:
        print(f"{value:.4f} {label(config)}")
    print("--- dev, one change at a time")
    single = [
        baseline,
        ("production", True, False, 0.0),
        ("production", False, True, 0.0),
        *[("production", False, False, weight) for weight in POPULARITY_WEIGHTS[1:]],
    ]
    if "without_name" in catalog.variants:
        single.append(("without_name", False, False, 0.0))
    for config in single:
        print(f"{score(catalog, pool, splits['dev'], expected, config).mean():.4f} {label(config)}")

    best = dev[0][1]
    report = {}
    for name, config in (("baseline", baseline), ("best_on_dev", best)):
        test_values = score(catalog, pool, splits["test"], expected, config)
        golden_values = score(catalog, pool, golden_queries, golden_expected, config)
        report[name] = {
            "config": label(config),
            "igdb_test": interval(test_values),
            "golden": interval(golden_values),
            "golden_hit_rate": float((golden_values > 0).mean()),
        }
    pairs = {
        "igdb_test": score(catalog, pool, splits["test"], expected, best) - score(catalog, pool, splits["test"], expected, baseline),
        "golden": score(catalog, pool, golden_queries, golden_expected, best) - score(catalog, pool, golden_queries, golden_expected, baseline),
    }
    report["difference"] = {name: interval(values) for name, values in pairs.items()}

    names = catalog.games["name"]
    report["examples"] = {
        names[query]: [names[i] for i in rank(catalog, query, *pool[best[0]][query], best[1], best[2], best[3])]
        for query in golden_queries
    }
    (ROOT / "data" / "experiment_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("--- report")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))
    for query, top in report["examples"].items():
        print(query, "->", top)


if __name__ == "__main__":
    sys.exit(main())
