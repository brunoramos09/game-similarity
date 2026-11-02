import json

import numpy as np

from resolve import ROOT, load_golden
from run_experiments import K, Catalog, build_igdb_splits, interval, shares_name

FLOORS = [0, 1, 2, 3, 5, 10, 20]


def top_k(catalog, query, indexes, scores, name_filter, floor, ratings):
    chosen = []
    query_tokens = catalog.name_tokens[query]
    query_franchise = catalog.franchise[query]
    for position in np.argsort(-scores, kind="stable"):
        index = indexes[position]
        if index == query or catalog.version_group[index] == catalog.version_group[query]:
            continue
        if query_franchise is not None and catalog.franchise[index] == query_franchise:
            continue
        if ratings[index] < floor:
            continue
        if name_filter and shares_name(query_tokens, catalog.name_tokens[index]):
            continue
        chosen.append(index)
        if len(chosen) == K:
            break
    return chosen


def values(catalog, pool, queries, expected, name_filter, floor, ratings):
    return np.array(
        [len(set(top_k(catalog, q, *pool[q], name_filter, floor, ratings)) & expected[q]) / K for q in queries]
    )


def main():
    catalog = Catalog()
    ratings = catalog.games["rating_count"].to_numpy()
    golden = load_golden(catalog.games)
    golden_queries = [int(catalog.position[q]) for q in golden["query_id"].unique()]
    golden_expected = {
        int(catalog.position[q]): {int(catalog.position[e]) for e in rows["expected_id"]}
        for q, rows in golden.groupby("query_id")
    }
    expected, splits = build_igdb_splits(catalog, set(golden_queries))
    queries = sorted(set(splits["dev"]) | set(splits["test"]) | set(golden_queries))
    pool = catalog.candidates(catalog.production, queries)

    print("share of catalog per floor:", {f: round(float((ratings >= f).mean()), 3) for f in FLOORS})
    grid = []
    for name_filter in (False, True):
        for floor in FLOORS:
            mean = values(catalog, pool, splits["dev"], expected, name_filter, floor, ratings).mean()
            grid.append((mean, name_filter, floor))
            print(f"dev {mean:.4f} name_filter={name_filter} floor={floor}")
    _, best_filter, best_floor = max(grid)
    print("selected on dev:", best_filter, best_floor)

    configs = {"production": (False, 0), "selected": (best_filter, best_floor), "floor_1": (False, 1), "floor_1_name": (True, 1)}
    report = {}
    base_test = values(catalog, pool, splits["test"], expected, False, 0, ratings)
    base_golden = values(catalog, pool, golden_queries, golden_expected, False, 0, ratings)
    names = catalog.games["name"]
    for label, (name_filter, floor) in configs.items():
        test = values(catalog, pool, splits["test"], expected, name_filter, floor, ratings)
        gold = values(catalog, pool, golden_queries, golden_expected, name_filter, floor, ratings)
        report[label] = {
            "name_filter": name_filter,
            "floor": floor,
            "igdb_test": interval(test),
            "igdb_test_diff": interval(test - base_test),
            "golden": interval(gold),
            "golden_diff": interval(gold - base_golden),
            "golden_hit_rate": float((gold > 0).mean()),
        }
    print(json.dumps(report, indent=1))
    selected = configs["floor_1_name"]
    for q in golden_queries:
        top = top_k(catalog, q, *pool[q], selected[0], selected[1], ratings)
        print(names[q], "->", [f"{names[i]} ({ratings[i]})" for i in top])
    (ROOT / "data" / "quality_floor_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
