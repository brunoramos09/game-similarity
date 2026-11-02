import json

import numpy as np
import pandas as pd

from quality_floor import top_k
from resolve import ROOT, load_golden
from run_experiments import RAW, Catalog, normalize, rank

CATALOG_SIZE = 1500
FLOOR = 10


def label(games, index):
    row = games.loc[index]
    year = str(row["release_date"])[:4] if pd.notna(row["release_date"]) else ""
    return f"{row['name']} ({year})" if year else row["name"]


def metrics():
    finetune = json.loads((ROOT / "data" / "finetune_report.json").read_text())
    floor = json.loads((ROOT / "data" / "quality_floor_report.json").read_text())
    versions = [
        ("Fine-tuned bge-small", finetune["versions"]["bge_finetuned"]),
        ("First version (voyage-4-lite)", finetune["versions"]["voyage_production"]),
        ("Chosen (voyage-4-lite + floor of 10 ratings)", floor["selected"]),
    ]
    return [
        {
            "version": name,
            "golden": values["golden"],
            "igdb_test": values["igdb_test"],
            "hit_rate": values["golden_hit_rate"],
        }
        for name, values in versions
    ]


def main():
    catalog = Catalog()
    games = catalog.games
    ratings = games["rating_count"].to_numpy()
    golden = load_golden(games)
    golden_queries = [int(catalog.position[q]) for q in golden["query_id"].unique()]
    golden_expected = {
        int(catalog.position[query]): [int(catalog.position[e]) for e in rows["expected_id"]]
        for query, rows in golden.groupby("query_id")
    }

    top = games.sort_values("rating_count", ascending=False).index[:CATALOG_SIZE].tolist()
    queries = sorted({int(q) for q in top} | set(golden_queries))
    finetuned = normalize(np.load(RAW / "embeddings_bge_finetuned.npy"))
    voyage_pool = catalog.candidates(catalog.production, queries)
    finetuned_pool = catalog.candidates(finetuned, queries)

    results = {}
    order = sorted(queries, key=lambda q: (q not in golden_queries, -ratings[q]))
    for query in order:
        key = label(games, query)
        if key in results:
            continue
        entry = {
            "first": [label(games, i) for i in rank(catalog, query, *voyage_pool[query], False, False, 0.0)],
            "chosen": [label(games, i) for i in top_k(catalog, query, *voyage_pool[query], False, FLOOR, ratings)],
            "finetuned": [label(games, i) for i in rank(catalog, query, *finetuned_pool[query], False, False, 0.0)],
        }
        if query in golden_expected:
            entry["expected"] = [label(games, i) for i in golden_expected[query]]
        results[key] = entry

    output = {"metrics": metrics(), "games": dict(sorted(results.items(), key=lambda item: item[0].casefold()))}
    path = ROOT / "space" / "data.json"
    path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    golden_count = sum("expected" in entry for entry in results.values())
    print(f"{len(results)} games, {golden_count} from the golden set, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
