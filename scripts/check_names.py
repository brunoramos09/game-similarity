import pandas as pd

from resolve import GAME_COLUMNS, ROOT, resolve

games = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet", columns=GAME_COLUMNS)
golden = pd.read_csv(ROOT / "data" / "golden_set.csv")
print("queries", golden["query"].nunique(), "pairs", len(golden), "per query", golden.groupby("query").size().unique())
problems = 0
for query_label, rows in golden.groupby("query", sort=False):
    query = resolve(query_label, games)
    parts = []
    for label in rows["expected"]:
        try:
            expected = resolve(label, games)
        except KeyError:
            parts.append(f"!!MISS {label}")
            problems += 1
            continue
        same = pd.notna(query["franchise"]) and query["franchise"] == expected["franchise"]
        problems += int(same)
        parts.append(f"{'!!SAME ' if same else ''}{expected['name']} ({str(expected['release_date'])[:4]})")
    print(f"{query['name']} ({str(query['release_date'])[:4]}) -> " + "; ".join(parts))
print("problems", problems)
