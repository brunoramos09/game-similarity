from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GAME_COLUMNS = ["game_id", "name", "franchise", "release_date", "rating_count"]


def split_year(label):
    name, _, year = label.rpartition(" @")
    return (name, year) if year.isdigit() else (label, None)


def resolve(label, games):
    name, year = split_year(label)
    found = games[games["name"].str.casefold() == name.casefold()]
    if year:
        found = found[found["release_date"].astype(str).str.startswith(year)]
    if found.empty:
        raise KeyError(label)
    return found.sort_values("rating_count", ascending=False).iloc[0]


def load_golden(games):
    golden = pd.read_csv(ROOT / "data" / "golden_set.csv")
    golden["query_id"] = golden["query"].map(lambda label: resolve(label, games)["game_id"])
    golden["expected_id"] = golden["expected"].map(lambda label: resolve(label, games)["game_id"])
    return golden
