import re

import pandas as pd

from embed_without_themes import BATCH_SIZE, embed
from resolve import ROOT

OUTPUT = ROOT / "data" / "raw" / "embeddings_without_name.parquet"


def build_text(row):
    pattern = re.compile(re.escape(row["name"]), re.IGNORECASE)
    parts = []
    if len(row["genres"]) > 0:
        parts.append("Genres: " + ", ".join(row["genres"]))
    if len(row["themes"]) > 0:
        parts.append("Themes: " + ", ".join(row["themes"]))
    parts.append(pattern.sub("this game", row["summary"]))
    return "\n".join(parts)


def main():
    games = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet", columns=["game_id", "name", "genres", "themes", "summary"])
    texts = games.apply(build_text, axis=1).tolist()
    ids = games["game_id"].tolist()
    vectors, tokens = [], 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch, used = embed(texts[start : start + BATCH_SIZE])
        vectors.extend(batch)
        tokens += used
        if start % (BATCH_SIZE * 40) == 0:
            print(f"{start + len(batch)}/{len(texts)} tokens={tokens}", flush=True)
    pd.DataFrame({"game_id": ids, "embedding": vectors}).to_parquet(OUTPUT)
    print(f"done {len(vectors)} games, {tokens} tokens")


if __name__ == "__main__":
    main()
