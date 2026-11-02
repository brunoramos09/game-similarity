import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

from resolve import ROOT

MODEL = "voyage-4-lite"
DIMENSIONS = 512
BATCH_SIZE = 128
OUTPUT = ROOT / "data" / "raw" / "embeddings_without_themes.parquet"
API_URL = "https://api.voyageai.com/v1/embeddings"


def embed(texts):
    body = json.dumps({"model": MODEL, "input": texts, "input_type": "document", "output_dimension": DIMENSIONS}).encode()
    headers = {"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}", "Content-Type": "application/json"}
    for attempt in range(6):
        request = urllib.request.Request(API_URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            ordered = sorted(payload["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in ordered], payload["usage"]["total_tokens"]
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise SystemExit(f"Voyage returned {error.code}")
            time.sleep(2**attempt)
    raise SystemExit("Voyage kept failing")


def build_text(row):
    parts = [row["name"]]
    if len(row["genres"]) > 0:
        parts.append("Genres: " + ", ".join(row["genres"]))
    parts.append(row["summary"])
    return "\n".join(parts)


def main():
    games = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet", columns=["game_id", "name", "genres", "themes", "summary"])
    with_themes = games[games["themes"].map(len) > 0]
    texts = with_themes.apply(build_text, axis=1).tolist()
    ids = with_themes["game_id"].tolist()
    vectors, tokens = [], 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch, used = embed(texts[start : start + BATCH_SIZE])
        vectors.extend(batch)
        tokens += used
        if start % (BATCH_SIZE * 20) == 0:
            print(f"{start + len(batch)}/{len(texts)} tokens={tokens}", flush=True)
    pd.DataFrame({"game_id": ids, "embedding": vectors}).to_parquet(OUTPUT)
    print(f"done {len(vectors)} games, {tokens} tokens")


if __name__ == "__main__":
    main()
