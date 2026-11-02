import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "raw" / "games.parquet"

QUERY = """
  SELECT
    g.id::text AS game_id,
    g.name,
    g.franchise,
    g.release_date::text AS release_date,
    g.igdb_rating_count AS rating_count,
    g.genres,
    g.themes,
    s.text AS summary,
    e.embedding::text AS embedding
  FROM game_embeddings e
  JOIN games g ON g.id = e.game_id
  JOIN game_summaries s ON s.game_id = g.id AND s.lang = 'en'
"""


def connection_uri():
    project = os.environ["NEON_PROJECT_ID"]
    api = f"https://console.neon.tech/api/v2/projects/{project}"
    headers = {"Authorization": f"Bearer {os.environ['NEON_API_KEY']}", "Accept": "application/json"}

    def get(path):
        request = urllib.request.Request(f"{api}{path}", headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    branches = get("/branches")["branches"]
    branch = next(b for b in branches if b.get("default"))
    database = get(f"/branches/{branch['id']}/databases")["databases"][0]
    query = f"?branch_id={branch['id']}&database_name={database['name']}&role_name={database['owner_name']}"
    return get(f"/connection_uri{query}")["uri"]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(connection_uri()) as connection:
        connection.read_only = True
        rows = connection.execute(QUERY).fetchall()
        columns = ["game_id", "name", "franchise", "release_date", "rating_count", "genres", "themes", "summary", "embedding"]
    games = pd.DataFrame(rows, columns=columns)
    games["embedding"] = games["embedding"].map(json.loads)
    games.to_parquet(OUTPUT)
    print(f"{len(games)} games exported to {OUTPUT}")


if __name__ == "__main__":
    main()
