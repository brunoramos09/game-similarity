import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd
import psycopg

from export_games import connection_uri
from resolve import ROOT

RAW = ROOT / "data" / "raw"
BATCH_SIZE = 500
PAUSE_SECONDS = 0.3

SIGNALS_QUERY = """
  WITH activity AS (
    SELECT game_id, COUNT(*) * 3 AS score FROM reviews GROUP BY game_id
    UNION ALL
    SELECT game_id, COUNT(*) FROM wishlist GROUP BY game_id
    UNION ALL
    SELECT game_id, COUNT(*) FROM list_games GROUP BY game_id
  )
  SELECT
    g.id::text AS game_id,
    g.igdb_id,
    COALESCE(g.version_root_igdb_id, g.igdb_id) AS version_group,
    g.developers,
    COALESCE(SUM(a.score), 0)::int AS checkpointd_activity
  FROM game_embeddings e
  JOIN games g ON g.id = e.game_id
  LEFT JOIN activity a ON a.game_id = g.id
  GROUP BY g.id
"""


def igdb_token():
    params = urllib.parse.urlencode(
        {
            "client_id": os.environ["IGDB_CLIENT_ID"],
            "client_secret": os.environ["IGDB_CLIENT_SECRET"],
            "grant_type": "client_credentials",
        }
    )
    request = urllib.request.Request(f"https://id.twitch.tv/oauth2/token?{params}", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["access_token"]


def fetch_similar(token, igdb_ids):
    body = f"fields similar_games; where id = ({','.join(map(str, igdb_ids))}); limit {len(igdb_ids)};".encode()
    headers = {"Client-ID": os.environ["IGDB_CLIENT_ID"], "Authorization": f"Bearer {token}"}
    request = urllib.request.Request("https://api.igdb.com/v4/games", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    with psycopg.connect(connection_uri()) as connection:
        connection.read_only = True
        rows = connection.execute(SIGNALS_QUERY).fetchall()
    signals = pd.DataFrame(rows, columns=["game_id", "igdb_id", "version_group", "developers", "checkpointd_activity"])
    signals.to_parquet(RAW / "signals.parquet")
    print(f"{len(signals)} signals, {(signals['checkpointd_activity'] > 0).sum()} games with activity", flush=True)

    token = igdb_token()
    ids = signals["igdb_id"].tolist()
    pairs = []
    for start in range(0, len(ids), BATCH_SIZE):
        for game in fetch_similar(token, ids[start : start + BATCH_SIZE]):
            pairs.extend({"igdb_id": game["id"], "similar_igdb_id": other} for other in game.get("similar_games", []))
        time.sleep(PAUSE_SECONDS)
    pd.DataFrame(pairs).to_parquet(RAW / "igdb_similar.parquet")
    print(f"{len(pairs)} IGDB similar pairs")


if __name__ == "__main__":
    main()
