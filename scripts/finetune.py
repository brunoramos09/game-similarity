import hashlib
import json

import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from resolve import ROOT, load_golden
from run_experiments import K, Catalog, RAW, build_igdb_splits, interval, normalize, score

BASE_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 13
OUTPUT = ROOT / "models" / "game-similarity-bge-small"
BASELINE_CONFIG = (False, False, 0.0)


def build_text(row):
    parts = [row["name"]]
    if len(row["genres"]) > 0:
        parts.append("Genres: " + ", ".join(row["genres"]))
    if len(row["themes"]) > 0:
        parts.append("Themes: " + ", ".join(row["themes"]))
    parts.append(row["summary"])
    return "\n".join(parts)


def bucket(game_id):
    return hashlib.sha256(game_id.encode()).digest()[0]


def encode(model, texts):
    return np.asarray(
        model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def evaluate_matrix(catalog, matrix, name, queries, expected):
    catalog.variants[name] = normalize(matrix)
    pool = {name: catalog.candidates(catalog.variants[name], queries)}
    return score(catalog, pool, queries, expected, (name, *BASELINE_CONFIG))


def main():
    catalog = Catalog()
    games = catalog.games
    texts = games.apply(build_text, axis=1).tolist()

    golden = load_golden(games)
    golden_queries = [int(catalog.position[q]) for q in golden["query_id"].unique()]
    golden_expected = {
        int(catalog.position[query]): {int(catalog.position[e]) for e in rows["expected_id"]}
        for query, rows in golden.groupby("query_id")
    }
    golden_games = set(golden_queries) | {i for values in golden_expected.values() for i in values}

    expected, splits = build_igdb_splits(catalog, set(golden_queries))
    buckets = games["game_id"].map(bucket).to_numpy()
    validation = [q for q in splits["dev"] if buckets[q] % 8 == 0]
    test = splits["test"]

    pairs = pd.read_parquet(RAW / "igdb_similar.parquet")
    by_igdb = pd.Series(games.index, index=catalog.igdb_id)
    pairs = pairs[pairs["igdb_id"].isin(by_igdb.index) & pairs["similar_igdb_id"].isin(by_igdb.index)]
    train_pairs = set()
    for first, second in zip(by_igdb[pairs["igdb_id"]].to_numpy(), by_igdb[pairs["similar_igdb_id"]].to_numpy()):
        first, second = int(first), int(second)
        usable = all(buckets[i] % 2 == 0 and buckets[i] % 8 != 0 and i not in golden_games for i in (first, second))
        same_franchise = catalog.franchise[first] is not None and catalog.franchise[first] == catalog.franchise[second]
        same_version = catalog.version_group[first] == catalog.version_group[second]
        if usable and not same_franchise and not same_version:
            train_pairs.add((min(first, second), max(first, second)))
    print(f"train pairs {len(train_pairs)}, validation queries {len(validation)}, test queries {len(test)}", flush=True)

    model = SentenceTransformer(BASE_MODEL, device="cuda")
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    base_matrix = encode(model, texts)
    np.save(RAW / "embeddings_bge_base.npy", base_matrix)
    best_score = evaluate_matrix(catalog, base_matrix, "bge_base", validation, expected).mean()
    best_epoch = 0
    best_matrix = base_matrix
    print(f"validation base {best_score:.4f}", flush=True)

    examples = [InputExample(texts=[texts[a], texts[b]]) for a, b in sorted(train_pairs)]
    np.random.default_rng(SEED).shuffle(examples)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    steps_per_epoch = len(loader)
    for epoch in range(1, EPOCHS + 1):
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=1,
            warmup_steps=int(steps_per_epoch * 0.1) if epoch == 1 else 0,
            optimizer_params={"lr": LEARNING_RATE},
            use_amp=True,
            show_progress_bar=False,
        )
        matrix = encode(model, texts)
        value = evaluate_matrix(catalog, matrix, f"bge_epoch_{epoch}", validation, expected).mean()
        print(f"validation epoch {epoch} {value:.4f}", flush=True)
        if value > best_score:
            best_score, best_epoch, best_matrix = value, epoch, matrix
            model.save(str(OUTPUT))

    np.save(RAW / "embeddings_bge_finetuned.npy", best_matrix)
    print(f"selected epoch {best_epoch}", flush=True)

    runs = {
        "voyage_production": catalog.production,
        "bge_base": base_matrix,
        "bge_finetuned": best_matrix,
    }
    per_query = {}
    report = {"train_pairs": len(train_pairs), "selected_epoch": best_epoch, "versions": {}}
    for name, matrix in runs.items():
        test_values = evaluate_matrix(catalog, matrix, name, test, expected)
        golden_values = evaluate_matrix(catalog, matrix, name, golden_queries, golden_expected)
        per_query[name] = (test_values, golden_values)
        report["versions"][name] = {
            "igdb_test": interval(test_values),
            "golden": interval(golden_values),
            "golden_hit_rate": float((golden_values > 0).mean()),
        }
    report["differences"] = {
        "finetuned_vs_base": {
            "igdb_test": interval(per_query["bge_finetuned"][0] - per_query["bge_base"][0]),
            "golden": interval(per_query["bge_finetuned"][1] - per_query["bge_base"][1]),
        },
        "finetuned_vs_production": {
            "igdb_test": interval(per_query["bge_finetuned"][0] - per_query["voyage_production"][0]),
            "golden": interval(per_query["bge_finetuned"][1] - per_query["voyage_production"][1]),
        },
    }
    names = games["name"]
    finetuned = normalize(best_matrix)
    report["examples"] = {}
    pool = catalog.candidates(finetuned, golden_queries)
    catalog.variants["bge_finetuned"] = finetuned
    from run_experiments import rank

    for query in golden_queries:
        top = rank(catalog, query, *pool[query], False, False, 0.0)
        report["examples"][names[query]] = [names[i] for i in top[:K]]
    (ROOT / "data" / "finetune_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))
    for query, top in report["examples"].items():
        print(query, "->", top)


if __name__ == "__main__":
    main()
