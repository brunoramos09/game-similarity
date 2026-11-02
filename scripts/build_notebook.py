import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from resolve import ROOT

CELLS = [
    new_markdown_cell(
        "# Evaluating similar games\n\n"
        "How good are the \"similar games\" on Checkpointd? This notebook measures it with a hand-written golden set: "
        "30 games from different genres and eras, each with 5 games a fan would expect to see. "
        "The golden set was written before looking at any result.\n\n"
        "Retrieval works like production: cosine distance over `voyage-4-lite` embeddings (512 dimensions), "
        "leaving out the game itself and games from the same franchise, because the game page already has a franchise section."
    ),
    new_code_cell(
        "import sys\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "sys.path.insert(0, str(Path.cwd().parent / 'scripts'))\n"
        "from resolve import ROOT, load_golden\n\n"
        "K = 5\n"
        "games = pd.read_parquet(ROOT / 'data' / 'raw' / 'games.parquet').reset_index(drop=True)\n"
        "golden = load_golden(games)\n"
        "position = pd.Series(games.index, index=games['game_id'])\n"
        "len(games), golden['query'].nunique(), len(golden)"
    ),
    new_markdown_cell(
        "## Retrieval and metrics\n\n"
        "- **precision@5**: how many of the 5 results are in the golden set. Each game has exactly 5 expected games, so the maximum is 1.0.\n"
        "- **hit@5**: the share of games with at least one expected game in the top 5.\n\n"
        "A good result that isn't in the golden set counts as wrong, so these numbers are a lower bound on the real quality."
    ),
    new_code_cell(
        "def normalize(matrix):\n"
        "    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)\n\n\n"
        "def allowed_mask(query_index):\n"
        "    mask = np.ones(len(games), dtype=bool)\n"
        "    mask[query_index] = False\n"
        "    franchise = games.at[query_index, 'franchise']\n"
        "    if pd.notna(franchise):\n"
        "        mask &= (games['franchise'] != franchise).to_numpy()\n"
        "    return mask\n\n\n"
        "def top_k_by_score(scores, query_index):\n"
        "    ranked = np.where(allowed_mask(query_index), scores, -np.inf)\n"
        "    return np.argsort(-ranked, kind='stable')[:K]\n\n\n"
        "def evaluate(retrieve):\n"
        "    rows = []\n"
        "    for query_id, expected in golden.groupby('query_id', sort=False):\n"
        "        query_index = position[query_id]\n"
        "        top = games.loc[retrieve(query_index), 'game_id'].tolist()\n"
        "        hits = len(set(top) & set(expected['expected_id']))\n"
        "        rows.append({\n"
        "            'query': games.at[query_index, 'name'],\n"
        "            'hits': hits,\n"
        "            'precision_at_5': hits / K,\n"
        "            'results': games.set_index('game_id').loc[top, 'name'].tolist(),\n"
        "        })\n"
        "    return pd.DataFrame(rows)"
    ),
    new_markdown_cell(
        "## Three versions\n\n"
        "- **Production**: name, genres, themes and summary.\n"
        "- **Without themes**: the same text without the themes line. Games that have no themes keep their embedding, because their text is identical.\n"
        "- **Baseline**: no embeddings at all, just Jaccard overlap of genres and themes, with ties broken by popularity. If embeddings don't beat this, they aren't worth paying for.\n"
    ),
    new_code_cell(
        "production = normalize(np.array(games['embedding'].tolist(), dtype=np.float32))\n\n"
        "replacements = pd.read_parquet(ROOT / 'data' / 'raw' / 'embeddings_without_themes.parquet')\n"
        "without_themes = np.array(games['embedding'].tolist(), dtype=np.float32)\n"
        "without_themes[position[replacements['game_id']].to_numpy()] = np.array(replacements['embedding'].tolist(), dtype=np.float32)\n"
        "without_themes = normalize(without_themes)\n\n"
        "tags = [set(g) | set(t) for g, t in zip(games['genres'], games['themes'])]\n"
        "popularity = games['rating_count'].to_numpy(dtype=float)\n"
        "tie_break = popularity / (popularity.max() + 1) * 1e-6\n\n\n"
        "def jaccard_scores(query_index):\n"
        "    own = tags[query_index]\n"
        "    return np.array([len(own & t) / len(own | t) if own | t else 0.0 for t in tags]) + tie_break\n\n\n"
        "runs = {\n"
        "    'production': evaluate(lambda i: top_k_by_score(production @ production[i], i)),\n"
        "    'without_themes': evaluate(lambda i: top_k_by_score(without_themes @ without_themes[i], i)),\n"
        "    'baseline_jaccard': evaluate(lambda i: top_k_by_score(jaccard_scores(i), i)),\n"
        "}\n"
        "summary = pd.DataFrame({\n"
        "    name: {\n"
        "        'precision@5': run['precision_at_5'].mean(),\n"
        "        'hit@5': (run['hits'] > 0).mean(),\n"
        "        'games with 0 hits': int((run['hits'] == 0).sum()),\n"
        "    }\n"
        "    for name, run in runs.items()\n"
        "}).T\n"
        "summary"
    ),
    new_markdown_cell(
        "## Is the difference real?\n\n"
        "With only 30 games the average moves a lot by chance, so this resamples the games 10,000 times to get a 95% interval for each version and for the difference against production."
    ),
    new_code_cell(
        "generator = np.random.default_rng(42)\n"
        "samples = generator.integers(0, 30, size=(10_000, 30))\n"
        "base = runs['production']['precision_at_5'].to_numpy()\n"
        "rows = []\n"
        "for name, run in runs.items():\n"
        "    values = run['precision_at_5'].to_numpy()\n"
        "    means = values[samples].mean(axis=1)\n"
        "    differences = (values - base)[samples].mean(axis=1)\n"
        "    rows.append({\n"
        "        'version': name,\n"
        "        'ci_low': np.percentile(means, 2.5),\n"
        "        'ci_high': np.percentile(means, 97.5),\n"
        "        'diff_vs_production': (values - base).mean(),\n"
        "        'diff_low': np.percentile(differences, 2.5),\n"
        "        'diff_high': np.percentile(differences, 97.5),\n"
        "    })\n"
        "pd.DataFrame(rows).round(3)"
    ),
    new_markdown_cell("## Game by game"),
    new_code_cell(
        "pd.set_option('display.max_colwidth', None)\n"
        "pd.set_option('display.width', 250)\n"
        "per_game = runs['production'][['query', 'hits']].rename(columns={'hits': 'production'})\n"
        "per_game['without_themes'] = runs['without_themes']['hits']\n"
        "per_game['baseline_jaccard'] = runs['baseline_jaccard']['hits']\n"
        "per_game.sort_values('production', ascending=False).reset_index(drop=True)"
    ),
    new_markdown_cell("## Worst cases in production"),
    new_code_cell(
        "runs['production'].sort_values('hits').head(10)[['query', 'hits', 'results']].reset_index(drop=True)"
    ),
    new_markdown_cell(
        "## What was tried to improve it\n\n"
        "These come from `scripts/run_experiments.py`, `scripts/finetune.py` and `scripts/quality_floor.py`. "
        "Settings were chosen on a development split built from IGDB's similar games lists, then checked once on a separate test split and on the golden set."
    ),
    new_code_cell(
        "import json\n\n"
        "fine_tuning = json.loads((ROOT / 'data' / 'finetune_report.json').read_text())\n"
        "floor = json.loads((ROOT / 'data' / 'quality_floor_report.json').read_text())\n"
        "rows = [\n"
        "    {'version': name, 'golden': v['golden'][0], 'igdb_test': v['igdb_test'][0], 'golden_hit_rate': v['golden_hit_rate']}\n"
        "    for name, v in fine_tuning['versions'].items()\n"
        "]\n"
        "rows += [\n"
        "    {'version': f\"voyage + floor of {floor[name]['floor']} ratings\", 'golden': floor[name]['golden'][0], 'igdb_test': floor[name]['igdb_test'][0], 'golden_hit_rate': floor[name]['golden_hit_rate']}\n"
        "    for name in ['floor_1', 'selected']\n"
        "]\n"
        "pd.DataFrame(rows).round(3)"
    ),
    new_markdown_cell(
        "The fine-tuned model is the main lesson here. Trained on IGDB's lists, it got better at reproducing those lists and worse on the human golden set. "
        "Those lists are a noisy proxy, so it learned the proxy instead of the goal. It wasn't chosen, and the rating floor was chosen instead."
    ),
]


def main():
    notebook = new_notebook(cells=CELLS)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = ROOT / "notebooks" / "evaluate_similar_games.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
