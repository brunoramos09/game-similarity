---
language: en
license: mit
base_model: BAAI/bge-small-en-v1.5
library_name: sentence-transformers
tags:
  - sentence-similarity
  - games
---

# game-similarity-bge-small

`BAAI/bge-small-en-v1.5` fine-tuned to find similar video games from their text. **It didn't work as hoped, and it's published as an honest record of that.** For real recommendations, the base model or a commercial embedding model did better in my evaluation. The full evaluation is on [GitHub](https://github.com/brunoramos09/game-similarity).

## Training

- **Data:** 24,048 pairs of games from IGDB's similar games lists, leaving out pairs of the same franchise or versions of the same game.
- **Text:** name, genres, themes and summary, up to 256 tokens.
- **Loss:** `MultipleNegativesRankingLoss`, batch size 64, learning rate 2e-5.
- **Epochs:** 3, picked on validation games kept out of training.
- **Hold-outs:** test games, golden set queries and their expected games were kept out of training.

## Evaluation

precision@5 on the same catalog of 31,749 games, with 95% bootstrap intervals.

| Model | Hand-written golden set (30 games) | IGDB test lists (3,000 games) |
| --- | --- | --- |
| `bge-small-en-v1.5` | 0.053 (0.020 to 0.093) | 0.015 (0.013 to 0.017) |
| **This model** | **0.020 (0.000 to 0.047)** | **0.074 (0.069 to 0.078)** |
| `voyage-4-lite` (production) | 0.127 (0.067 to 0.187) | 0.036 (0.033 to 0.040) |

## Why it got worse

It got about 5 times better at predicting IGDB's lists and worse on the human list. IGDB's lists are noisy: Portal's includes Outlast and Alien: Isolation, for example. The model learned to reproduce them, noise included, rather than what makes two games similar. The golden set has only 30 games and was drafted with an AI assistant, but the drop against production is outside the margin of error (interval 0.060 to 0.160).

Cleaner pairs would be the next thing to try, for example from people's lists or from games played by the same users.

## Limitations

English text only. Trained on a catalog of 31,749 games with English summaries, and not evaluated on anything else.
