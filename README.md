---
title: Maze Explorer
emoji: 🧩
colorFrom: teal
colorTo: pink
sdk: streamlit
app_file: app.py
pinned: false
suggested_hardware: cpu-basic
short_description: Navigate a maze — K-Means clustering reveals your play style
---

# 🧩 Maze Explorer

An interactive maze game that uses **K-Means clustering** to classify how you play — not just whether you won.

## What it does

You navigate a randomly generated 10×10 maze with arrow buttons. While you play, the app quietly
records your completion time, total moves, wrong turns, backtracks, and hint usage. When you finish,
it engineers a feature vector, scales it, and feeds it into a trained K-Means model to classify your
**navigation style** into one of four archetypes:

| Style | Traits |
|---|---|
| 🚀 Speed Runner | Fast, few moves, takes some risks |
| 🧭 Explorer | Slow, wanders, lots of wrong turns/backtracks |
| 🧘 Cautious Navigator | Slow but careful, leans on hints, few mistakes |
| 🎯 Path Optimizer | Fast **and** highly efficient — near-optimal route |

## Project structure

```
maze-explorer/
├── app.py              # Streamlit application (UI + game loop)
├── maze.py             # Maze generation (DFS) + solving (BFS)
├── ml_model.py         # StandardScaler + K-Means training/prediction
├── utils.py            # Maze rendering, efficiency bar, radar chart
├── requirements.txt
├── data/               # player_logs.csv (created after first playthrough)
└── models/             # scaler.joblib, kmeans.joblib, cluster_labels.joblib
```

## How the ML actually works

1. **Bootstrap data**: since there's no player history on first run, `ml_model._synthetic_dataset()`
   generates ~120 synthetic playthroughs per archetype with realistic noise.
2. **Scaling**: `StandardScaler` normalizes all 6 features so no single metric (e.g. raw seconds)
   dominates the distance calculation.
3. **Clustering**: `KMeans(n_clusters=4)` fits on the scaled data.
4. **Labeling**: cluster indices from K-Means are arbitrary (cluster "2" isn't inherently
   "Explorer"). `_assign_labels()` ranks each cluster's centroid across all 6 features and solves
   a small assignment problem to match each cluster to the archetype it best resembles.
5. **Prediction**: a real playthrough's feature vector is scaled with the *same* fitted scaler,
   then assigned to the nearest centroid. A rough confidence score compares the distance to the
   nearest vs. second-nearest centroid.
6. **Continuous learning**: every real playthrough is appended to `data/player_logs.csv`.
   Call `ml_model.retrain_from_logs()` to retrain K-Means on synthetic + real data blended together.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Deploying to Hugging Face Spaces (CPU Basic)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space), choose the **Streamlit** SDK,
   and set hardware to **CPU basic** (the free tier — plenty for this app; no GPU needed since it's
   just K-Means, BFS, and matplotlib, all lightweight).
2. Creating the Space this way auto-fills a `sdk_version` in `README.md`'s YAML block pinned to
   the latest Streamlit version HF currently supports — leave that as-is rather than hand-editing it,
   since not every Streamlit version is supported and a stale hardcoded version can break the build.
3. Upload `app.py`, `maze.py`, `ml_model.py`, `utils.py`, `requirements.txt`, and this `README.md`
   (the YAML block at the top of this file is what HF reads to configure the Space — don't remove it).
4. The Space installs `requirements.txt` and launches `app.py` automatically. The K-Means model
   trains itself on first boot (a few hundred synthetic points — this takes well under a second,
   even on CPU basic), so no pre-trained artifacts need to be uploaded.

One CPU-basic-specific note: Spaces on the free tier go to sleep after a period of inactivity and
cold-start on the next visit. The model retrains from scratch on every cold start (it's fast), but
`data/player_logs.csv` — the log of real playthroughs — does **not** persist across restarts unless
you enable persistent storage, since CPU basic uses ephemeral disk.

## Notes on the game mechanics

- **Wrong turn**: counted when you step into a cell with only one open passage (a dead end)
  that isn't the exit — you'll have to turn back.
- **Backtrack**: counted when your very next move undoes your previous move (stepping back
  to the cell you just came from).
- **Path efficiency**: `optimal_path_length / your_total_moves`, capped at 100%.
- **Hint**: runs a fresh BFS from your *current* position to the exit and highlights the next
  step — it always adapts to wherever you've wandered.
