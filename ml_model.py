"""
ml_model.py
-----------
Unsupervised learning core of Maze Explorer.

Since we don't ship with a database of real players, we bootstrap the
K-Means model with a synthetic-but-realistic dataset of navigation
"archetypes" (speed runner, explorer, cautious navigator, path optimizer).
Every real playthrough recorded by the app is appended to
`data/player_logs.csv`, so the model can (optionally) be retrained on
real data over time via `retrain_from_logs()`.

Feature vector (in order):
    [completion_time_s, total_moves, wrong_turns, backtracks, hints_used, path_efficiency]
"""

import itertools
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "completion_time",
    "total_moves",
    "wrong_turns",
    "backtracks",
    "hints_used",
    "path_efficiency",
]

MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "cluster_labels.joblib")
LOG_PATH = os.path.join("data", "player_logs.csv")

N_CLUSTERS = 4

# Human-readable style descriptions, each with a "profile" used to
# score how well a cluster centroid matches the archetype.
STYLE_PROFILES = {
    "🚀 Speed Runner": dict(time=-2, moves=-2, wrong=-1, back=-1, hints=-1, eff=1),
    "🧭 Explorer": dict(time=1, moves=2, wrong=2, back=2, hints=0, eff=-1),
    "🧘 Cautious Navigator": dict(time=2, moves=1, wrong=-1, back=-1, hints=1, eff=0),
    "🎯 Path Optimizer": dict(time=-1, moves=-1, wrong=-2, back=-2, hints=-1, eff=2),
}
STYLE_NAMES = list(STYLE_PROFILES.keys())

STYLE_DESCRIPTIONS = {
    "🚀 Speed Runner": "Fast and decisive — few moves, minimal hesitation.",
    "🧭 Explorer": "Curious and thorough — happily wanders down side paths.",
    "🧘 Cautious Navigator": "Careful and deliberate — takes time to avoid mistakes.",
    "🎯 Path Optimizer": "Fast AND efficient — moves are close to the optimal route.",
}


# ---------------------------------------------------------------------- #
# Synthetic bootstrap dataset
# ---------------------------------------------------------------------- #
def _synthetic_dataset(n_per_cluster: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def sample(time_r, moves_r, wrong_r, back_r, hints_r, eff_r, n):
        return pd.DataFrame({
            "completion_time": rng.normal(*time_r, n).clip(10, None),
            "total_moves": rng.normal(*moves_r, n).clip(5, None),
            "wrong_turns": rng.normal(*wrong_r, n).clip(0, None),
            "backtracks": rng.normal(*back_r, n).clip(0, None),
            "hints_used": rng.normal(*hints_r, n).clip(0, None).round(),
            "path_efficiency": rng.normal(*eff_r, n).clip(0.3, 1.0),
        })

    # Speed Runner: fast time, few moves, takes some shortcuts/risks (moderate efficiency)
    rows.append(sample((40, 5), (18, 2), (1.5, 0.7), (1, 0.5), (0.1, 0.2), (0.62, 0.05), n_per_cluster))
    # Explorer: slow, many moves, lots of wrong turns / backtracks, rarely asks for hints
    rows.append(sample((230, 18), (85, 8), (14, 2.5), (10, 2), (0.2, 0.3), (0.35, 0.04), n_per_cluster))
    # Cautious Navigator: slow (leans on hints), moderate moves, but FEW mistakes
    rows.append(sample((170, 15), (40, 4), (1, 0.5), (1, 0.5), (3, 0.8), (0.75, 0.04), n_per_cluster))
    # Path Optimizer: fast AND highly efficient, almost no wasted moves
    rows.append(sample((50, 6), (24, 2), (0.2, 0.3), (0.1, 0.2), (0.1, 0.2), (0.95, 0.02), n_per_cluster))

    df = pd.concat(rows, ignore_index=True)
    return df[FEATURE_NAMES]


# ---------------------------------------------------------------------- #
# Cluster -> style-name assignment
# ---------------------------------------------------------------------- #
def _rank_score(values):
    """Convert raw values to rank order (0 = lowest) for robust comparison."""
    order = np.argsort(np.argsort(values))
    return order


def _assign_labels(centroids_unscaled: np.ndarray) -> dict[int, str]:
    """
    Rank clusters on each raw feature, then find the label permutation
    that maximizes agreement with each style's expected profile.
    """
    time_rank = _rank_score(centroids_unscaled[:, 0])
    moves_rank = _rank_score(centroids_unscaled[:, 1])
    wrong_rank = _rank_score(centroids_unscaled[:, 2])
    back_rank = _rank_score(centroids_unscaled[:, 3])
    hints_rank = _rank_score(centroids_unscaled[:, 4])
    eff_rank = _rank_score(centroids_unscaled[:, 5])

    n = centroids_unscaled.shape[0]

    def score_cluster_for_style(cluster_idx, style_name):
        p = STYLE_PROFILES[style_name]
        s = 0.0
        s += p["time"] * (time_rank[cluster_idx] - (n - 1) / 2)
        s += p["moves"] * (moves_rank[cluster_idx] - (n - 1) / 2)
        s += p["wrong"] * (wrong_rank[cluster_idx] - (n - 1) / 2)
        s += p["back"] * (back_rank[cluster_idx] - (n - 1) / 2)
        s += p["hints"] * (hints_rank[cluster_idx] - (n - 1) / 2)
        s += p["eff"] * (eff_rank[cluster_idx] - (n - 1) / 2)
        return s

    best_perm, best_total = None, -np.inf
    for perm in itertools.permutations(range(n)):
        # perm[i] = cluster index assigned to STYLE_NAMES[i]
        total = sum(score_cluster_for_style(perm[i], STYLE_NAMES[i]) for i in range(n))
        if total > best_total:
            best_total, best_perm = total, perm

    return {best_perm[i]: STYLE_NAMES[i] for i in range(n)}


# ---------------------------------------------------------------------- #
# Train / load
# ---------------------------------------------------------------------- #
def train_model(df: pd.DataFrame | None = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if df is None:
        df = _synthetic_dataset()

    X = df[FEATURE_NAMES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    kmeans.fit(X_scaled)

    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    labels_map = _assign_labels(centroids_unscaled)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(labels_map, LABELS_PATH)
    return scaler, kmeans, labels_map


def load_or_train():
    if all(os.path.exists(p) for p in (SCALER_PATH, KMEANS_PATH, LABELS_PATH)):
        try:
            scaler = joblib.load(SCALER_PATH)
            kmeans = joblib.load(KMEANS_PATH)
            labels_map = joblib.load(LABELS_PATH)
            return scaler, kmeans, labels_map
        except Exception:
            pass  # corrupted or incompatible cache file — fall through and retrain
    return train_model()


def retrain_from_logs():
    """Retrain using the synthetic dataset blended with any logged real games."""
    df = _synthetic_dataset()
    if os.path.exists(LOG_PATH):
        real = pd.read_csv(LOG_PATH)
        if not real.empty:
            df = pd.concat([df, real[FEATURE_NAMES]], ignore_index=True)
    return train_model(df)


# ---------------------------------------------------------------------- #
# Prediction
# ---------------------------------------------------------------------- #
def predict_style(feature_vector, scaler=None, kmeans=None, labels_map=None):
    """feature_vector: list/array in FEATURE_NAMES order."""
    if scaler is None or kmeans is None or labels_map is None:
        scaler, kmeans, labels_map = load_or_train()

    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    style = labels_map[cluster_id]

    # distance to assigned centroid vs. others -> a rough "confidence"
    distances = np.linalg.norm(kmeans.cluster_centers_ - X_scaled, axis=1)
    ordered = np.sort(distances)
    confidence = float(1 - ordered[0] / (ordered[0] + ordered[1] + 1e-9))

    return {
        "cluster_id": cluster_id,
        "style": style,
        "description": STYLE_DESCRIPTIONS[style],
        "confidence": round(confidence * 100, 1),
    }


def log_playthrough(feature_vector):
    """Append a real playthrough to the CSV log for future retraining."""
    os.makedirs("data", exist_ok=True)
    row = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
    if os.path.exists(LOG_PATH):
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, mode="w", header=True, index=False)