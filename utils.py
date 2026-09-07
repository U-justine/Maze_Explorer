"""
utils.py
--------
Rendering helpers: the maze itself, plus the results screen's
path-efficiency bar and radar chart. All return matplotlib Figures,
which Streamlit displays directly via st.pyplot().
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow

PALETTE = {
    "bg": "#0f1117",
    "wall": "#4ECDC4",
    "path": "#FF6B6B",
    "player": "#FFE66D",
    "start": "#2ECC71",
    "end": "#E74C3C",
    "hint": "#60a5fa",
    "text": "#e5e7eb",
}


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------- #
# Maze rendering
# ---------------------------------------------------------------------- #
def render_maze(maze, player_pos, hint_cell=None, show_solution=False):
    n = maze.size
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Draw walls: for each cell, draw the borders that are NOT open passages
    for r in range(n):
        for c in range(n):
            open_dirs = maze.connections[r][c]
            x, y = c, n - 1 - r  # flip so row 0 is at top
            if "up" not in open_dirs:
                ax.plot([x, x + 1], [y + 1, y + 1], color=PALETTE["wall"], lw=2.4, solid_capstyle="round")
            if "down" not in open_dirs:
                ax.plot([x, x + 1], [y, y], color=PALETTE["wall"], lw=2.4, solid_capstyle="round")
            if "left" not in open_dirs:
                ax.plot([x, x], [y, y + 1], color=PALETTE["wall"], lw=2.4, solid_capstyle="round")
            if "right" not in open_dirs:
                ax.plot([x + 1, x + 1], [y, y + 1], color=PALETTE["wall"], lw=2.4, solid_capstyle="round")

    if show_solution:
        xs, ys = [], []
        for (r, c) in maze.solution_path:
            xs.append(c + 0.5)
            ys.append(n - 1 - r + 0.5)
        ax.plot(xs, ys, color=PALETTE["path"], lw=2.2, alpha=0.5, zorder=2)

    # Start / End markers
    sr, sc = maze.start
    er, ec = maze.end
    ax.add_patch(Circle((sc + 0.5, n - 1 - sr + 0.5), 0.28, color=PALETTE["start"], zorder=3))
    ax.add_patch(Circle((ec + 0.5, n - 1 - er + 0.5), 0.28, color=PALETTE["end"], zorder=3))
    ax.text(sc + 0.5, n - 1 - sr + 0.5, "S", ha="center", va="center", fontsize=9, fontweight="bold", color="#052e16", zorder=4)
    ax.text(ec + 0.5, n - 1 - er + 0.5, "E", ha="center", va="center", fontsize=9, fontweight="bold", color="#450a0a", zorder=4)

    if hint_cell is not None:
        hr, hc = hint_cell
        ax.add_patch(Circle((hc + 0.5, n - 1 - hr + 0.5), 0.34, facecolor="none",
                             edgecolor=PALETTE["hint"], lw=2.5, zorder=5))

    # Player marker
    pr, pc = player_pos
    ax.add_patch(Circle((pc + 0.5, n - 1 - pr + 0.5), 0.24, color=PALETTE["player"], zorder=6))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    plt.close(fig)  # release from pyplot's global figure manager (Streamlit still renders it fine)
    return fig


# ---------------------------------------------------------------------- #
# Results visualizations
# ---------------------------------------------------------------------- #
def path_efficiency_chart(efficiency: float):
    fig, ax = plt.subplots(figsize=(5.5, 1.4), dpi=110)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.barh([0], [1.0], color="#374151", height=0.5)
    ax.barh([0], [efficiency], color=PALETTE["path"], height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=PALETTE["text"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(efficiency, 0, f" {efficiency*100:.0f}%", va="center",
             ha="left" if efficiency < 0.9 else "right",
             color=PALETTE["text"], fontsize=10, fontweight="bold")
    ax.set_title("Path Efficiency (your moves vs. optimal route)",
                 color=PALETTE["text"], fontsize=10, loc="left")
    fig.tight_layout(pad=0.4)
    plt.close(fig)
    return fig


def radar_chart(scores: dict):
    """scores: dict of {dimension_name: 0-100 value}"""
    labels = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=110, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.plot(angles, values, color=PALETTE["hint"], lw=2)
    ax.fill(angles, values, color=PALETTE["hint"], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([])
    ax.set_ylim(0, 100)
    ax.spines["polar"].set_color("#374151")
    ax.grid(color="#374151", alpha=0.6)
    ax.set_title("Performance Radar", color=PALETTE["text"], fontsize=11, pad=18)
    fig.tight_layout(pad=0.6)
    plt.close(fig)
    return fig


def compute_radar_scores(stats: dict, optimal_length: int):
    """Map raw stats into five 0-100 'nicer is higher' dimensions."""
    moves = max(stats["total_moves"], 1)
    speed = max(0, 100 - stats["completion_time"] / 3)
    efficiency = stats["path_efficiency"] * 100
    accuracy = max(0, 100 - (stats["wrong_turns"] / moves) * 300)
    consistency = max(0, 100 - (stats["backtracks"] / moves) * 300)
    independence = max(0, 100 - stats["hints_used"] * 25)
    return {
        "Speed": round(min(speed, 100), 1),
        "Efficiency": round(min(efficiency, 100), 1),
        "Accuracy": round(min(accuracy, 100), 1),
        "Consistency": round(min(consistency, 100), 1),
        "Independence": round(min(independence, 100), 1),
    }