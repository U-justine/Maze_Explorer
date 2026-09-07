"""
app.py
------
Maze Explorer
"""

import random
import re
import time

import streamlit as st

import ml_model
from maze import Maze
from utils import (
    compute_radar_scores,
    format_time,
    path_efficiency_chart,
    radar_chart,
    render_maze,
)

MAZE_SIZE = 10

st.set_page_config(
    page_title="Maze Explorer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# SVG icons
# ------------------------------------------------------------------ #
ICONS = {
    "compass": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
        '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="currentColor"/>'
        "</svg>"
    ),
    "target": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>'
    ),
    "star": (
        '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none">'
        '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 '
        '5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'
    ),
    "lightbulb": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M9 18h6"/><path d="M10 22h4"/>'
        '<path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/></svg>'
    ),
    "clock": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/></svg>'
    ),
    "footprints": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 16v-2.38C4 11.5 5.5 10 7.5 10S11 11.5 11 13.62V16"/>'
        '<path d="M13 16v-2.38C13 11.5 14.5 10 16.5 10S20 11.5 20 13.62V16"/>'
        '<path d="M7.5 10V8a2.5 2.5 0 0 1 5 0v2"/>'
        '<path d="M16.5 10V8a2.5 2.5 0 0 1 5 0v2"/></svg>'
    ),
    "x": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
    ),
    "cycle": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
        '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"/>'
        '<path d="M20.49 15a9 9 0 0 1-14.85 3.36L1 14"/></svg>'
    ),
    "rocket": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/>'
        '<path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>'
        '<path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>'
    ),
    "lotus": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 22c-4-3-6-7-6-11a6 6 0 0 1 12 0c0 4-2 8-6 11z"/>'
        '<path d="M12 22V11"/><path d="M8 14c-2-1-4-3-4-6"/><path d="M16 14c2-1 4-3 4-6"/></svg>'
    ),
    "bullseye": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>'
    ),
    "trophy": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
        '<path d="M4 22h16"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/></svg>'
    ),
    "chart": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>'
    ),
    "block": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>'
    ),
    "sparkle": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>'
    ),
    "fire": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 '
        '.5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>'
    ),
    "book": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
        '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>'
    ),
    "flag": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/>'
        '<line x1="4" y1="22" x2="4" y2="15"/></svg>'
    ),
    "map": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/>'
        '<line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>'
    ),
}


def icon(name, size="18px", color="currentColor"):
    svg = ICONS.get(name, "")
    if not svg:
        return ""
    svg = re.sub(
        r"<svg\b",
        f'<svg width="{size}" height="{size}" style="display:inline-block;vertical-align:middle"',
        svg,
        count=1,
    )
    if color != "currentColor":
        svg = svg.replace("currentColor", color)
    return svg


STYLE_ICONS = {
    "🚀 Speed Runner": "rocket",
    "🧭 Explorer": "compass",
    "🧘 Cautious Navigator": "lotus",
    "🎯 Path Optimizer": "bullseye",
}

FUN_FACTS = [
    "The world's largest hedge maze is 8 acres!",
    "Solving mazes improves your spatial memory.",
    "The shortest path is called the 'optimal route'.",
    "Ancient mazes were used in storytelling.",
    "Maze-solving is a classic testbed for AI agents.",
]

HERO_QUOTES = [
    "Every maze tells a story. What's yours?",
    "The shortest path isn't always the most fun.",
    "In every maze, there's a way out.",
]

# ------------------------------------------------------------------ #
# Custom CSS
# ------------------------------------------------------------------ #
st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #1a1d2e 0%, #0f1117 55%, #0b0d14 100%);
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4, #FFE66D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 4px;
        margin-bottom: 2px;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .hero-quote {
        text-align: center;
        color: #4ECDC4;
        font-size: 0.9rem;
        font-style: italic;
        opacity: 0.8;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    div.stButton > button {
        font-weight: 600;
        border-radius: 14px;
        border: none;
        padding: 0.5rem 1.5rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        width: 100% !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF6B6B, #ee5a24) !important;
        color: white !important;
        box-shadow: 0 4px 18px rgba(255, 107, 107, 0.3);
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 26px rgba(255, 107, 107, 0.45);
    }
    div.stButton > button[kind="secondary"] {
        background: rgba(78, 205, 196, 0.12) !important;
        border: 1px solid rgba(78, 205, 196, 0.35) !important;
        color: #e5e7eb !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        transform: translateY(-2px);
        border-color: #4ECDC4;
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.25);
    }
    .stat-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label {
        color: #94a3b8;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stat-value {
        color: #e5e7eb;
        font-size: 16px;
        font-weight: 600;
    }
    .stat-value.green { color: #4ECDC4; }
    .stat-value.red { color: #FF6B6B; }
    .stat-value.yellow { color: #FFE66D; }
    .stat-value.blue { color: #60a5fa; }
    .msg-bar {
        border-radius: 12px;
        background: rgba(78, 205, 196, 0.06);
        border: 1px solid rgba(78, 205, 196, 0.15);
        color: #e5e7eb;
        padding: 8px 16px;
        margin: 6px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
    }
    .progress-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
    }
    .result-card {
        background: linear-gradient(135deg, rgba(255,107,107,0.08), rgba(78,205,196,0.08));
        border-radius: 20px;
        padding: 26px;
        border: 1px solid rgba(255,255,255,0.06);
        text-align: center;
        margin: 18px 0;
    }
    .style-name {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-top: 8px;
    }
    .confidence-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.06);
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        color: #4ECDC4;
        margin-top: 10px;
    }
    .fun-fact {
        text-align: center;
        color: #4ECDC4;
        font-size: 0.9rem;
        padding: 12px;
        background: rgba(78, 205, 196, 0.04);
        border-radius: 10px;
        border: 1px dashed rgba(78, 205, 196, 0.15);
        margin-top: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .section-header {
        color: #e5e7eb;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.1rem;
    }
    /* Maze container and controls */
    .maze-container {
        background: rgba(0,0,0,0.2);
        border-radius: 16px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .maze-controls {
        display: flex;
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 6px;
    }
    .maze-controls > div {
        flex: 1;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #
@st.cache_resource
def get_model():
    return ml_model.load_or_train()


SCALER, KMEANS, LABELS_MAP = get_model()


# ------------------------------------------------------------------ #
# Game state
# ------------------------------------------------------------------ #
def new_game_state(seed=None):
    maze = Maze(size=MAZE_SIZE, seed=seed)
    return {
        "maze": maze,
        "pos": maze.start,
        "path_history": [maze.start],
        "start_time": time.time(),
        "moves": 0,
        "wrong_turns": 0,
        "backtracks": 0,
        "hints_used": 0,
        "last_direction": None,
        "hint_cell": None,
        "finished": False,
        "final_time": None,
        "celebrated": False,
        "message": "You're in! Find the exit →",
        "message_icon": "sparkle",
        "fun_fact": random.choice(FUN_FACTS),
        "quote": random.choice(HERO_QUOTES),
    }


def build_feature_vector(state):
    moves = max(state["moves"], 1)
    efficiency = min(state["maze"].optimal_length / moves, 1.0)
    return [
        state["final_time"],
        state["moves"],
        state["wrong_turns"],
        state["backtracks"],
        state["hints_used"],
        efficiency,
    ]


def do_move(state, direction):
    if state["finished"]:
        return
    maze = state["maze"]
    new_pos = maze.move(state["pos"], direction)

    if new_pos is None:
        state["message"] = "Wall ahead!"
        state["message_icon"] = "block"
        return

    if len(state["path_history"]) >= 2 and new_pos == state["path_history"][-2]:
        state["backtracks"] += 1

    state["path_history"].append(new_pos)
    state["pos"] = new_pos
    state["moves"] += 1
    state["last_direction"] = direction
    state["hint_cell"] = None

    if maze.degree(new_pos) == 1 and new_pos != maze.end:
        state["wrong_turns"] += 1

    if new_pos == maze.end:
        state["finished"] = True
        state["final_time"] = time.time() - state["start_time"]
        state["message"] = "You escaped! Amazing!"
        state["message_icon"] = "trophy"

        features = build_feature_vector(state)
        prediction = ml_model.predict_style(features, SCALER, KMEANS, LABELS_MAP)
        ml_model.log_playthrough(features)
        state["prediction"] = prediction
        state["features"] = features
    else:
        progress = state["moves"] / max(maze.optimal_length, 1)
        if progress < 0.4:
            state["message"] = "You're in! Find the exit →"
            state["message_icon"] = "sparkle"
        elif progress < 0.9:
            state["message"] = "Making progress — keep going!"
            state["message_icon"] = "compass"
        else:
            state["message"] = "You're close! The exit is near."
            state["message_icon"] = "fire"


def do_hint(state):
    if state["finished"]:
        return
    maze = state["maze"]
    direction, hint_pos = maze.next_hint_step(state["pos"])
    if hint_pos is None:
        state["message"] = "You're already at the exit!"
        state["message_icon"] = "flag"
        return
    state["hints_used"] += 1
    state["hint_cell"] = hint_pos
    state["message"] = "Follow the blue glow!"
    state["message_icon"] = "lightbulb"


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
if "game" not in st.session_state:
    st.session_state.game = new_game_state()

state = st.session_state.game

# ------------------------------------------------------------------ #
# Hero
# ------------------------------------------------------------------ #
st.markdown(
    f'<div class="hero-title">{icon("compass", "36px", "#FF6B6B")} MAZE EXPLORER</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="hero-subtitle">{icon("target", "16px", "#4ECDC4")} '
    f"Find your way out. Discover your style.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="hero-quote">{icon("star", "14px", "#FFE66D")} "{state["quote"]}"</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Status message
# ------------------------------------------------------------------ #
st.markdown(
    f'<div class="msg-bar">{icon(state["message_icon"], "18px", "#4ECDC4")}'
    f"<span>{state['message']}</span></div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Main play area
# ------------------------------------------------------------------ #
maze_col, stats_col = st.columns([2, 1])

with maze_col:
    st.markdown('<div class="maze-container">', unsafe_allow_html=True)
    
    fig = render_maze(
        state["maze"],
        state["pos"],
        hint_cell=state["hint_cell"],
        show_solution=state["finished"],
    )
    st.pyplot(fig, width='stretch')
    
    # ✅ BUTTONS UNDER THE MAZE
    st.markdown('<div class="maze-controls">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Adventure", icon=":material/map:", type="primary", use_container_width=True):
            st.session_state.game = new_game_state()
            st.rerun()
    with col2:
        if st.button(
            "💡 Need a Clue?",
            icon=":material/lightbulb:",
            type="secondary",
            disabled=state["finished"],
            use_container_width=True,
        ):
            do_hint(state)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with stats_col:
    st.markdown(
        f'<div class="section-header">{icon("chart", "20px", "#e5e7eb")} Your Progress</div>',
        unsafe_allow_html=True,
    )
    elapsed = state["final_time"] if state["finished"] else time.time() - state["start_time"]

    st.markdown(
        f"""
    <div style="background: rgba(255,255,255,0.02); border-radius: 12px; padding: 8px 14px;">
        <div class="stat-row">
            <span class="stat-label">{icon("clock", "15px", "#94a3b8")} Time</span>
            <span class="stat-value green">{format_time(elapsed)}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">{icon("footprints", "15px", "#94a3b8")} Moves</span>
            <span class="stat-value red">{state['moves']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">{icon("x", "15px", "#94a3b8")} Wrong</span>
            <span class="stat-value red">{state['wrong_turns']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">{icon("cycle", "15px", "#94a3b8")} Back</span>
            <span class="stat-value yellow">{state['backtracks']}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">{icon("lightbulb", "15px", "#94a3b8")} Clues Used</span>
            <span class="stat-value blue">{state['hints_used']}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not state["finished"]:
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;font-weight:500;color:#94a3b8;margin-bottom:6px;">Move</div>',
            unsafe_allow_html=True,
        )

        _, up_c, _ = st.columns([1, 1, 1])
        with up_c:
            if st.button("", icon=":material/keyboard_arrow_up:", key="up", use_container_width=True):
                do_move(state, "up")
                st.rerun()

        left_c, down_c, right_c = st.columns([1, 1, 1])
        with left_c:
            if st.button("", icon=":material/keyboard_arrow_left:", key="left", use_container_width=True):
                do_move(state, "left")
                st.rerun()
        with down_c:
            if st.button("", icon=":material/keyboard_arrow_down:", key="down", use_container_width=True):
                do_move(state, "down")
                st.rerun()
        with right_c:
            if st.button("", icon=":material/keyboard_arrow_right:", key="right", use_container_width=True):
                do_move(state, "right")
                st.rerun()

# ------------------------------------------------------------------ #
# Progress bar
# ------------------------------------------------------------------ #
if not state["finished"]:
    progress = min(state["moves"] / max(state["maze"].optimal_length * 1.2, 1), 1.0)
    st.markdown(
        f'<div class="progress-label">{icon("map", "14px", "#94a3b8")} '
        f"DISTANCE COVERED {int(progress * 100)}%</div>",
        unsafe_allow_html=True,
    )
    st.progress(progress)
else:
    st.markdown(
        f'<div class="progress-label">{icon("flag", "14px", "#4ECDC4")} MAZE COMPLETE</div>',
        unsafe_allow_html=True,
    )
    st.progress(1.0)

# ------------------------------------------------------------------ #
# Results
# ------------------------------------------------------------------ #
if state["finished"]:
    st.markdown("---")

    if not state["celebrated"]:
        st.balloons()
        state["celebrated"] = True

    prediction = state["prediction"]
    features = state["features"]
    efficiency = features[5]
    style_icon = STYLE_ICONS.get(prediction["style"], "target")

    st.markdown(
        f"""
    <div class="result-card">
        <div>{icon(style_icon, "48px", "#FF6B6B")}</div>
        <div class="style-name">{icon("trophy", "28px", "#FFE66D")} You are a {prediction['style']}</div>
        <div style="color:#94a3b8;font-size:1rem;margin:8px 0;">{prediction['description']}</div>
        <div class="confidence-pill">{icon("target", "14px", "#4ECDC4")} Confidence: {prediction['confidence']}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"| Completion | Moves | Wrong | Back | Hints | Optimal |\n"
        f"|---|---|---|---|---|---|\n"
        f"| {format_time(state['final_time'])} | {state['moves']} | {state['wrong_turns']} "
        f"| {state['backtracks']} | {state['hints_used']} | {state['maze'].optimal_length} moves |"
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(path_efficiency_chart(efficiency), width='stretch')
    with chart_col2:
        radar_scores = compute_radar_scores(
            {
                "completion_time": state["final_time"],
                "total_moves": state["moves"],
                "wrong_turns": state["wrong_turns"],
                "backtracks": state["backtracks"],
                "hints_used": state["hints_used"],
                "path_efficiency": efficiency,
            },
            state["maze"].optimal_length,
        )
        st.pyplot(radar_chart(radar_scores), width='stretch')

    st.markdown(
        f'<div class="fun-fact">{icon("book", "16px", "#4ECDC4")} {state["fun_fact"]}</div>',
        unsafe_allow_html=True,
    )

    # "Play Again" button now uses "New Adventure" label
    if st.button("🔄 New Adventure", icon=":material/replay:", type="primary", use_container_width=True):
        st.session_state.game = new_game_state()
        st.rerun()