"""
app.py — Valora Movie Recommender  |  Streamlit UI
Backward-compatible upgrade: all original UI preserved + new features.

NEW
───
• Strategy options: Hybrid-SVD, SVD
• Popularity Penalty toggle in sidebar
• Full evaluation panel (Precision@K, Recall@K, Hit Rate, RMSE)
• Strategy comparison table
"""

import streamlit as st
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)

from recommender.utils import (
    load_tmdb_movies,
    load_movielens_ratings,
    load_movielens_movies,
    map_movielens_to_tmdb,
    build_tmdb_ratings_matrix,
    get_actual_poster,
)
from recommender.hybrid_engine import HybridRecommender
from recommender.evaluation import (
    evaluate_recommender,
    summarize_results,
    evaluate_full,
    compare_strategies,
    evaluate_user_holdout,
)

st.set_page_config(
    page_title="Valora Movie Recommender",
    page_icon="🎬",
    layout="wide",
)



# ─────────────────────────────────────────────────────────────────────
# THEME (Netflix-inspired: mostly black, subtle red accents, premium fonts)
# ─────────────────────────────────────────────────────────────────────
import base64
from pathlib import Path

NETFLIX_BG_PATH = "assets/bg.png"  # optional: place your chosen background here

def apply_netflix_theme(bg_path='assets/bg.png'):
    bg_css = ""
    if bg_path and Path(bg_path).exists():
        img_bytes = Path(bg_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        bg_css = f"""
        .stApp {{
            background-image:
              radial-gradient(circle at 20% 10%, rgba(223,7,7,0.14), rgba(0,0,0,0) 42%),
              linear-gradient(rgba(11,0,0,0.90), rgba(11,0,0,0.92)),
              url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        # Fallback: subtle pattern (no external asset required)
        bg_css = """
        .stApp {
            background-image:
              radial-gradient(circle at 20% 10%, rgba(223,7,7,0.12), rgba(0,0,0,0) 40%),
              linear-gradient(rgba(11,0,0,0.92), rgba(11,0,0,0.96)),
              url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Ctext x='50%25' y='55%25' font-family='Arial Black, sans-serif' font-size='72' fill='%23df0707' fill-opacity='0.05' text-anchor='middle' dominant-baseline='middle'%3EV%3C/text%3E%3C/svg%3E");
            background-size: auto, auto, 120px 120px;
            background-attachment: fixed;
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap');

        :root {{
            --bg: #0b0000;
            --red: #df0707;
            --red2: #ba0c0c;
            --red3: #980a0a;
            --text: rgba(255,255,255,0.92);
            --muted: rgba(255,255,255,0.68);
            --stroke: rgba(255,255,255,0.08);
            --card: rgba(18, 10, 10, 0.70);
        }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        }}
        {bg_css}

        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.86) !important;
            border-right: 1px solid rgba(223,7,7,0.22);
            backdrop-filter: blur(10px);
        }}

        .valora-title {{
            font-family: 'Libre Baskerville', serif;
            font-weight: 700;
            letter-spacing: 0.6px;
            color: var(--text);
            margin: 0.2rem 0 0.1rem 0;
        }}
        .valora-title .accent {{
            color: var(--red);
        }}
        .valora-subtitle {{
            color: var(--muted);
            margin-top: 0.35rem;
        }}

        div[data-testid="stAlert"] {{
            background: rgba(0,0,0,0.44);
            border: 1px solid rgba(223,7,7,0.18);
            border-radius: 14px;
            color: var(--text);
        }}

        .stButton>button {{
            background: linear-gradient(180deg, var(--red), var(--red2)) !important;
            color: white !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            width: 100% !important;
            letter-spacing: 0.3px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.35);
            transition: transform 0.08s ease, filter 0.15s ease;
        }}
        .stButton>button:hover {{
            filter: brightness(1.05);
            transform: translateY(-1px);
        }}

        div[data-baseweb="select"] > div {{
            background: rgba(0,0,0,0.42);
            border: 1px solid var(--stroke);
            border-radius: 12px;
        }}

        .movie-card {{
            background: var(--card);
            border-radius: 18px;
            padding: 12px 12px 14px 12px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.07);
            transition: transform 0.12s ease, border-color 0.15s ease, background 0.15s ease;
            margin-bottom: 20px;
            min-height: 360px;
            box-shadow: 0 14px 34px rgba(0,0,0,0.35);
        }}
        .movie-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(223,7,7,0.30);
            background: rgba(18, 10, 10, 0.82);
        }}
        .movie-title {{
            color: var(--text);
            font-weight: 700;
            margin-top: 10px;
            font-size: 14px;
            height: 2.5em;
            overflow: hidden;
        }}
        .score-tag {{
            color: rgba(255,255,255,0.78);
            font-size: 12px;
            font-weight: 600;
            margin-top: 6px;
        }}
        .score-tag b {{
            color: var(--red);
            font-weight: 700;
        }}
        .strategy-tag {{
            color: rgba(255,255,255,0.55);
            font-size: 10px;
            margin-top: 3px;
        }}

        .stDataFrame {{
            background: rgba(0,0,0,0.35);
            border: 1px solid var(--stroke);
            border-radius: 14px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


apply_netflix_theme(NETFLIX_BG_PATH)


# ─────────────────────────────────────────────────────────────────────
# DATA + ENGINE INIT  (cached across sessions)
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising Valora engine")
def initialize_system():
    tmdb     = load_tmdb_movies()
    ml_m     = load_movielens_movies()
    ml_r     = load_movielens_ratings()
    mapping  = map_movielens_to_tmdb(tmdb, ml_m)
    ratings  = build_tmdb_ratings_matrix(ml_r, mapping)
    engine   = HybridRecommender()
    engine.set_mapped_ratings(ratings)
    return tmdb, ratings, engine


tmdb_df, mapped_ratings, hybrid_engine = initialize_system()


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def score_label_for_strategy(strategy: str) -> str:
    """Return a user-friendly label for the score shown on movie cards."""
    s = (strategy or "").strip()
    return {
        "Content-Based": "Cosine similarity",
        "Collaborative": "Item-item similarity",
        "Hybrid": "Blended score (normalised)",
        "Hybrid-SVD": "Blended score (normalised)",
        "SVD": "Predicted rating (normalised)",
    }.get(s, "Score")


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <h2 style="
            color:#df0707;
            font-family:'Libre Baskerville', serif;
            font-weight:600;
            letter-spacing:0.4px;
            margin-bottom:0.5rem;
        ">
        Configuration
        </h2>
        """,
        unsafe_allow_html=True
    )
    movie_list = sorted(tmdb_df["title"].unique())
    # --- Sidebar: Controls (clean + technical) ---
    movie_list = sorted(tmdb_df["title"].unique())
    selected_movie_name = st.selectbox(
        "Search movie",
        [""] + movie_list,
        help="Pick a seed movie to generate recommendations."
    )

    # User-facing labels (clean + technical)
    STRATEGY_UI = {
        "Hybrid": "Hybrid",
        "Hybrid + SVD": "Hybrid-SVD",
        "Content": "Content-Based",
        "Collaborative": "Collaborative",
        "Matrix Factorization (SVD)": "SVD",
    }
    strategy_label = st.radio(
        "Recommendation strategy",
        list(STRATEGY_UI.keys()),
        help="Choose how the system ranks recommendations."
    )

    # Convert back to your internal strategy key
    strategy = STRATEGY_UI[strategy_label]

    # Only show User ID when strategy actually needs it
    selected_user = None
    if strategy in {"SVD", "Hybrid-SVD"}:
        user_ids = sorted(mapped_ratings["userId"].unique().tolist())
        selected_user = st.selectbox(
            "User ID (required for SVD)",
            [None] + user_ids,
            help="SVD-based methods personalize using user history."
        )

    top_k = st.slider(
        "Number of recommendations",
        1, 20, 6,
        help="How many movies to return."
    )

    st.divider()

    penalise_pop = st.toggle(
        "Popularity Bias Control",
        value=False,
        help="Adjust ranking to reduce dominance of highly popular titles."
    )

    show_compare = st.checkbox(
        "Model Performance Comparison",
        value=False,
        help="View offline evaluation metrics across recommendation strategies."
    )

    st.divider()

    predict_btn = st.button("Get recommendations", use_container_width=True)
# ─────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────
# --- Floating Contact Button (Top Right) ---
st.markdown(
    """
    <style>
    .contact-btn {
        position: fixed;
        top: 20px;
        right: 30px;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<h1 class="valora-title">Valora - Movie Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="valora-subtitle">Cinematic picks, powered by data. '
        'Integrating content similarity, collaborative signals, and matrix factorization models.</p>',
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border:0.5px solid rgba(255,255,255,0.08); margin-top:10px; margin-bottom:20px;'>", unsafe_allow_html=True)

import streamlit as st

CONTACT_EMAIL = "shreyaajoshi88@email.com"

st.markdown(
    f"""
    <style>
      .valora-contact {{
        position: fixed;
        top: 15px;
        right: 15px;
        z-index: 999999;
      }}

      .valora-contact summary {{
        list-style: none;
        cursor: pointer;
        padding: 8px 16px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: #fff;
        backdrop-filter: blur(10px);
        font-weight: 500;
        transition: 0.2s;
      }}

      .valora-contact summary:hover {{
        background: rgba(223, 7, 7, 0.2);
        border-color: rgba(223, 7, 7, 0.5);
      }}

      .valora-contact .panel {{
        position: absolute;
        right: 0;
        top: 100%;
        margin-top: 10px;
        width: 280px;
        background: rgba(15, 15, 15, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        color: white;
      }}

      .valora-contact .email {{
        margin-top: 6px;
        font-weight: 700;
        color: rgba(255,255,255,0.92);
        letter-spacing: 0.2px;
      }}

      .valora-contact a {{
        color: #ff4b4b;
        text-decoration: none;
      }}
      .valora-contact a:hover {{
        text-decoration: underline;
      }}
    </style>

    <div class="valora-contact">
      <details>
        <summary>Contact ▾</summary>
        <div class="panel">
          <b>Hello there!</b>
          <p>Hope you liked <b>Valora</b>.</p>
          <p>For queries, feedback, or suggestions:</p>
          <div class="email">{CONTACT_EMAIL}</div>
          <br>
          <a href="mailto:{CONTACT_EMAIL}?subject=Valora%20Feedback">📩 Send Email</a>
        </div>
      </details>
    </div>
    """,
    unsafe_allow_html=True
)
# Strategy description banner (technical + product-grade tone)
STRATEGY_DESC = {
    "Hybrid":
        "Hybrid architecture combining TF-IDF content similarity and item-item collaborative filtering with weighted score blending.",
    "Hybrid-SVD":
        "Extended hybrid model integrating content similarity, collaborative filtering, and SVD-based latent factor predictions.",
    "Content-Based":
        "Content-driven recommender using TF-IDF vectorization and cosine similarity over movie metadata.",
    "Collaborative":
        "Item-item collaborative filtering leveraging user interaction patterns and behavioral similarity.",
    "SVD":
        "Matrix factorization (Singular Value Decomposition) modeling latent user–item preference signals (requires User ID).",
}
st.info(f"**{strategy}** — {STRATEGY_DESC.get(strategy, '')}")

if strategy == "SVD" and selected_user is None:
    st.warning("SVD strategy requires a User ID. Please select one in the sidebar.")
if strategy == "Hybrid-SVD" and selected_user is None:
    st.info("💡 Hybrid-SVD uses global latent-factor scores without a User ID. "
            "Select a User ID for fully personalised SVD fusion.")


# ─────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────

if predict_btn:
    needs_seed = strategy in {"Content-Based", "Collaborative", "Hybrid", "Hybrid-SVD"}

    if needs_seed and selected_movie_name == "":
        st.error("Please select a seed movie first!")
    else:
        seed_id = None
        if selected_movie_name:
            seed_id = int(tmdb_df[tmdb_df["title"] == selected_movie_name]["tmdb_id"].values[0])

        with st.spinner("Scanning database…"):
            recommendations = hybrid_engine.recommend(
                user_id=selected_user,
                seed_movie_id=seed_id,
                strategy=strategy,
                top_k=top_k,
                penalise_popularity=penalise_pop,
            )

        if recommendations:
            label = f"Because you liked **{selected_movie_name}**:" if selected_movie_name else "Top Picks For You:"
            st.markdown(f"### {label}")

            cols = st.columns(min(6, len(recommendations)))
            score_label = score_label_for_strategy(strategy)
            for i, rec in enumerate(recommendations):
                col_index = i % len(cols)
                with cols[col_index]:
                    img_url = get_actual_poster(rec["tmdb_id"], title=rec.get("title", ""))
                    score   = rec.get("similarity_score", rec.get("svd_score", 0.0))
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{img_url}" style="width:100%; border-radius:5px;">
                            <div class="movie-title">{rec['title']}</div>
                            <div class="score-tag"><b>{score_label}</b>: {score:.2f}</div>
                            <div class="strategy-tag">{strategy}</div>
                        </div>
                    """, unsafe_allow_html=True)


            # ── Full offline comparison table ────────────────────────
            if show_compare:
                st.divider()
                st.subheader("📈 Strategy Comparison")
                st.caption("Evaluates on 20% held-out test set (up to 100 users) — may take ~30s")

                with st.spinner("Running offline evaluation…"):
                    def _make_func(s, sid):
                        def _fn(uid):
                            recs = hybrid_engine.recommend(
                                user_id=uid,
                                seed_movie_id=sid,
                                strategy=s,
                                top_k=top_k,
                            )
                            return [r["tmdb_id"] for r in recs]
                        return _fn

                    strategy_funcs = {
                        name: _make_func(name, seed_id)
                        for name in ["Hybrid", "Content-Based", "Collaborative"]
                    }
                    if selected_user:
                        strategy_funcs["SVD"] = _make_func("SVD", seed_id)

                    cmp_df = compare_strategies(
                        strategy_funcs,
                        mapped_ratings,
                        top_k=top_k,
                        svd_model=hybrid_engine.svd_model,
                        max_users=100,
                        engine=hybrid_engine,   # FIX: user-centric seeding
                    )

                # Format
                fmt = {c: "{:.3f}" for c in cmp_df.columns if c != "n_users_evaluated"}
                try:
                    styled = cmp_df.style.format(fmt).background_gradient(cmap="RdYlGn", axis=0)
                except Exception:
                    styled = cmp_df
                st.dataframe(styled, use_container_width=True)

        else:
            st.warning("No recommendations found. Try a different movie or strategy.")
else:
    st.markdown("""
    <div style='text-align:center; margin-top:80px; color:#555;'>
        <p style='font-size:18px;'>Select a movie and strategy in the sidebar, then click <b style='color:#df0707;'>GET RECOMMENDATIONS</b>.</p>
    </div>
    """, unsafe_allow_html=True)
