"""
utils.py — Shared utilities for Valora recommender system.
Backward-compatible: all original public functions preserved.
Poster system upgraded for deployment stability:
- env-based TMDB key (with optional fallback)
- stronger ID validation
- safer HTTP calls + timeouts
- cache pre-warm support from tmdb_5000_movies.csv poster_path
"""

import os
import re
import logging
from pathlib import Path
from difflib import get_close_matches

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# In-memory poster cache
# ─────────────────────────────────────────────
_poster_cache: dict = {}
_tmdb_id_cache: dict = {}   # title → tmdb_id fallback cache
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://placehold.co/500x750/1a1a1a/e50914?text=Poster+Unavailable"


# ─────────────────────────────────────────────
# Title cleaning
# ─────────────────────────────────────────────
def clean_title(title: str) -> str:
    if pd.isna(title):
        return ""
    title = title.lower()
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"[^a-z0-9 ]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


# ─────────────────────────────────────────────
# Robust path resolution (works when script is run from any cwd)
# ─────────────────────────────────────────────
def resolve_data_path(path: str) -> str:
    """Resolve a data file path reliably across different working directories."""
    p = Path(path)

    # 1) Absolute path
    if p.is_absolute() and p.exists():
        return str(p)

    # 2) Relative to current working directory
    if p.exists():
        return str(p.resolve())

    here = Path(__file__).resolve().parent

    # 3) Relative to this module / project root candidates
    candidates = []
    for parent in [here] + list(here.parents)[:6]:
        candidates.append(parent / p)
        # common layout: <root>/data/<file>
        if "data" not in p.parts:
            candidates.append(parent / "data" / p.name)

    for c in candidates:
        if c.exists():
            return str(c)

    tried = [str(c) for c in candidates[:10]]
    raise FileNotFoundError(
        f"Could not find data file '{path}'. Tried (sample): {tried}. "
        "Fix by placing files under a 'data/' folder at project root, or pass an absolute path."
    )


# ─────────────────────────────────────────────
# Load TMDB data
# ─────────────────────────────────────────────
def load_tmdb_movies(path="data/tmdb_5000_movies.csv") -> pd.DataFrame:
    df = pd.read_csv(resolve_data_path(path))
    if "poster_path" not in df.columns:
        df["poster_path"] = np.nan
    df = df[["id", "title", "genres", "overview", "vote_average", "popularity", "poster_path"]]
    df.rename(columns={"id": "tmdb_id"}, inplace=True)
    df["clean_title"] = df["title"].apply(clean_title)
    return df


# ─────────────────────────────────────────────
# Load MovieLens data
# ─────────────────────────────────────────────
def load_movielens_movies(path="data/movielens_movies.csv") -> pd.DataFrame:
    df = pd.read_csv(resolve_data_path(path))
    df["clean_title"] = df["title"].apply(clean_title)
    return df


def load_movielens_ratings(path="data/movielens_ratings.csv") -> pd.DataFrame:
    return pd.read_csv(resolve_data_path(path))


# ─────────────────────────────────────────────
# TMDB ↔ MovieLens mapping
# ─────────────────────────────────────────────
def map_movielens_to_tmdb(tmdb_df, ml_movies_df, cutoff=0.85):
    tmdb_titles = tmdb_df["clean_title"].tolist()
    tmdb_lookup = dict(zip(tmdb_df["clean_title"], tmdb_df["tmdb_id"]))
    rows = []
    for _, row in ml_movies_df.iterrows():
        matches = get_close_matches(row["clean_title"], tmdb_titles, n=1, cutoff=cutoff)
        if matches:
            rows.append({"movieId": row["movieId"], "tmdb_id": tmdb_lookup[matches[0]]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Build TMDB ratings dataset
# ─────────────────────────────────────────────
def build_tmdb_ratings_matrix(ratings_df, mapping_df):
    merged = ratings_df.merge(mapping_df, on="movieId", how="inner")
    return merged[["userId", "tmdb_id", "rating"]]


# ─────────────────────────────────────────────
# User-Item Matrix (required by collaborative)
# ─────────────────────────────────────────────
def create_user_item_matrix(ratings_df):
    """Pivot table required for collaborative filtering."""
    return ratings_df.pivot_table(index="userId", columns="tmdb_id", values="rating")


# ─────────────────────────────────────────────
# Score normalisation utility
# ─────────────────────────────────────────────
def normalize_scores(score_dict: dict) -> dict:
    """
    Min-Max normalise a {movie_id: score} dict to [0, 1].
    Safe against division-by-zero (all-equal scores → all 0.5).
    """
    if not score_dict:
        return {}
    values = np.array(list(score_dict.values()), dtype=float)
    min_v, max_v = values.min(), values.max()
    if max_v - min_v < 1e-9:
        return {k: 0.5 for k in score_dict}
    return {k: float((v - min_v) / (max_v - min_v)) for k, v in score_dict.items()}


# ─────────────────────────────────────────────
# Popularity-bias penalty
# ─────────────────────────────────────────────
def apply_popularity_penalty(score_dict: dict, tmdb_df: pd.DataFrame) -> dict:
    """
    Penalise hyper-popular movies so niche gems surface.
    adjusted = score / log(1 + popularity)
    """
    pop_map = tmdb_df.set_index("tmdb_id")["popularity"].to_dict()
    result = {}
    for movie_id, score in score_dict.items():
        pop = pop_map.get(movie_id, 1.0)
        result[movie_id] = score / np.log1p(max(pop, 1e-3))
    return result


# ─────────────────────────────────────────────
# Poster pre-cache (optional)
# ─────────────────────────────────────────────
def precache_posters_from_tmdb_df(tmdb_df: pd.DataFrame) -> int:
    """
    Pre-warm _poster_cache using poster_path already available in tmdb_5000_movies.csv.
    This avoids most live API calls and improves stability on deployment.
    Returns number of posters cached.
    """
    if tmdb_df is None or tmdb_df.empty:
        return 0

    cached = 0
    if "poster_path" not in tmdb_df.columns:
        return 0

    for _, row in tmdb_df[["tmdb_id", "poster_path"]].dropna().iterrows():
        tmdb_id = row["tmdb_id"]
        path = row["poster_path"]
        try:
            key = str(int(tmdb_id))
        except Exception:
            continue
        if key not in _poster_cache and isinstance(path, str) and path.strip():
            _poster_cache[key] = f"{TMDB_IMAGE_BASE}{path}"
            cached += 1
    return cached


# ─────────────────────────────────────────────
# Poster fetching with cache + search fallback
# ─────────────────────────────────────────────
def _safe_int_tmdb_id(tmdb_id):
    """Convert tmdb_id to int if valid, else None."""
    try:
        if tmdb_id is None:
            return None
        if str(tmdb_id).lower() == "nan":
            return None
        return int(tmdb_id)
    except Exception:
        return None


def _fetch_poster_by_id(tmdb_id: int) -> str | None:
    """Direct TMDB movie endpoint."""
    if not TMDB_API_KEY:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        r = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=4)
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return f"{TMDB_IMAGE_BASE}{path}"
        # rate limit / not found / other errors just fall through
    except Exception as e:
        logger.debug("Poster fetch by ID failed: %s", e)
    return None


def _search_tmdb_by_title(title: str) -> tuple[int | None, str | None]:
    """
    TMDB search endpoint fallback when TMDB id is missing/invalid.
    Returns (tmdb_id, poster_url) or (None, None).
    """
    if not TMDB_API_KEY:
        return None, None

    title = (title or "").strip()
    if not title:
        return None, None

    if title in _tmdb_id_cache:
        cached_id = _tmdb_id_cache[title]
        poster = _fetch_poster_by_id(cached_id)
        return cached_id, poster

    try:
        url = "https://api.themoviedb.org/3/search/movie"
        r = requests.get(
            url,
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=4,
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                hit = results[0]
                found_id = hit.get("id")
                path = hit.get("poster_path")
                if found_id:
                    _tmdb_id_cache[title] = found_id
                if path:
                    return found_id, f"{TMDB_IMAGE_BASE}{path}"
    except Exception as e:
        logger.debug("TMDB title search failed: %s", e)

    return None, None


def get_actual_poster(tmdb_id, title: str = "") -> str:
    """
    Fetches real poster URL with multi-level fallback:
      1) In-memory cache hit (includes pre-warmed cache)
      2) Direct TMDB movie API (by id)
      3) TMDB search API (by title)
      4) Placeholder image
    """
    tmdb_id_int = _safe_int_tmdb_id(tmdb_id)
    cache_key = str(tmdb_id_int) if tmdb_id_int is not None else f"none:{title}"

    # 1) Cache
    if cache_key in _poster_cache:
        return _poster_cache[cache_key]

    poster = None

    # 2) Direct lookup
    if tmdb_id_int is not None:
        poster = _fetch_poster_by_id(tmdb_id_int)

    # 3) Title search fallback
    if not poster and title:
        _, poster = _search_tmdb_by_title(title)

    # 4) Placeholder
    if not poster:
        poster = PLACEHOLDER

    _poster_cache[cache_key] = poster
    return poster