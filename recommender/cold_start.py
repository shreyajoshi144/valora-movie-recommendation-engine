import pandas as pd
import ast

from recommender.utils import load_tmdb_movies


# -------------------------------------------------
# Helper: parse genres safely
# -------------------------------------------------

def parse_genres(genres_str):
    """
    Converts TMDB genre string to list of genre names.
    """
    try:
        genres = ast.literal_eval(genres_str)
        return [g["name"].lower() for g in genres]
    except Exception:
        return []


# -------------------------------------------------
# Popularity-based recommendation (true cold start)
# -------------------------------------------------

def recommend_popular_movies(top_k=10):
    """
    Recommends globally popular and well-rated movies.
    Used when:
    - new user
    - no interaction history
    """

    tmdb_df = load_tmdb_movies()

    popular = (
        tmdb_df
        .sort_values(
            by=["vote_average", "popularity"],
            ascending=False
        )
        .head(top_k)
    )

    return popular[
        ["tmdb_id", "title", "vote_average", "popularity"]
    ].to_dict(orient="records")


# -------------------------------------------------
# Genre-based fallback
# -------------------------------------------------

def recommend_by_genre(
    preferred_genres,
    top_k=10
):
    """
    Recommends movies based on genre preference.
    Used when:
    - user selected genres
    - limited interaction data
    """

    tmdb_df = load_tmdb_movies()

    tmdb_df["genre_list"] = tmdb_df["genres"].apply(parse_genres)

    preferred_genres = [g.lower() for g in preferred_genres]

    filtered = tmdb_df[
        tmdb_df["genre_list"].apply(
            lambda genres: any(
                g in genres for g in preferred_genres
            )
        )
    ]

    if filtered.empty:
        return recommend_popular_movies(top_k)

    filtered = filtered.sort_values(
        by=["vote_average", "popularity"],
        ascending=False
    ).head(top_k)

    return filtered[
        ["tmdb_id", "title", "vote_average", "popularity"]
    ].to_dict(orient="records")


# -------------------------------------------------
# Cold start decision logic
# -------------------------------------------------

def cold_start_recommender(
    user_has_history: bool,
    preferred_genres=None,
    top_k=10
):
    """
    Master cold-start handler.
    """

    if user_has_history:
        return []

    if preferred_genres:
        return recommend_by_genre(
            preferred_genres,
            top_k
        )

    return recommend_popular_movies(top_k)
