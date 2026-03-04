import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recommender.utils import (
    load_tmdb_movies,
    load_movielens_movies,
    load_movielens_ratings,
    map_movielens_to_tmdb,
    build_tmdb_ratings_matrix,
    create_user_item_matrix
)

# -------------------------------------------------
# Load & prepare collaborative filtering data
# -------------------------------------------------

def prepare_collaborative_data():
    """
    End-to-end preparation for collaborative filtering.
    Returns:
        - user_item_matrix
        - tmdb_movies_df
    """

    # Load datasets
    tmdb_df = load_tmdb_movies()
    ml_movies_df = load_movielens_movies()
    ml_ratings_df = load_movielens_ratings()

    # Map MovieLens -> TMDB
    mapping_df = map_movielens_to_tmdb(tmdb_df, ml_movies_df)

    # Build TMDB-aligned ratings
    tmdb_ratings_df = build_tmdb_ratings_matrix(
        ml_ratings_df,
        mapping_df
    )

    # User-item matrix
    user_item_matrix = create_user_item_matrix(tmdb_ratings_df)

    return user_item_matrix, tmdb_df


# -------------------------------------------------
# Item-based Collaborative Filtering
# -------------------------------------------------

def compute_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Computes item-item cosine similarity matrix.
    """

    # Transpose: items as rows
    item_matrix = user_item_matrix.T.fillna(0)

    similarity = cosine_similarity(item_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=item_matrix.index,
        columns=item_matrix.index
    )

    return similarity_df


def recommend_similar_movies_cf(
    tmdb_id: int,
    similarity_df: pd.DataFrame,
    tmdb_movies_df: pd.DataFrame,
    top_k: int = 10
):
    """
    Recommend movies using item-based collaborative filtering.
    """

    if tmdb_id not in similarity_df.index:
        return []

    scores = similarity_df.loc[tmdb_id]

    scores = scores.sort_values(ascending=False)

    scores = scores.iloc[1:top_k + 1]  # remove self

    recommendations = (
        tmdb_movies_df[
            tmdb_movies_df["tmdb_id"].isin(scores.index)
        ][["tmdb_id", "title", "vote_average", "popularity"]]
        .assign(similarity_score=lambda x: x["tmdb_id"].map(scores))
        .sort_values(by="similarity_score", ascending=False)
    )

    return recommendations.to_dict(orient="records")


# -------------------------------------------------
# High-level CF pipeline
# -------------------------------------------------

def collaborative_recommender(
    seed_tmdb_id: int,
    top_k: int = 10
):
    """
    Full collaborative filtering pipeline.
    """

    user_item_matrix, tmdb_movies_df = prepare_collaborative_data()

    similarity_df = compute_item_similarity(user_item_matrix)

    return recommend_similar_movies_cf(
        seed_tmdb_id,
        similarity_df,
        tmdb_movies_df,
        top_k
    )
