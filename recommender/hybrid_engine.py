"""
hybrid_engine.py — Production Hybrid Recommender for Valora.

"""
import pandas as pd
import numpy as np
import logging

from recommender.collaborative import (
    recommend_similar_movies_cf,
    prepare_collaborative_data,
    compute_item_similarity,
)
from recommender.cold_start import cold_start_recommender
from recommender.content_based import (
    content_based_recommender,
    cosine_sim,
    tmdb_id_to_index,
    tmdb_df as _content_tmdb_df,
)
from recommender.matrix_factorization import SVDRecommender
from recommender.utils import (
    load_tmdb_movies,
    normalize_scores,
    apply_popularity_penalty,
    create_user_item_matrix,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Score dict helpers
# ──────────────────────────────────────────────────────────────────────

def _recs_to_score_dict(recs: list[dict], score_key: str) -> dict:
    return {r["tmdb_id"]: float(r.get(score_key, 0.0))
            for r in recs if "tmdb_id" in r}


def _svd_recs_to_score_dict(recs: list[dict]) -> dict:
    return {r["tmdb_id"]: float(r.get("svd_score", 0.0))
            for r in recs if "tmdb_id" in r}


# ──────────────────────────────────────────────────────────────────────
# Dynamic alpha (training data only — no leakage)
# ──────────────────────────────────────────────────────────────────────

def _compute_alpha(user_id, ratings_df) -> float:
    """
    Alpha = content weight.  Uses TRAINING ratings only.
    0 ratings  → 0.80   1-4 ratings → 0.60   5+ ratings → 0.40
    """
    if ratings_df is None or user_id is None:
        return 0.80
    n = len(ratings_df[ratings_df["userId"] == user_id])
    if n == 0:
        return 0.80
    if n < 5:
        return 0.60
    return 0.40


# ──────────────────────────────────────────────────────────────────────
# Score fusion
# ──────────────────────────────────────────────────────────────────────

def _fuse_scores(content_norm, collab_norm, alpha, svd_norm=None, use_svd=False):
    """Weighted fusion of normalised score dicts."""
    all_ids = set(content_norm) | set(collab_norm)
    if use_svd and svd_norm:
        all_ids |= set(svd_norm)
    result = {}
    for mid in all_ids:
        c  = content_norm.get(mid, 0.0)
        cf = collab_norm.get(mid, 0.0)
        if use_svd and svd_norm:
            svd  = svd_norm.get(mid, 0.0)
            beta = (1.0 - alpha) / 2.0
            result[mid] = alpha * c + beta * cf + beta * svd
        else:
            result[mid] = alpha * c + (1.0 - alpha) * cf
    return result



def _find_cf_proxy(seed_tmdb_id: int, cf_index: set, search_depth: int = 50):
    """
    When seed isn't in CF matrix, walk down the content-similarity list
    to find the nearest movie that IS in the CF matrix.
    Returns proxy tmdb_id or None.
    """
    if seed_tmdb_id not in tmdb_id_to_index:
        return None
    seed_idx     = tmdb_id_to_index[seed_tmdb_id]
    sim_scores   = list(enumerate(cosine_sim[seed_idx]))
    sim_sorted   = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    for idx, score in sim_sorted[1:search_depth + 1]:
        candidate_id = int(_content_tmdb_df.iloc[idx]["tmdb_id"])
        if candidate_id in cf_index:
            logger.info("CF proxy: seed=%s → proxy=%s (sim=%.3f)", seed_tmdb_id, candidate_id, score)
            return candidate_id
    return None


# ──────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────

class HybridRecommender:
    def __init__(self, top_k: int = 10, n_svd_factors: int = 150):
        self.top_k   = top_k
        self.tmdb_df = load_tmdb_movies()

        # Collaborative
        self.user_item_matrix, _ = prepare_collaborative_data()
        self.cf_similarity_df    = compute_item_similarity(self.user_item_matrix)
        self._cf_index           = set(self.cf_similarity_df.index.tolist())

        # SVD
        self.svd_model = SVDRecommender(n_components=n_svd_factors)
        self.svd_model.fit(self.user_item_matrix)

        # FIX #3: global SVD mean — shape (n_items,)
        self._svd_global_mean = self.svd_model._predicted.mean(axis=0)
        self._svd_tmdb_ids    = self.user_item_matrix.columns.tolist()

        # Training ratings for alpha
        self._alpha_ratings = None

    def set_mapped_ratings(self, mapped_ratings):
        """Inject mapped ratings for alpha (production)."""
        self._alpha_ratings = mapped_ratings

    def set_training_ratings(self, ratings_df: pd.DataFrame | None):
        """Inject a training split (used for evaluation).
        This updates:
        - alpha calibration (cold-start / dynamic blending)
        - CF similarity matrix (item-item)
        - SVD model (to avoid test leakage during offline evaluation)
        """
        self._alpha_ratings = ratings_df

        if ratings_df is None or len(ratings_df) == 0:
            return

        required = {"userId", "tmdb_id", "rating"}
        if not required.issubset(set(ratings_df.columns)):
            return

        uim = create_user_item_matrix(
            ratings_df[["userId", "tmdb_id", "rating"]].dropna()
        )

        # Update CF similarity (train-split)
        self.user_item_matrix = uim
        self.cf_similarity_df = compute_item_similarity(self.user_item_matrix)
        self._cf_index = set(self.cf_similarity_df.index.tolist())

        # Refit SVD (train-split)
        try:
            self.svd_model.fit(self.user_item_matrix)
            self._svd_tmdb_ids    = self.user_item_matrix.columns.tolist()
            self._svd_global_mean = self.svd_model._predicted.mean(axis=0)
        except Exception:
            pass


    def recommend(
        self,
        user_id=None,
        seed_movie_id=None,
        strategy="Hybrid",
        top_k=None,
        penalise_popularity=False,
    ):
        k = top_k if top_k else self.top_k

        if strategy == "Content-Based" and seed_movie_id:
            fetch_k = k * 8 if penalise_popularity else k
            recs = content_based_recommender(seed_movie_id, top_k=fetch_k)
            return self._maybe_penalise(recs, "similarity_score", penalise_popularity, top_k=k)

        # FIX #1: Collaborative with CF-proxy fallback
        if strategy == "Collaborative" and seed_movie_id:
            return self._collaborative_with_fallback(seed_movie_id, k, penalise_popularity)

        if strategy == "SVD":
            return self._svd_recommend(user_id, k, penalise_popularity)

        if strategy == "Hybrid" and seed_movie_id:
            return self._hybrid_recommend(user_id, seed_movie_id, k, use_svd=False,
                                          penalise_popularity=penalise_popularity)

        if strategy == "Hybrid-SVD" and seed_movie_id:
            return self._hybrid_recommend(user_id, seed_movie_id, k, use_svd=True,
                                          penalise_popularity=penalise_popularity)

        return cold_start_recommender(user_has_history=False, top_k=k)

    # FIX #4: User-centric recommendation for offline evaluation
    def recommend_for_user(self, user_id, strategy="Hybrid", top_k=10, n_seeds=3):
        if self._alpha_ratings is None:
            return []

        user_ratings = (
            self._alpha_ratings[self._alpha_ratings["userId"] == user_id]
            .sort_values("rating", ascending=False)
            .head(n_seeds)
        )
        if user_ratings.empty:
            return []

        seed_ids  = user_ratings["tmdb_id"].tolist()
        all_scores: dict = {}

        for seed_id in seed_ids:
            try:
                recs = self.recommend(
                    user_id=user_id,
                    seed_movie_id=int(seed_id),
                    strategy=strategy,
                    top_k=top_k * 3,
                )
                for r in recs:
                    mid   = r["tmdb_id"]
                    score = r.get("similarity_score", 0.0)
                    all_scores[mid] = max(all_scores.get(mid, 0.0), score)
            except Exception as e:
                logger.debug("Seed %s failed: %s", seed_id, e)

        for s in seed_ids:
            all_scores.pop(int(s), None)

        return sorted(all_scores, key=lambda x: all_scores[x], reverse=True)[:top_k]

    # ── Internal: Collaborative with fallback ─────────────────────────

    def _collaborative_with_fallback(self, seed_movie_id, k, penalise_popularity):
        fetch_k = k * 8 if penalise_popularity else k
        recs = recommend_similar_movies_cf(
            seed_movie_id, self.cf_similarity_df, self.tmdb_df, top_k=fetch_k
        )
        if not recs:
            proxy_id = _find_cf_proxy(seed_movie_id, self._cf_index)
            if proxy_id:
                recs = recommend_similar_movies_cf(
                    proxy_id, self.cf_similarity_df, self.tmdb_df, top_k=fetch_k
                )
        if not recs:
            recs = content_based_recommender(seed_movie_id, top_k=fetch_k)
        return self._maybe_penalise(recs, "similarity_score", penalise_popularity, top_k=k)

    # ── Internal: SVD-only ────────────────────────────────────────────

    def _svd_recommend(self, user_id, k, penalise_popularity):
        if user_id is None:
            return cold_start_recommender(user_has_history=False, top_k=k)
        svd_recs   = self.svd_model.recommend_svd(user_id, top_k=k * 5)
        svd_scores = normalize_scores(_svd_recs_to_score_dict(svd_recs))
        if penalise_popularity:
            svd_scores = normalize_scores(apply_popularity_penalty(svd_scores, self.tmdb_df))
        top_ids = sorted(svd_scores, key=lambda x: svd_scores[x], reverse=True)[:k]
        return self._enrich(top_ids, svd_scores)

    # ── Internal: global SVD signal  ─────────────────────────

    def _global_svd_scores(self, top_n):
        """Mean predicted rating across all users — population-level latent signal."""
        top_idx = np.argsort(self._svd_global_mean)[::-1][:top_n]
        return {self._svd_tmdb_ids[i]: float(self._svd_global_mean[i])
                for i in top_idx}

    # ── Internal: hybrid fusion  ─────────────────

    def _hybrid_recommend(self, user_id, seed_movie_id, k, use_svd, penalise_popularity):
        alpha       = _compute_alpha(user_id, self._alpha_ratings)
        candidate_k = min(max(k * 8, 80), 200) if penalise_popularity else min(max(k * 5, 50), 100)

        # Content
        content_recs = content_based_recommender(seed_movie_id, top_k=candidate_k)
        content_norm = normalize_scores(
            _recs_to_score_dict(content_recs, "similarity_score")
        )

        # CF with proxy
        collab_recs = recommend_similar_movies_cf(
            seed_movie_id, self.cf_similarity_df, self.tmdb_df, top_k=candidate_k
        )
        if not collab_recs:
            proxy_id = _find_cf_proxy(seed_movie_id, self._cf_index)
            if proxy_id:
                collab_recs = recommend_similar_movies_cf(
                    proxy_id, self.cf_similarity_df, self.tmdb_df, top_k=candidate_k
                )
        collab_norm = normalize_scores(
            _recs_to_score_dict(collab_recs, "similarity_score")
        )

        # SVD signal (personalised or global)
        svd_norm = {}
        if use_svd:
            if user_id is not None:
                svd_recs = self.svd_model.recommend_svd(user_id, top_k=candidate_k)
                svd_norm = normalize_scores(_svd_recs_to_score_dict(svd_recs))
            else:
                # Global latent popularity — differs from item-item CF
                svd_norm = normalize_scores(self._global_svd_scores(top_n=candidate_k))

        final_scores = _fuse_scores(content_norm, collab_norm, alpha,
                                    svd_norm=svd_norm, use_svd=use_svd)
        final_scores.pop(seed_movie_id, None)

        if penalise_popularity:
            final_scores = normalize_scores(
                apply_popularity_penalty(final_scores, self.tmdb_df)
            )

        top_ids = sorted(final_scores, key=lambda x: final_scores[x], reverse=True)[:k]
        return self._enrich(top_ids, final_scores)

    # ── Helpers ───────────────────────────────────────────────────────

    def _enrich(self, tmdb_ids, score_dict):
        meta    = self.tmdb_df.set_index("tmdb_id")
        results = []
        for mid in tmdb_ids:
            if mid not in meta.index:
                continue
            row   = meta.loc[mid]
            score = round(float(score_dict.get(mid, 0.0)), 4)
            results.append({
                "tmdb_id":          int(mid),
                "title":            row["title"],
                "vote_average":     float(row.get("vote_average", 0.0)),
                "popularity":       float(row.get("popularity", 0.0)),
                "similarity_score": score,
                "svd_score":        score,
            })
        return results

    def _maybe_penalise(self, recs, score_key, do_penalise, top_k=None):
        if not do_penalise:
            # Still trim to top_k if an expanded pool was passed
            return recs[:top_k] if top_k else recs
        score_dict = {r["tmdb_id"]: r.get(score_key, 0.0) for r in recs}
        penalised  = normalize_scores(apply_popularity_penalty(score_dict, self.tmdb_df))
        for r in recs:
            v = round(penalised.get(r["tmdb_id"], 0.0), 4)
            r[score_key]          = v
            r["similarity_score"] = v
        sorted_recs = sorted(recs, key=lambda x: x.get(score_key, 0.0), reverse=True)
        return sorted_recs[:top_k] if top_k else sorted_recs