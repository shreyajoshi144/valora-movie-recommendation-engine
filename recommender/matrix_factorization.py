"""
matrix_factorization.py — SVD-based collaborative filtering.
Uses sklearn TruncatedSVD (no Surprise dependency required).
Backward-compatible drop-in: exposes recommend_svd(user_id, top_k).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import logging

logger = logging.getLogger(__name__)


class SVDRecommender:
    """
    Latent-factor recommender using Truncated SVD on the user-item matrix.

    Algorithm
    ---------
    1. Fill NaN ratings with 0 (implicit absence).
    2. Decompose R ≈ U Σ Vᵀ with n_components latent factors.
    3. Reconstruct full predicted-ratings matrix R̂ = U Σ Vᵀ.
    4. For a given user, rank unrated items by predicted score.
    """

    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._user_item_matrix: pd.DataFrame | None = None
        self._predicted: np.ndarray | None = None
        self._trained = False

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────
    def fit(self, user_item_matrix: pd.DataFrame) -> "SVDRecommender":
        """
        Fit SVD on the provided user-item matrix.

        Parameters
        ----------
        user_item_matrix : pd.DataFrame
            Rows = users, columns = tmdb_ids, values = ratings (NaN allowed).
        """
        self._user_item_matrix = user_item_matrix.copy()
        R = user_item_matrix.fillna(0).values.astype(float)

        # Mean-centre each user's ratings (improves quality)
        self._user_means = np.true_divide(
            R.sum(axis=1),
            (R != 0).sum(axis=1).clip(min=1)
        )
        R_centred = R.copy()
        for i, mean in enumerate(self._user_means):
            R_centred[i, R[i] != 0] -= mean

        # Clamp n_components to valid range
        max_components = min(R_centred.shape) - 1
        n = min(self.n_components, max_components)
        self._svd.n_components = n

        U_sigma = self._svd.fit_transform(R_centred)          # (users × n)
        Vt = self._svd.components_                             # (n × items)
        self._predicted = U_sigma @ Vt                         # (users × items)

        # Add back user means
        for i, mean in enumerate(self._user_means):
            self._predicted[i] += mean

        self._trained = True
        logger.info(
            "SVDRecommender fitted: %d users × %d items, %d factors, "
            "explained variance ratio sum = %.3f",
            *R.shape,
            n,
            self._svd.explained_variance_ratio_.sum(),
        )
        return self

    # ─────────────────────────────────────────
    # Inference helpers
    # ─────────────────────────────────────────
    def _user_row_index(self, user_id) -> int | None:
        """Return positional row index for user_id, or None if unknown."""
        if self._user_item_matrix is None:
            return None
        idx_list = self._user_item_matrix.index.tolist()
        if user_id in idx_list:
            return idx_list.index(user_id)
        return None

    def recommend_svd(self, user_id, top_k: int = 10) -> list[dict]:
        """
        Recommend top_k movies for user_id using SVD predicted scores.

        Returns
        -------
        List of dicts: [{tmdb_id, svd_score}, ...]
        Scores are raw predicted ratings (not normalised here — caller normalises).
        """
        if not self._trained:
            raise RuntimeError("Call fit() before recommend_svd().")

        row_idx = self._user_row_index(user_id)
        tmdb_ids = self._user_item_matrix.columns.tolist()

        if row_idx is None:
            # Unknown user → return top globally predicted items
            logger.info("SVD: unknown user %s — using global popularity fallback", user_id)
            mean_scores = self._predicted.mean(axis=0)
            top_indices = np.argsort(mean_scores)[::-1][:top_k]
            return [
                {"tmdb_id": tmdb_ids[i], "svd_score": float(mean_scores[i])}
                for i in top_indices
            ]

        user_scores = self._predicted[row_idx]
        already_rated = self._user_item_matrix.iloc[row_idx]
        unrated_mask = already_rated.isna().values  # True = not yet rated

        # Rank unrated movies by predicted score
        candidate_scores = np.where(unrated_mask, user_scores, -np.inf)
        top_indices = np.argsort(candidate_scores)[::-1][:top_k]

        return [
            {"tmdb_id": tmdb_ids[i], "svd_score": float(user_scores[i])}
            for i in top_indices
        ]

    def predict_score(self, user_id, tmdb_id) -> float:
        """
        Predict a single user-movie score. Returns 0.0 if unknown.
        Used by evaluation (RMSE).
        """
        if not self._trained:
            return 0.0
        row_idx = self._user_row_index(user_id)
        if row_idx is None:
            return float(self._user_means.mean())
        tmdb_ids = self._user_item_matrix.columns.tolist()
        if tmdb_id not in tmdb_ids:
            return float(self._user_means[row_idx])
        col_idx = tmdb_ids.index(tmdb_id)
        return float(self._predicted[row_idx, col_idx])
