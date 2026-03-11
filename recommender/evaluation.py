"""
evaluation.py — Offline evaluation for Valora recommender system.

Backward-compatible: original precision_at_k, hit_rate, evaluate_recommender,
and summarize_results are preserved with identical signatures.

New additions
─────────────
• recall_at_k()               — fraction of relevant items retrieved
• rmse()                      — root mean squared error for rating predictions
• train_test_split_ratings()  — per-user chronological 80/20 split
• evaluate_full()             — proper held-out evaluation pipeline
• compare_strategies()        — multi-strategy comparison DataFrame

Chronological split design
──────────────────────────────────────
- Per user: sort by timestamp, take last 20% as test.
- Users with < min_ratings_per_user (default 5) go fully into train.
  This avoids unstable Recall@K from tiny test sets.
- split_idx = max(1, int(n * 0.8)) ensures at least 1 training rating.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# ORIGINAL METRICS — unchanged signatures
# ══════════════════════════════════════════════════════════════════════

def precision_at_k(recommended_items, relevant_items, k=10) -> float:
    """
    Precision@K: fraction of top-K recommendations that are relevant.

    Parameters
    ----------
    recommended_items : list  — ordered list of recommended item IDs
    relevant_items    : list  — items the user actually liked
    k                 : int   — cut-off rank
    """
    if k <= 0:
        return 0.0
    recommended_items = list(recommended_items)[:k]
    hits = sum(1 for item in recommended_items if item in set(relevant_items))
    return hits / k


def hit_rate(recommended_items, relevant_items) -> int:
    """
    Hit Rate: 1 if any recommended item is in relevant_items, else 0.
    """
    relevant_set = set(relevant_items)
    return int(any(item in relevant_set for item in recommended_items))


def evaluate_recommender(recommendation_func, test_data, top_k=10) -> pd.DataFrame:
    """
    Evaluate recommender across multiple users.

    Parameters
    ----------
    recommendation_func : callable(user_id) → list[tmdb_id]
    test_data           : dict  {user_id: [relevant_tmdb_ids]}
    top_k               : int

    Returns
    -------
    pd.DataFrame  — per-user Precision@K and Hit Rate.
    """
    results = []
    for user_id, relevant_items in test_data.items():
        recommended_items = recommendation_func(user_id)
        results.append({
            "user_id": user_id,
            f"precision_at_{top_k}": precision_at_k(recommended_items, relevant_items, k=top_k),
            "hit_rate": hit_rate(recommended_items, relevant_items),
        })
    return pd.DataFrame(results)


def summarize_results(results_df: pd.DataFrame) -> dict:
    """Returns mean Precision@K and mean Hit Rate."""
    return {
        "mean_precision": results_df.filter(like="precision").mean().values[0],
        "mean_hit_rate":  results_df["hit_rate"].mean(),
    }


# ══════════════════════════════════════════════════════════════════════
# NEW METRICS
# ══════════════════════════════════════════════════════════════════════

def recall_at_k(recommended_items, relevant_items, k=10) -> float:
    """
    Recall@K: fraction of relevant items captured in top-K recommendations.
    Returns 0 if relevant_items is empty.
    """
    if not relevant_items:
        return 0.0
    recommended_top_k = list(recommended_items)[:k]
    relevant_set = set(relevant_items)
    hits = sum(1 for item in recommended_top_k if item in relevant_set)
    return hits / len(relevant_items)


def rmse(predictions: list[tuple[float, float]]) -> float:
    """
    Root Mean Squared Error from [(predicted, actual), ...] pairs.
    Returns NaN if list is empty.
    """
    if not predictions:
        return float("nan")
    errors = [(p - a) ** 2 for p, a in predictions]
    return float(np.sqrt(np.mean(errors)))


# ══════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT (chronological, per-user)
# ══════════════════════════════════════════════════════════════════════

def train_test_split_ratings(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.20,
    min_ratings_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user chronological 80/20 split.

    Design
    ─────────────────
    • Sort each user's ratings by timestamp (if present) or stable row order.
    • split_idx = max(1, int(n * 0.8)) — guarantees ≥ 1 rating in train.
    • Users with < min_ratings_per_user ratings go entirely into train to
      avoid unstable single-item test sets.

    Parameters
    ----------
    ratings_df            : pd.DataFrame  — must have userId, tmdb_id, rating.
                            Optionally: timestamp (used for chronological sort).
    test_ratio            : float         — fraction held out as test.
    min_ratings_per_user  : int           — minimum ratings to be split.

    Returns
    -------
    (train_df, test_df)
    """
    has_timestamp = "timestamp" in ratings_df.columns

    train_parts, test_parts = [], []

    for user_id, group in ratings_df.groupby("userId", sort=False):
        # Chronological sort if timestamp available, else preserve original order
        if has_timestamp:
            group = group.sort_values("timestamp", ascending=True)

        n = len(group)

        # Cold-start users: keep all data in train, none in test
        if n < min_ratings_per_user:
            train_parts.append(group)
            continue

        # Chronological cut: first 80% train, last 20% test
        split_idx = max(1, int(n * (1.0 - test_ratio)))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = (pd.concat(train_parts, ignore_index=True)
                if train_parts else pd.DataFrame(columns=ratings_df.columns))
    test_df  = (pd.concat(test_parts,  ignore_index=True)
                if test_parts  else pd.DataFrame(columns=ratings_df.columns))

    logger.info(
        "Train/test split | train=%d | test=%d | test_ratio=%.0f%%",
        len(train_df), len(test_df), test_ratio * 100,
    )
    return train_df, test_df


# ══════════════════════════════════════════════════════════════════════
# COMPREHENSIVE OFFLINE EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_full(
    recommendation_func,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    relevance_threshold: float = 4.0,
    svd_model=None,
    max_users: int = 200,
    engine=None,
    strategy: str = "Hybrid",
) -> dict:
    """
    Full offline evaluation pipeline (fixed methodology).

    FIX: Previously, one fixed seed was used for ALL users → 0 precision.
    Now uses engine.recommend_for_user() which picks each user's OWN top-rated
    training movies as seeds. This is the correct approach for seed-based recs.

    When engine=None, falls back to recommendation_func(user_id) legacy mode.
    """
    train_df, test_df = train_test_split_ratings(ratings_df)

    # Inject train-only data for alpha (no leakage)
    if engine is not None:
        engine.set_training_ratings(train_df)

    test_relevant = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("userId")["tmdb_id"]
        .apply(list)
        .to_dict()
    )

    eval_users = [u for u, items in test_relevant.items() if items]
    if max_users and len(eval_users) > max_users:
        rng = np.random.default_rng(42)
        eval_users = list(rng.choice(eval_users, size=max_users, replace=False))

    precisions, recalls, hits = [], [], []
    rmse_pairs = []

    for user_id in eval_users:
        relevant = test_relevant[user_id]
        try:
            if engine is not None:
                # FIX: user-centric — each user gets recs from their own seeds
                recommended = engine.recommend_for_user(
                    user_id, strategy=strategy, top_k=top_k
                )
            else:
                recommended = recommendation_func(user_id)
        except Exception as exc:
            logger.warning("Recommendation failed for user %s: %s", user_id, exc)
            continue

        precisions.append(precision_at_k(recommended, relevant, k=top_k))
        recalls.append(recall_at_k(recommended, relevant, k=top_k))
        hits.append(hit_rate(recommended, relevant))

        if svd_model is not None:
            for _, row in test_df[test_df["userId"] == user_id].iterrows():
                pred = svd_model.predict_score(user_id, row["tmdb_id"])
                rmse_pairs.append((pred, row["rating"]))

    result = {
        "top_k":             top_k,
        "n_users_evaluated": len(precisions),
        "precision_at_k":    float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k":       float(np.mean(recalls))    if recalls    else 0.0,
        "hit_rate":          float(np.mean(hits))        if hits       else 0.0,
    }
    if svd_model is not None:
        result["rmse"] = rmse(rmse_pairs)

    logger.info("evaluate_full complete: %s", result)
    return result


# ══════════════════════════════════════════════════════════════════════
# STRATEGY COMPARISON
# ══════════════════════════════════════════════════════════════════════

def compare_strategies(
    strategy_funcs: dict,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    svd_model=None,
    max_users: int = 100,
    engine=None,
) -> pd.DataFrame:
    """
    Evaluate multiple strategies and return a comparison table.

    When engine is provided, uses user-centric seeding (recommend_for_user)
    for accurate Precision@K and Recall@K — fixes the all-zeros bug.
    """
    rows = []
    for name, func in strategy_funcs.items():
        logger.info("Evaluating strategy: %s", name)
        metrics = evaluate_full(
            func,
            ratings_df,
            top_k=top_k,
            svd_model=(svd_model if name in {"SVD", "Hybrid-SVD"} else None),
            max_users=max_users,
            engine=engine,
            strategy=name,
        )
        metrics["strategy"] = name
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("strategy")


def evaluate_user_holdout(
    engine,
    ratings_df: pd.DataFrame,
    user_id: int,
    strategy: str,
    top_k: int = 10,
    like_threshold: float = 4.0,
    n_seeds: int = 3,
):
    """Evaluate a single user using the same holdout protocol as offline evaluation.

    - Chronological per-user 80/20 split (via train_test_split_ratings)
    - Relevant items are in the user's *test* set with rating >= like_threshold
    - Recommendations are generated from the user's *train* history via engine.recommend_for_user

    Returns dict with: precision_at_k, recall_at_k, hit, n_test_relevant, n_train, rmse (optional)
    """
    train_df, test_df = train_test_split_ratings(ratings_df, test_ratio=0.20)

    user_train = train_df[train_df["userId"] == user_id]
    user_test = test_df[test_df["userId"] == user_id]

    relevant_test = (
        user_test[user_test["rating"] >= like_threshold]["tmdb_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    if len(relevant_test) == 0:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "hit": None,
            "n_test_relevant": 0,
            "n_train": int(len(user_train)),
            "rmse": None,
            "note": "Not enough positive items in the user's test split to evaluate.",
        }

    # Train engine on the train split (needed for SVD and for seed selection)
    engine.set_training_ratings(user_train)

    recs = engine.recommend_for_user(
        user_id=user_id,
        strategy=strategy,
        top_k=top_k,
        n_seeds=n_seeds,
    )

    prec = precision_at_k(recs, relevant_test, top_k)
    rec = recall_at_k(recs, relevant_test, top_k)
    hit = hit_rate(recs, relevant_test)

    rmse_val = None
    if hasattr(engine, "svd_model") and getattr(engine, "svd_model", None) is not None and strategy in ("SVD", "Hybrid-SVD"):
        try:
            rmse_val = rmse(engine.svd_model, user_test[["userId", "tmdb_id", "rating"]].dropna())
        except Exception:
            rmse_val = None

    return {
        "precision_at_k": float(prec) if prec is not None else None,
        "recall_at_k": float(rec) if rec is not None else None,
        "hit": bool(hit) if hit is not None else None,
        "n_test_relevant": int(len(relevant_test)),
        "n_train": int(len(user_train)),
        "rmse": rmse_val,
        "note": None,
    }
