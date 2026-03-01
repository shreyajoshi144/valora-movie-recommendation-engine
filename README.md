
#  Valora —  Movie Recommendation System

A production-ready hybrid recommendation engine combining **content similarity, collaborative filtering, and matrix factorization**, built with a modular ML architecture and an offline evaluation framework.

Designed to handle:

* Cold start
* Data sparsity
* Popularity bias
* Evaluation leakage
* Strategy benchmarking

---

##  Problem Statement

Movie recommendation systems face:

* Sparse user-item interactions
* Cold-start users and items
* Popularity dominance
* Evaluation bias from improper splits

Valora addresses these using a dynamic hybrid architecture and a rigorous evaluation pipeline.

---

##  Core Recommendation Strategies

| Strategy      | Technique                      | Personalization | Use Case                   |
| ------------- | ------------------------------ | --------------- | -------------------------- |
| Content-Based | TF-IDF + Cosine Similarity     | Seed-driven     | New users                  |
| Collaborative | Item-Item CF                   | Behavior-driven | Known interactions         |
| SVD           | Matrix Factorization           | User-specific   | Latent preference modeling |
| Hybrid        | Weighted fusion (Content + CF) | Adaptive        | Balanced ranking           |
| Hybrid-SVD    | Content + CF + Latent factors  | Strongest       | Full personalization       |

---

## ⚙️ Architecture Overview

```
User Input (Movie + Strategy + User ID)
        │
        ▼
HybridRecommender Engine
        │
 ┌──────────────┬───────────────┬──────────────┐
 │              │               │              │
Content       CF Engine        SVD Model   Cold Start
TF-IDF        Cosine Sim       TruncatedSVD  Popularity/Genre
        │
Score Normalization
        │
Dynamic Fusion (α weighted)
        │
Popularity Penalty (optional)
        ▼
Ranked Recommendations
```

---

##  Engineering Highlights

###  1. Dynamic Hybrid Weighting

Alpha (content weight) adapts based on user interaction history:

* 0 ratings → content-heavy
* Moderate history → balanced
* Rich history → collaborative-heavy

Prevents overfitting and improves cold-start stability.

---

###  2. CF Coverage Gap Fix

If a seed movie is missing from the collaborative similarity matrix:

* System finds nearest content-similar movie
* Uses it as CF proxy
* Prevents empty or degenerate results

This ensures Hybrid ≠ Content.

---

###  3. Proper Offline Evaluation

Implements:

* Per-user chronological 80/20 split
* Train-only hybrid alpha computation
* User-centric seed selection for evaluation
* No test leakage

Metrics:

* Precision@K
* Recall@K
* Hit Rate
* RMSE (for SVD)

Avoids the common zero-precision bug caused by global seed usage.

---

###  4. Matrix Factorization

* sklearn TruncatedSVD
* Mean-centered ratings
* Full reconstructed prediction matrix
* Global latent ranking fallback for unknown users

No heavy Surprise dependency → deployable anywhere.

---

###  5. Popularity Bias Control

Optional ranking adjustment:
Allows niche content discovery.

---

###  6. Robust Poster Retrieval System

* Cached API calls
* Fallback via TMDB search endpoint
* Placeholder fallback
* No broken UI states

---

##  Evaluation Framework

The system supports multi-strategy benchmarking:

| Metric      | Purpose                              |
| ----------- | ------------------------------------ |
| Precision@K | Recommendation accuracy              |
| Recall@K    | Coverage of relevant items           |
| Hit Rate    | At least one relevant item retrieved |
| RMSE        | Rating prediction quality            |

Includes cross-strategy comparison for performance diagnostics.

---

##  Project Structure

```
valora-movie-recommender/
│
├── app.py
├── recommender/
│   ├── hybrid_engine.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── matrix_factorization.py
│   ├── evaluation.py
│   ├── cold_start.py
│   └── utils.py
│
├── data/
├── assets/
├── requirements.txt
└── README.md
```

---

##  Deployment Ready

* Streamlit Cloud
* AWS EC2
* Render
* Azure
* Railway

Lightweight dependencies, no GPU requirement.

---

##  Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Requests

---

##  This Demonstrates

* Applied machine learning engineering
* Hybrid model design
* Evaluation rigor
* Data leakage prevention
* Cold-start strategy design
* Ranking system thinking
* Model comparison & diagnostics

---
