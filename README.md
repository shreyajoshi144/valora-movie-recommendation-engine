
#  Valora - Movie Recommendation Engine

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
## Project Pipeline

<img width="3695" height="432" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/aeff4d39-60ae-4915-bb8b-f5dfab2afd54" />

##  Architecture Overview

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

1. Dynamic Hybrid Weighting
   
Adaptive blending of content and collaborative signals based on user rating history.
Cold-start users receive higher content weighting, while experienced users leverage behavioral signals.

3. CF Coverage Handling

Implements content-similarity proxy when a seed movie is absent from the CF matrix.
Prevents empty outputs and ensures meaningful hybrid differentiation.

3. Leakage-Free Offline Evaluation

Per-user chronological 80/20 split with user-centric seed selection.
Evaluated using Precision@K, Recall@K, Hit Rate, and RMSE to ensure realistic performance measurement.

4. Matrix Factorization (SVD)

Mean-centered ratings with TruncatedSVD to capture latent user–item factors.
Includes personalized ranking and global fallback for unknown users.

5. Popularity Bias Mitigation

Optional log-based popularity penalty to reduce dominance of highly popular titles and improve ranking diversity.

6. Robust Production Utilities

Poster retrieval with caching and multi-level fallback to prevent UI failures and broken states.

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

##  Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Requests

---

##  This Project Demonstrates

* Applied machine learning engineering
* Hybrid model design
* Evaluation rigor
* Data leakage prevention
* Cold-start strategy design
* Ranking system thinking
* Model comparison & diagnostics

---
## Future Improvements

* Neural collaborative filtering
* Implicit feedback modeling
* Time-aware recommendation (temporal decay)
* ANN search (FAISS) for scalability
* Online A/B testing framework

---

## Run Locally
git clone https://github.com/shreyajoshi144/valora-movie-recommendation-engine.git    
cd valora-movie-recommendation-engine    
pip install -r requirements.txt    
streamlit run app.py       

App will launch at:
http://localhost:8501

Ensure the following files exist inside the data/ folder:

* tmdb_5000_movies.csv
* movielens_movies.csv
* movielens_ratings.csv
---
If you found this project useful, consider starring the repository.
