
# 🎬 Valora - Hybrid Movie Recommendation Engine

A production-ready hybrid recommendation system that combines **content-based filtering, collaborative filtering, and matrix factorization** to deliver accurate and personalized movie recommendations.

Built with a modular machine learning architecture, offline evaluation framework, and an interactive Streamlit application, Valora addresses common recommendation system challenges such as **cold start, sparse interactions, popularity bias, and realistic model evaluation**.

---

## Tech Stack

**Python • Pandas • NumPy • Scikit-learn • SciPy • Streamlit • Requests**

---

## Problem Statement

Traditional recommendation systems often struggle with:

* Sparse user-item interactions
* Cold-start users and movies
* Popularity bias toward highly rated titles
* Evaluation leakage caused by improper train/test splits

Valora addresses these challenges through a hybrid recommendation architecture that combines multiple recommendation strategies with rigorous offline evaluation.

---

## Project Preview

<img width="1470" height="956" alt="Screenshot 2026-06-28 at 5 00 23 PM" src="https://github.com/user-attachments/assets/2f7ca966-4074-4577-bf35-697a2ad5212a" />

<img width="1470" height="956" alt="Screenshot 2026-06-28 at 5 01 11 PM" src="https://github.com/user-attachments/assets/111593aa-bbaf-4bb1-a514-5940b8d118e4" />

<img width="1470" height="956" alt="Screenshot 2026-06-28 at 5 01 52 PM" src="https://github.com/user-attachments/assets/d854654d-0203-44ad-aace-b6d8084811a7" />

<img width="1470" height="956" alt="Screenshot 2026-06-28 at 5 01 25 PM" src="https://github.com/user-attachments/assets/24ff7ba9-1031-49bf-90cd-b3349f8b89eb" />

---


## Live Demo : https://valora-movies.streamlit.app/

## Recommendation Strategies

| Strategy                | Technique                  | Personalization    | Best For                       |
| ----------------------- | -------------------------- | ------------------ | ------------------------------ |
| Content-Based           | TF-IDF + Cosine Similarity | Seed-based         | New users & similar movies     |
| Collaborative Filtering | Item-Item Similarity       | Behavior-based     | Existing users                 |
| Matrix Factorization    | Truncated SVD              | Personalized       | Learning latent preferences    |
| Hybrid                  | Content + Collaborative    | Adaptive           | Balanced recommendations       |
| Hybrid-SVD              | Content + CF + SVD         | Fully Personalized | Highest recommendation quality |
| Genre Explorer          | Genre Filtering + Ranking  | None               | Browsing by genre              |

---

# Project Architecture

<img width="2264" height="2107" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/6b6936c0-0e42-4888-8e0a-afca10ab1601" />


---

# Key Features

* Hybrid recommendation engine combining multiple algorithms
* Dynamic weighting based on user interaction history
* Cold-start handling for new users and unseen movies
* Popularity bias mitigation using configurable penalties
* Leakage-free offline evaluation pipeline
* Interactive Streamlit interface with movie posters
* Genre Explorer for Netflix-style browsing
* Cached poster retrieval with graceful fallback handling

---

# Engineering Highlights

### Dynamic Hybrid Weighting

Recommendation scores are adaptively blended based on the user's rating history. New users receive stronger content-based recommendations, while experienced users benefit more from collaborative filtering.

### Cold-Start Handling

When collaborative filtering cannot generate recommendations, the engine automatically falls back to content similarity, ensuring reliable recommendations without empty results.

### Matrix Factorization

Uses mean-centered ratings with TruncatedSVD to learn latent user preferences and improve recommendation quality beyond explicit similarity methods.

### Leakage-Free Evaluation

Implements a per-user chronological 80/20 train-test split with user-centric seed selection to avoid data leakage and produce realistic evaluation metrics.

### Popularity Bias Mitigation

Supports optional log-based popularity penalties to increase recommendation diversity and reduce over-representation of blockbuster movies.

### Production Utilities

Poster retrieval includes caching, retry logic, and multi-level fallbacks to ensure a smooth user experience.

---

# Evaluation Framework

The project includes an offline benchmarking framework for comparing multiple recommendation strategies.

| Metric      | Purpose                                                  |
| ----------- | -------------------------------------------------------- |
| Precision@K | Recommendation accuracy                                  |
| Recall@K    | Relevant item coverage                                   |
| Hit Rate    | Whether at least one relevant recommendation is returned |
| RMSE        | Rating prediction quality                                |

The evaluation pipeline enables objective comparison between Content-Based, Collaborative, Hybrid, and Matrix Factorization models.

---

# Project Structure

```text
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

# What This Project Demonstrates

* Machine Learning Engineering
* Recommendation System Design
* Hybrid Model Architecture
* Offline Evaluation Methodology
* Cold-Start Strategy Design
* Ranking System Optimization
* Production-Oriented ML Development
* Modular Python Architecture

---

# Future Improvements

* Neural Collaborative Filtering
* Implicit Feedback Models
* Time-Aware Recommendations
* FAISS-Based Approximate Nearest Neighbor Search
* Online A/B Testing Framework
* Model Monitoring & Drift Detection

---

# Run Locally

```bash
git clone https://github.com/shreyajoshi144/valora-movie-recommendation-engine.git

cd valora-movie-recommendation-engine

pip install -r requirements.txt

streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

Required datasets inside the `data/` directory:

* tmdb_5000_movies.csv
* movielens_movies.csv
* movielens_ratings.csv

---

If you found this project useful, consider ⭐ starring the repository.


