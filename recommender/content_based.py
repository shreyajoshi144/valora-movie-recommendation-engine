import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender.utils import load_tmdb_movies

# Load TMDB movies
tmdb_df = load_tmdb_movies()

# Precompute content similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
# Combine overview + genres as text features
tmdb_df['content'] = tmdb_df['overview'].fillna('') + ' ' + tmdb_df['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(tmdb_df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping from tmdb_id to index
tmdb_id_to_index = pd.Series(tmdb_df.index, index=tmdb_df['tmdb_id']).to_dict()


def content_based_recommender(seed_movie_id, top_k=6):
    """
    Returns top_k content-based recommendations for a given seed_movie_id.
    """
    if seed_movie_id not in tmdb_id_to_index:
        # fallback to first movie if seed not found
        seed_movie_id = tmdb_df['tmdb_id'].iloc[0]

    idx = tmdb_id_to_index[seed_movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sim_scores[1:top_k+1]:  # skip the movie itself
        movie = tmdb_df.iloc[i]
        recommendations.append({
            "tmdb_id": movie['tmdb_id'],
            "title": movie['title'],
            "poster_path": movie.get('poster_path', None),
            "similarity_score": score
        })

    return recommendations
