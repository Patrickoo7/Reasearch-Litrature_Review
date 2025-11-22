# Lesson 2: Content-Based Filtering 🎨

**Module 8: Recommender Systems | Lesson 2 of 4**

Master content-based recommendations using item features!

---

## Idea

**Recommend items similar to what user liked in the past**
- Based on item features
- No need for other users' data

---

## 1. TF-IDF for Text Features 📝

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie descriptions
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Toy Story', 'Finding Nemo'],
    'description': [
        'Sci-fi action computer hacker reality',
        'Sci-fi thriller dream heist',
        'Animation toys adventure children',
        'Animation fish ocean adventure'
    ]
})

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Similarity matrix:")
print(pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title']))

# Recommend similar movies
def get_recommendations(title, cosine_sim, movies, n=3):
    idx = movies[movies['title'] == title].index[0]

    # Similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices (skip first - itself)
    movie_indices = [i[0] for i in sim_scores[1:n+1]]

    return movies.iloc[movie_indices][['title', 'description']]

# Get recommendations
recs = get_recommendations('The Matrix', cosine_sim, movies, n=2)
print("\nRecommendations for 'The Matrix':")
print(recs)
```

---

## 2. Using Multiple Features 🎯

```python
from sklearn.preprocessing import MultiLabelBinarizer

# Movies with genres
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Toy Story', 'Finding Nemo'],
    'genres': [['Sci-Fi', 'Action'], ['Sci-Fi', 'Thriller'],
              ['Animation', 'Comedy'], ['Animation', 'Adventure']],
    'year': [1999, 2010, 1995, 2003],
    'rating': [8.7, 8.8, 8.3, 8.1]
})

# One-hot encode genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies['genres'])

# Normalize year and rating
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(movies[['year', 'rating']])

# Combine features
features = np.hstack([genres_encoded, numerical_features])

# Compute similarity
cosine_sim = cosine_similarity(features)

print(pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title']))
```

---

## 3. User Profile Building 👤

```python
# User history
user_history = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2],
    'movie_id': [0, 1, 2, 1, 3],  # Indices
    'rating': [5, 4, 3, 5, 4]
})

def build_user_profile(user_id, user_history, tfidf_matrix):
    """Build user profile from rated movies"""
    user_movies = user_history[user_history['user_id'] == user_id]

    # Weighted average of movie features
    weights = user_movies['rating'].values
    movie_indices = user_movies['movie_id'].values

    # Get movie features
    movie_features = tfidf_matrix[movie_indices].toarray()

    # Weighted average
    user_profile = (movie_features.T @ weights) / weights.sum()

    return user_profile

# Build profile for user 1
user_profile = build_user_profile(1, user_history, tfidf_matrix)

# Recommend movies
def recommend_for_user(user_profile, tfidf_matrix, movies, n=3):
    # Compute similarity between user profile and all movies
    similarities = cosine_similarity([user_profile], tfidf_matrix)[0]

    # Sort
    movie_indices = np.argsort(similarities)[::-1][:n]

    return movies.iloc[movie_indices][['title']]

recs = recommend_for_user(user_profile, tfidf_matrix, movies, n=2)
print("\nRecommendations:")
print(recs)
```

---

## Quick Reference 📖

**Content-Based Steps:**
1. Extract item features
2. Compute item similarity
3. Build user profile (weighted features)
4. Recommend similar items

**Feature Types:**
- **Text:** TF-IDF
- **Categories:** One-hot encoding
- **Numerical:** Normalization
- **Images:** CNN features

---

## Key Takeaways 💡

1. **Content-based** uses item features only
2. **No cold start** for new users
3. **Limited diversity** (filter bubble)
4. **Needs good features**
5. **Works for new items**
6. **Combine with collaborative filtering** (hybrid)

---

**Next:** [Lesson 3 - Matrix Factorization →](Lesson%203%20-%20Matrix%20Factorization.md)
