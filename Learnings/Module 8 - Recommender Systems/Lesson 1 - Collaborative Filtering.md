# Lesson 1: Collaborative Filtering 🤝

**Module 8: Recommender Systems | Lesson 1 of 4**

Master collaborative filtering - the foundation of recommendation systems!

---

## What are Recommender Systems?

**Suggest items users might like:**
- Netflix movies
- Amazon products
- Spotify songs
- YouTube videos

---

## Types of Recommendations

1. **Collaborative Filtering:** Based on user behavior
2. **Content-Based:** Based on item features
3. **Hybrid:** Combination of both

---

## 1. User-Based Collaborative Filtering 👥

**Idea:** Recommend items liked by similar users

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# User-item ratings matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# Users: 0-4, Items: 0-3

# Compute user similarity
user_sim = cosine_similarity(ratings)

print("User similarity matrix:")
print(user_sim)

# Predict rating for user 0, item 2 (currently 0)
user_id = 0
item_id = 2

# Find similar users who rated this item
rated_mask = ratings[:, item_id] > 0
similarities = user_sim[user_id] * rated_mask

# Weighted average
if similarities.sum() > 0:
    prediction = (similarities @ ratings[:, item_id]) / similarities.sum()
else:
    prediction = ratings[:, item_id].mean()

print(f"Predicted rating for user {user_id}, item {item_id}: {prediction:.2f}")
```

---

## 2. Item-Based Collaborative Filtering 📦

**Idea:** Recommend similar items to what user liked

```python
# Compute item similarity
item_sim = cosine_similarity(ratings.T)

print("Item similarity matrix:")
print(item_sim)

# Predict rating
item_id = 2
user_id = 0

# Items rated by this user
rated_mask = ratings[user_id] > 0
similarities = item_sim[item_id] * rated_mask

# Weighted average
if similarities.sum() > 0:
    prediction = (similarities @ ratings[user_id]) / similarities.sum()
else:
    prediction = 0

print(f"Predicted rating: {prediction:.2f}")
```

---

## 3. Surprise Library 📚

```python
from surprise import Dataset, Reader
from surprise import KNNBasic, KNNWithMeans
from surprise.model_selection import cross_validate

# Load dataset
data = Dataset.load_builtin('ml-100k')  # MovieLens 100K

# User-based CF
sim_options = {
    'name': 'cosine',
    'user_based': True
}

algo = KNNBasic(sim_options=sim_options)

# Cross-validation
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print(f"Mean RMSE: {results['test_rmse'].mean():.3f}")
print(f"Mean MAE: {results['test_mae'].mean():.3f}")
```

---

## 4. Complete Example 🎬

```python
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load MovieLens
data = Dataset.load_builtin('ml-100k')

# Train/test split
trainset, testset = train_test_split(data, test_size=0.25)

# Collaborative filtering with SVD
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

# Train
algo.fit(trainset)

# Test
predictions = algo.test(testset)

# Evaluate
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")

# Make predictions for user
user_id = '196'
item_id = '302'

prediction = algo.predict(user_id, item_id)
print(f"Predicted rating: {prediction.est:.2f}")

# Top-N recommendations
def get_top_n(predictions, n=10):
    """Get top-N recommendations for each user"""
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(predictions, n=10)

# Show recommendations for user
user_id = '196'
print(f"\nTop 10 recommendations for user {user_id}:")
for item_id, rating in top_n[user_id]:
    print(f"  Item {item_id}: {rating:.2f}")
```

---

## 5. Evaluation Metrics 📊

```python
# Precision@K
def precision_at_k(predictions, k=10, threshold=3.5):
    """Precision at K"""
    user_est_true = {}

    for uid, iid, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))

    precisions = {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Top-K
        top_k = user_ratings[:k]

        # Relevant (rating >= threshold)
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)

        precisions[uid] = n_rel_and_rec_k / k

    return sum(precisions.values()) / len(precisions)

# Calculate
precision = precision_at_k(predictions, k=10)
print(f"Precision@10: {precision:.3f}")

# Recall@K
def recall_at_k(predictions, k=10, threshold=3.5):
    """Recall at K"""
    user_est_true = {}

    for uid, iid, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))

    recalls = {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Top-K
        top_k = user_ratings[:k]
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel > 0 else 0

    return sum(recalls.values()) / len(recalls)

recall = recall_at_k(predictions, k=10)
print(f"Recall@10: {recall:.3f}")
```

---

## Quick Reference 📖

**User-based vs Item-based:**
- **User-based:** Find similar users
- **Item-based:** Find similar items (usually better)

**Similarity Metrics:**
- Cosine similarity
- Pearson correlation
- Euclidean distance

**Surprise Algorithms:**
```python
from surprise import KNNBasic, KNNWithMeans, SVD, NMF

# KNN
KNNBasic(k=40, sim_options={'name': 'cosine', 'user_based': True})

# Matrix Factorization
SVD(n_factors=100, n_epochs=20)
```

---

## Key Takeaways 💡

1. **Collaborative filtering** uses user behavior
2. **User-based** finds similar users
3. **Item-based** finds similar items (scales better)
4. **Cold start problem** for new users/items
5. **Sparsity** is main challenge
6. **Implicit feedback** (clicks, views) vs explicit (ratings)

---

**Next:** [Lesson 2 - Content-Based Filtering →](Lesson%202%20-%20Content-Based%20Filtering.md)
