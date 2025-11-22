# Lesson 4: Neural Recommenders 🧠

**Module 8: Recommender Systems | Lesson 4 of 4**

Master neural networks for recommendations - SOTA approach!

---

## Why Neural Networks?

**Traditional CF limitations:**
- Linear interactions only
- Manual feature engineering
- Fixed embeddings

**Neural networks:**
- Non-linear interactions
- Automatic feature learning
- End-to-end training

---

## 1. Embedding Layer 🎯

```python
import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Dot product
        rating = (user_emb * item_emb).sum(dim=1)

        return rating

# Example
n_users, n_items = 100, 50
model = EmbeddingModel(n_users, n_items, embedding_dim=32)

# Predict
user_ids = torch.LongTensor([0, 1, 2])
item_ids = torch.LongTensor([5, 10, 15])

ratings = model(user_ids, item_ids)
print(f"Predicted ratings: {ratings}")
```

---

## 2. Neural Collaborative Filtering (NCF) ⭐

```python
class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        layers = []
        input_size = embedding_dim * 2

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)

        # MLP
        rating = self.mlp(x).squeeze()

        return rating

# Create model
model = NCF(n_users=1000, n_items=500, embedding_dim=64)
print(model)
```

---

## 3. Training 🎯

```python
# Prepare data
train_users = torch.LongTensor([0, 1, 2, 3])
train_items = torch.LongTensor([5, 10, 15, 20])
train_ratings = torch.FloatTensor([4.5, 3.0, 5.0, 2.5])

# Model
model = NCF(n_users=100, n_items=50)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 100
for epoch in range(epochs):
    model.train()

    # Forward
    predictions = model(train_users, train_items)
    loss = criterion(predictions, train_ratings)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Predict
model.eval()
with torch.no_grad():
    test_users = torch.LongTensor([0, 1])
    test_items = torch.LongTensor([8, 12])
    predictions = model(test_users, test_items)
    print(f"Predictions: {predictions}")
```

---

## 4. Deep & Cross Network (DCN) 🎯

```python
class DCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32, n_cross_layers=3):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Cross layers
        input_dim = embedding_dim * 2
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(n_cross_layers)
        ])

        # Deep layers
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Final layer
        self.final = nn.Linear(input_dim + 64, 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        x0 = torch.cat([user_emb, item_emb], dim=1)

        # Cross network
        x_cross = x0
        for layer in self.cross_layers:
            x_cross = x0 * layer(x_cross) + x_cross

        # Deep network
        x_deep = self.deep(x0)

        # Combine
        x = torch.cat([x_cross, x_deep], dim=1)
        rating = self.final(x).squeeze()

        return rating
```

---

## 5. Two-Tower Model 🏗️

```python
class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()

        # User tower
        self.user_tower = nn.Sequential(
            nn.Embedding(n_users, embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Item tower
        self.item_tower = nn.Sequential(
            nn.Embedding(n_items, embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_ids, item_ids):
        user_vec = self.user_tower(user_ids)
        item_vec = self.item_tower(item_ids)

        # Dot product
        rating = (user_vec * item_vec).sum(dim=1)

        return rating

    def get_user_embeddings(self, user_ids):
        """Get user embeddings for retrieval"""
        return self.user_tower(user_ids)

    def get_item_embeddings(self, item_ids):
        """Get item embeddings for retrieval"""
        return self.item_tower(item_ids)
```

---

## Quick Reference 📖

**Neural RecSys Architectures:**
- **NCF:** Embeddings + MLP
- **DCN:** Cross + Deep network
- **Two-Tower:** Separate user/item towers
- **DeepFM:** Factorization + Deep

**Benefits:**
- Learn non-linear interactions
- Incorporate side features
- End-to-end training
- SOTA performance

---

## Key Takeaways 💡

1. **Embeddings** learn user/item representations
2. **NCF** = neural collaborative filtering
3. **Deep learning** captures complex patterns
4. **Two-tower** enables fast retrieval
5. **Combine explicit + implicit** signals
6. **Neural nets** = current SOTA

---

**Module 8 Complete!** 🎉

**Next Module:** [Module 9 - Explainable AI →](../Module%209%20-%20Explainable%20AI/Lesson%201%20-%20Feature%20Importance.md)
