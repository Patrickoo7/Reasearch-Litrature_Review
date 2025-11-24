# Lesson 2: Types of Machine Learning 🎯

**Module 1: Foundations | Lesson 2 of 4**

Now that you know what Machine Learning is, let's explore the different types of ML and when to use each one.

---

## Overview of ML Types

Machine Learning can be categorized based on **how the model learns**:

```
Machine Learning
├── Supervised Learning (labeled data)
├── Unsupervised Learning (unlabeled data)
├── Reinforcement Learning (rewards/penalties)
├── Semi-Supervised Learning (mix of labeled/unlabeled)
└── Self-Supervised Learning (create labels from data)
```

---

## 1. Supervised Learning 📊

**Definition:** Learning from **labeled examples** (input-output pairs).

### How It Works

```
Training:
Input: Email text        → Output: Spam
Input: House features   → Output: Price
Input: Patient symptoms → Output: Disease
... (thousands of examples)

Model learns: Input → Output mapping

Prediction:
New input → Model → Predicted output
```

### Two Main Types

#### A. Classification (Discrete Categories)

**Goal:** Assign inputs to predefined categories.

**Examples:**

**Binary Classification** (2 classes):
```python
from sklearn.linear_model import LogisticRegression

# Email spam detection
X_train = [[5, 10, 3],    # [num_links, num_caps, length]
           [2, 1, 5],
           [8, 15, 2]]
y_train = [1, 0, 1]       # 1=spam, 0=not spam

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict new email
new_email = [[6, 12, 4]]
prediction = model.predict(new_email)  # [1] = spam
```

**Multi-class Classification** (3+ classes):
```python
from sklearn.ensemble import RandomForestClassifier

# Image classification
X_train = image_features  # e.g., pixel values
y_train = [0, 1, 2, 1, 0, 2]  # 0=cat, 1=dog, 2=bird

model = RandomForestClassifier()
model.fit(X_train, y_train)

prediction = model.predict(new_image)  # [1] = dog
```

**Real-World Applications:**
- Email spam detection
- Sentiment analysis (positive/negative/neutral)
- Medical diagnosis (disease classification)
- Image recognition (cat/dog/bird)
- Fraud detection

#### B. Regression (Continuous Values)

**Goal:** Predict numerical values.

**Example:**
```python
from sklearn.linear_model import LinearRegression

# House price prediction
X_train = [[1200, 3, 10],   # [sq_ft, bedrooms, age]
           [1500, 4, 5],
           [1800, 3, 2]]
y_train = [200000, 300000, 350000]  # Prices in dollars

model = LinearRegression()
model.fit(X_train, y_train)

# Predict new house price
new_house = [[1600, 3, 7]]
price = model.predict(new_house)  # [285000]
```

**Real-World Applications:**
- Stock price prediction
- Temperature forecasting
- Sales forecasting
- Age estimation
- Risk assessment (insurance)

### When to Use Supervised Learning

✅ **Use when:**
- You have labeled training data
- Clear input-output relationship exists
- You can define what "correct" looks like

❌ **Don't use when:**
- No labeled data available
- Labeling data is too expensive/time-consuming
- Looking for hidden patterns (use unsupervised)

---

## 2. Unsupervised Learning 🔍

**Definition:** Learning from **unlabeled data** (find patterns without explicit labels).

### How It Works

```
Data: Just inputs, no labels
X = [samples...]  # No y labels

Model finds: Hidden structure, patterns, groupings
```

### Main Types

#### A. Clustering (Grouping)

**Goal:** Group similar data points together.

**Example:**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Customer segmentation
X = [[25, 50000],   # [age, income]
     [35, 80000],
     [28, 55000],
     [60, 90000],
     [65, 95000],
     [22, 45000]]

# Find 2 clusters
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(X)
# Result: [0, 1, 0, 1, 1, 0] - which cluster each belongs to

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
```

**Common Algorithms:**
- **K-Means**: Fast, spherical clusters
- **DBSCAN**: Arbitrary shapes, handles noise
- **Hierarchical**: Creates tree of clusters

**Real-World Applications:**
- Customer segmentation (marketing)
- Document organization
- Image compression
- Anomaly detection
- Gene sequence analysis

#### B. Dimensionality Reduction

**Goal:** Reduce number of features while preserving information.

**Example:**
```python
from sklearn.decomposition import PCA

# Original: 100 features
X = load_high_dimensional_data()  # Shape: (1000, 100)

# Reduce to 2 features for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # Shape: (1000, 2)

# Now can visualize in 2D
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.show()
```

**Common Algorithms:**
- **PCA (Principal Component Analysis)**: Linear reduction
- **t-SNE**: Non-linear, great for visualization
- **UMAP**: Faster than t-SNE, preserves structure

**Real-World Applications:**
- Data visualization
- Feature extraction
- Noise reduction
- Compression

#### C. Association Rule Learning

**Goal:** Find relationships between variables.

**Example - Market Basket Analysis:**
```
Data: Customer purchases
{bread, milk}
{bread, butter, milk}
{bread, butter}
{milk, butter, eggs}

Rules discovered:
- If {bread}, then {milk} (80% confidence)
- If {bread, butter}, then {milk} (90% confidence)
```

**Real-World Applications:**
- Recommendation systems
- Cross-selling strategies
- Medical diagnosis patterns

### When to Use Unsupervised Learning

✅ **Use when:**
- No labeled data available
- Want to discover hidden patterns
- Exploring data structure
- Preprocessing for supervised learning

❌ **Don't use when:**
- You know exactly what to predict
- Need specific predictions (use supervised)

---

## 3. Reinforcement Learning 🎮

**Definition:** Learning through **trial and error** with rewards and penalties.

### How It Works

```
Agent (learner) ←→ Environment
    ↓               ↓
  Action         State
    ↓               ↓
  Reward         New State

Goal: Maximize cumulative reward
```

### Key Concepts

**Components:**
- **Agent**: The learner/decision maker
- **Environment**: The world agent interacts with
- **State**: Current situation
- **Action**: What agent can do
- **Reward**: Feedback (+ve or -ve)
- **Policy**: Strategy (state → action mapping)

### Example: Learning to Play a Game

```python
# Simplified game: Move in a grid to reach goal

import gym

env = gym.make('FrozenLake-v1')
state = env.reset()

for step in range(100):
    # Agent chooses action (0=left, 1=down, 2=right, 3=up)
    action = agent.choose_action(state)

    # Take action
    next_state, reward, done, info = env.step(action)

    # Learn from experience
    agent.learn(state, action, reward, next_state)

    state = next_state

    if done:
        break
```

**Learning Process:**
1. Try random actions initially
2. Get rewards/penalties
3. Learn which actions work in which states
4. Improve policy over time

### Real-World Applications

- **Game playing**: Chess, Go, Atari games
- **Robotics**: Walking, grasping objects
- **Autonomous vehicles**: Self-driving cars
- **Resource management**: Power grid optimization
- **Trading**: Stock trading strategies
- **Personalization**: Content recommendation

### When to Use Reinforcement Learning

✅ **Use when:**
- Sequential decision-making problem
- Can simulate/interact with environment
- Clear reward signal exists
- Long-term consequences matter

❌ **Don't use when:**
- Simple one-shot predictions needed
- Can't interact with environment
- Supervised learning data available

---

## 4. Semi-Supervised Learning 🔄

**Definition:** Learning from **small labeled + large unlabeled data**.

### Why It Exists

**Problem:**
- Labeling data is expensive (requires human experts)
- But unlabeled data is abundant and cheap

**Solution:**
- Use small labeled set to learn initial patterns
- Use large unlabeled set to improve

### Example

```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

# Small labeled dataset
X_labeled = [[1, 2], [2, 3], [3, 1]]
y_labeled = [0, 1, 0]

# Large unlabeled dataset
X_unlabeled = [[1.5, 2], [2.5, 3], [3, 1.5], ...]

# Combine
X_train = np.vstack([X_labeled, X_unlabeled])
y_train = np.hstack([y_labeled, [-1] * len(X_unlabeled)])  # -1 = unlabeled

# Semi-supervised learning
base_classifier = SVC(probability=True)
self_training_model = SelfTrainingClassifier(base_classifier)
self_training_model.fit(X_train, y_train)
```

### Real-World Applications

- **Medical imaging**: Few labeled scans, many unlabeled
- **Speech recognition**: Limited transcribed audio
- **Web page classification**: Few manually labeled pages
- **Protein function prediction**: Few experimentally verified

---

## 5. Self-Supervised Learning 🧠

**Definition:** Create **labels automatically from the data itself**.

### How It Works

Instead of human labels, create "pretext tasks" from data:

**Example 1: Language Models**
```
Original sentence:
"The cat sat on the mat"

Pretext task: Predict masked word
"The cat [MASK] on the mat" → predict "sat"

No human labeling needed!
```

**Example 2: Image Models**
```
Original image: [cat photo]

Pretext task: Predict rotation
Rotate image 90° → predict rotation angle

Or: Predict if image patches belong together
```

### Example Code Concept

```python
# Text: Predict next word
text = "Machine learning is"
# Model learns to predict: "amazing", "powerful", "useful", etc.

# Image: Predict if patches are from same image
image_patch_1 = crop_random(image)
image_patch_2 = crop_random(image)
# Model learns: Do these belong together? Yes/No
```

### Real-World Applications

- **BERT**: Masked language modeling
- **GPT**: Next token prediction
- **SimCLR**: Contrastive learning for images
- **Word2Vec**: Word embeddings

### Why It's Powerful

✅ No labeling cost
✅ Can use massive datasets
✅ Learns general representations
✅ Transfer to many downstream tasks

---

## Comparison Summary 📊

| Type | Data Needed | Goal | Example |
|------|-------------|------|---------|
| **Supervised** | Labeled (X, y) | Predict y from X | Spam detection |
| **Unsupervised** | Unlabeled (X) | Find patterns | Customer segmentation |
| **Reinforcement** | Environment + Rewards | Maximize rewards | Game playing |
| **Semi-Supervised** | Few labeled + many unlabeled | Improve with unlabeled | Medical imaging |
| **Self-Supervised** | Unlabeled (create tasks) | Learn representations | Language models |

---

## Choosing the Right Type 🗺️

### Decision Tree

```
Do you have labeled data?
├─ Yes → Supervised Learning
│         ├─ Discrete output? → Classification
│         └─ Continuous output? → Regression
│
├─ No → Do you need to find patterns?
│       ├─ Yes → Unsupervised Learning
│       │         ├─ Group similar? → Clustering
│       │         └─ Reduce dimensions? → PCA, t-SNE
│       │
│       └─ No → Sequential decisions with feedback?
│               └─ Yes → Reinforcement Learning
│
└─ Mix (some labeled, many unlabeled)? → Semi-Supervised
```

---

## Key Takeaways 💡

1. **Supervised**: Learn from labeled examples (most common in industry)
2. **Unsupervised**: Discover patterns without labels
3. **Reinforcement**: Learn through trial and error
4. **Semi-Supervised**: Leverage unlabeled data to improve
5. **Self-Supervised**: Create your own learning tasks

**Most real-world projects use Supervised Learning!** But knowing all types helps you choose the right tool.

---

## Practice Exercise 🏋️

For each problem, identify the ML type and subtype:

1. Predict customer lifetime value based on purchase history
2. Group news articles by topic (no predefined categories)
3. Train a robot to navigate a maze
4. Classify images as cat/dog with 100 labeled + 10,000 unlabeled
5. Reduce 1000 features to 10 for visualization
6. Predict if a transaction is fraudulent
7. Learn to play chess by playing against itself

<details>
<summary>Answers</summary>

1. **Supervised - Regression** (predicting continuous value)
2. **Unsupervised - Clustering** (grouping without labels)
3. **Reinforcement Learning** (sequential decisions, rewards)
4. **Semi-Supervised - Classification** (few labeled, many unlabeled)
5. **Unsupervised - Dimensionality Reduction** (PCA, t-SNE)
6. **Supervised - Classification** (binary: fraud/not fraud)
7. **Reinforcement Learning** (game playing, learn from wins/losses)

</details>

---

## What's Next?

Now that you understand ML types, let's learn the **end-to-end ML workflow** - how to go from problem to deployed model!

**Next:** [Lesson 3 - The ML Workflow →](Lesson%203%20-%20The%20ML%20Workflow.md)

---

## Further Reading 📚

- [Andrew Ng on ML Types](https://www.coursera.org/learn/machine-learning)
- [Reinforcement Learning Introduction](http://incompleteideas.net/book/the-book.html)
- [Self-Supervised Learning Explained](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/)

---

**Congratulations!** You now understand the different types of Machine Learning and when to use each! 🎉
