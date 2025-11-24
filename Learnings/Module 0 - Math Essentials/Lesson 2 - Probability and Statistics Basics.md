# Lesson 2: Probability & Statistics Basics 🎲

**Module 0: Math Essentials | Lesson 2 of 4**

Learn the probability and statistics essentials for machine learning - practical, not theoretical!

---

## Why Probability & Statistics for ML?

ML is fundamentally about **making predictions under uncertainty**:

- "What's the **probability** this email is spam?"
- "What's the **expected** house price?"
- "How **confident** are we in this prediction?"

**Everything in ML involves uncertainty!**

---

## 1. Probability Basics 🎲

### What is Probability?

**Probability**: Measure of how likely an event is (0 to 1).

```
P(Event) = Number of favorable outcomes / Total possible outcomes

0 = Impossible
0.5 = Equally likely
1 = Certain
```

**Example:**
```python
# Coin flip
P(Heads) = 0.5
P(Tails) = 0.5

# Die roll
P(Rolling 6) = 1/6 ≈ 0.167
```

### Basic Rules

**1. Complement Rule:**
```
P(NOT A) = 1 - P(A)

Example:
P(Not Spam) = 1 - P(Spam)
            = 1 - 0.3 = 0.7
```

**2. Addition Rule (OR):**
```
P(A OR B) = P(A) + P(B) - P(A AND B)

For mutually exclusive events:
P(A OR B) = P(A) + P(B)
```

**3. Multiplication Rule (AND):**
```
For independent events:
P(A AND B) = P(A) × P(B)

Example: Flip 2 coins
P(2 Heads) = P(H) × P(H) = 0.5 × 0.5 = 0.25
```

### Conditional Probability

**P(A|B)**: Probability of A given B has occurred.

```
P(A|B) = P(A AND B) / P(B)
```

**Example:**
```
Email has word "prize": P(Prize) = 0.2
Email is spam: P(Spam) = 0.3
Spam contains "prize": P(Prize AND Spam) = 0.15

P(Spam | Prize) = P(Prize AND Spam) / P(Prize)
                = 0.15 / 0.2 = 0.75

"75% of emails with 'prize' are spam"
```

### Bayes' Theorem (Super Important!)

```
P(A|B) = P(B|A) × P(A) / P(B)
```

**In ML terms:**
```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

Posterior = Likelihood × Prior / Evidence
```

**Example - Spam Detection:**
```python
P(Spam|"prize") = P("prize"|Spam) × P(Spam) / P("prize")

Given:
P("prize"|Spam) = 0.8     # 80% of spam has "prize"
P(Spam) = 0.3             # 30% of emails are spam
P("prize") = 0.2          # 20% of emails have "prize"

P(Spam|"prize") = 0.8 × 0.3 / 0.2 = 1.2... wait, > 1?

Actually: P("prize") should be calculated as:
P("prize") = P("prize"|Spam)×P(Spam) + P("prize"|Not Spam)×P(Not Spam)
           = 0.8×0.3 + 0.1×0.7 = 0.31

P(Spam|"prize") = 0.8 × 0.3 / 0.31 ≈ 0.77
```

---

## 2. Random Variables & Distributions 📊

### Random Variable

**Variable whose value is determined by chance.**

**Types:**
- **Discrete**: Countable values (die roll: 1, 2, 3, 4, 5, 6)
- **Continuous**: Any value in range (height: 5.1, 5.11, 5.111...)

### Common Distributions

#### Bernoulli Distribution (Binary)

**Single trial with two outcomes (success/failure).**

```python
import numpy as np

# Coin flip simulation
p_heads = 0.5
flip = np.random.binomial(1, p_heads)  # 0 or 1

# In ML: Binary classification output
```

#### Binomial Distribution

**Number of successes in n trials.**

```python
# 10 coin flips
n_flips = 10
p_heads = 0.5

n_heads = np.random.binomial(n_flips, p_heads)
# Could be 0, 1, 2, ..., 10
```

#### Normal (Gaussian) Distribution ⭐ Most Important!

**Bell curve - most common in nature and ML.**

```
N(μ, σ²)

μ (mu) = mean (center)
σ (sigma) = standard deviation (spread)
```

```python
import matplotlib.pyplot as plt

# Generate normal distribution
mu, sigma = 0, 1  # Mean 0, std 1 (Standard Normal)
data = np.random.normal(mu, sigma, 1000)

# Plot
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.show()
```

**Properties:**
- 68% of data within 1 standard deviation of mean
- 95% within 2 standard deviations
- 99.7% within 3 standard deviations

**Why important in ML:**
- Many real-world phenomena are normally distributed
- Weight initialization in neural networks
- Assumption in many algorithms (linear regression, Naive Bayes)

#### Uniform Distribution

**All values equally likely.**

```python
# Random number between 0 and 1
x = np.random.uniform(0, 1)

# Random number between -10 and 10
x = np.random.uniform(-10, 10)
```

---

## 3. Statistics Fundamentals 📈

### Measures of Central Tendency

**Mean (Average):**
```python
data = np.array([1, 2, 3, 4, 5, 100])  # Note outlier

mean = np.mean(data)  # 19.17
# Affected by outliers!
```

**Median (Middle value):**
```python
median = np.median(data)  # 3.5
# Robust to outliers ✅
```

**Mode (Most frequent):**
```python
from scipy import stats

data = np.array([1, 2, 2, 3, 3, 3, 4])
mode = stats.mode(data)  # 3
```

**When to use:**
- Mean: Normally distributed data
- Median: Skewed data or outliers
- Mode: Categorical data

### Measures of Spread

**Variance:**
```
Var(X) = E[(X - μ)²]
       = Average squared distance from mean
```

```python
data = np.array([1, 2, 3, 4, 5])
variance = np.var(data)  # 2.0

# Or manually
mean = np.mean(data)
variance = np.mean((data - mean) ** 2)
```

**Standard Deviation:**
```
σ = √Variance
```

```python
std = np.std(data)  # 1.414...
# Same units as original data (easier to interpret)
```

**Range:**
```python
range_val = np.max(data) - np.min(data)
```

**Interquartile Range (IQR):**
```python
Q1 = np.percentile(data, 25)  # 25th percentile
Q3 = np.percentile(data, 75)  # 75th percentile
IQR = Q3 - Q1

# Used for outlier detection:
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

### Covariance & Correlation

**Covariance**: How two variables change together.

```python
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])  # Perfectly correlated

cov = np.cov(X, Y)[0, 1]  # Positive value
# Positive: Both increase together
# Negative: One increases, other decreases
# Zero: No linear relationship
```

**Correlation**: Normalized covariance (-1 to 1).

```python
corr = np.corrcoef(X, Y)[0, 1]  # Close to 1

# -1: Perfect negative correlation
#  0: No correlation
#  1: Perfect positive correlation
```

**Pearson Correlation:**
```python
from scipy.stats import pearsonr

corr, p_value = pearsonr(X, Y)
print(f"Correlation: {corr:.3f}")
print(f"P-value: {p_value:.3f}")
```

**In ML:**
```python
import seaborn as sns
import pandas as pd

# Correlation matrix for feature selection
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 4, 6, 8, 10],
    'feature3': [5, 4, 3, 2, 1]
})

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
```

---

## 4. Statistical Inference

### Hypothesis Testing

**Framework for making decisions about populations from samples.**

**Steps:**
1. State null hypothesis (H₀) and alternative (H₁)
2. Choose significance level (α, typically 0.05)
3. Calculate test statistic
4. Compare p-value to α
5. Reject or fail to reject H₀

**Example - A/B Testing:**
```python
from scipy.stats import ttest_ind

# Two versions of website
version_A_conversions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
version_B_conversions = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]

# H₀: No difference between versions
# H₁: Version B is better

t_stat, p_value = ttest_ind(version_A_conversions, version_B_conversions)

if p_value < 0.05:
    print("Significant difference! Use version B")
else:
    print("No significant difference")
```

### Confidence Intervals

**Range likely to contain true value.**

```python
from scipy import stats

data = np.random.normal(100, 15, 100)  # Mean 100, std 15

# 95% confidence interval for mean
mean = np.mean(data)
sem = stats.sem(data)  # Standard error of mean
ci = stats.t.interval(0.95, len(data)-1, mean, sem)

print(f"95% CI: {ci}")
# "We're 95% confident true mean is in this range"
```

### P-values

**Probability of observing data if null hypothesis is true.**

```
p-value < 0.05: Reject null hypothesis (significant)
p-value ≥ 0.05: Fail to reject (not significant)
```

**Common mistake:**
```
❌ p-value = probability hypothesis is true
✅ p-value = probability of data given hypothesis is true
```

---

## 5. ML Applications of Probability & Statistics

### 1. Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB

# Uses Bayes' theorem
model = GaussianNB()
model.fit(X_train, y_train)

# Returns probabilities
probabilities = model.predict_proba(X_test)
```

### 2. Normal Distribution in Neural Networks

```python
# Weight initialization
weights = np.random.normal(0, 0.01, (100, 50))
# Mean 0, small std for stable training
```

### 3. Confidence in Predictions

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get probability estimates
probs = model.predict_proba(X_test)

# Confidence = max probability
confidence = np.max(probs, axis=1)
```

### 4. Statistical Feature Engineering

```python
# Rolling statistics for time series
df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['rolling_std'] = df['value'].rolling(window=7).std()
```

---

## Quick Reference 📖

| Concept | Formula | NumPy/SciPy |
|---------|---------|-------------|
| Mean | Σx / n | `np.mean(x)` |
| Variance | Σ(x-μ)² / n | `np.var(x)` |
| Std Dev | √Variance | `np.std(x)` |
| Covariance | E[(X-μₓ)(Y-μᵧ)] | `np.cov(X,Y)` |
| Correlation | Cov(X,Y)/(σₓσᵧ) | `np.corrcoef(X,Y)` |
| Normal PDF | ... | `stats.norm.pdf(x, μ, σ)` |
| T-test | ... | `stats.ttest_ind(a, b)` |

---

## Common Distributions Cheat Sheet

```python
import scipy.stats as stats

# Normal
x = stats.norm.rvs(loc=0, scale=1, size=1000)  # mean=0, std=1

# Uniform
x = stats.uniform.rvs(loc=0, scale=10, size=1000)  # 0 to 10

# Binomial
x = stats.binom.rvs(n=10, p=0.5, size=1000)  # 10 trials, p=0.5

# Poisson (counts)
x = stats.poisson.rvs(mu=5, size=1000)  # mean=5
```

---

## Practice Exercises 🏋️

1. Calculate mean, median, std of [1, 2, 3, 100]
2. What's P(Spam|"win") if P("win"|Spam)=0.9, P(Spam)=0.3, P("win")=0.35?
3. Generate 1000 samples from N(50, 10) and plot
4. Test if two datasets have different means

<details>
<summary>Solutions</summary>

```python
# 1. Statistics
data = np.array([1, 2, 3, 100])
print(f"Mean: {np.mean(data)}")      # 26.5
print(f"Median: {np.median(data)}")  # 2.5 (robust!)
print(f"Std: {np.std(data)}")        # 42.66

# 2. Bayes' theorem
P_win_given_spam = 0.9
P_spam = 0.3
P_win = 0.35
P_spam_given_win = (P_win_given_spam * P_spam) / P_win
# = 0.77

# 3. Normal distribution
data = np.random.normal(50, 10, 1000)
plt.hist(data, bins=50)
plt.show()

# 4. T-test
from scipy.stats import ttest_ind
group1 = np.random.normal(50, 10, 100)
group2 = np.random.normal(55, 10, 100)
t_stat, p_value = ttest_ind(group1, group2)
```
</details>

---

## Key Takeaways 💡

1. **Probability** measures uncertainty (0 to 1)
2. **Bayes' Theorem** is fundamental to ML
3. **Normal distribution** appears everywhere
4. **Mean vs Median** - use median for skewed data
5. **Correlation** ≠ Causation
6. **P-values** help test hypotheses
7. **Confidence intervals** quantify uncertainty

---

## What's Next?

You now understand probability and statistics for ML! Next, learn **calculus for machine learning** - don't worry, just the essentials!

**Next:** [Lesson 3 - Calculus for ML →](Lesson%203%20-%20Calculus%20for%20ML.md)

---

**Congratulations!** You can now reason about uncertainty in ML! 🎉
