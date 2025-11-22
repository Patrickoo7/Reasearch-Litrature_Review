# Lesson 4: A/B Testing & Experimentation Design 🧪

**Module 4: Feature Engineering & Model Evaluation | Lesson 4 of 4**

Master A/B testing - validate your ML model's real-world impact!

---

## Why A/B Testing for ML?

**Offline metrics ≠ Real-world impact**

Example:
- Model A: 95% accuracy offline
- Model B: 90% accuracy offline
- **In production:** Model B increases revenue by 10%!

**A/B testing = Only way to measure true business impact**

---

## 1. A/B Testing Basics 📊

### Simple A/B Test

```python
import numpy as np
import pandas as pd
from scipy import stats

# Simulate experiment
np.random.seed(42)

# Control group (existing model)
control = np.random.binomial(1, 0.10, 1000)  # 10% conversion

# Treatment group (new model)
treatment = np.random.binomial(1, 0.12, 1000)  # 12% conversion

# Calculate metrics
control_rate = control.mean()
treatment_rate = treatment.mean()
lift = (treatment_rate - control_rate) / control_rate * 100

print(f"Control rate: {control_rate:.3f}")
print(f"Treatment rate: {treatment_rate:.3f}")
print(f"Lift: {lift:.1f}%")

# Statistical significance (t-test)
t_stat, p_value = stats.ttest_ind(treatment, control)

print(f"\nt-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Statistically significant")
else:
    print("❌ Not statistically significant")
```

### Z-Test for Proportions

```python
from statsmodels.stats.proportion import proportions_ztest

# Count conversions
count = np.array([treatment.sum(), control.sum()])
nobs = np.array([len(treatment), len(control)])

# Z-test
z_stat, p_value = proportions_ztest(count, nobs)

print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.4f}")
```

---

## 2. Sample Size Calculation 📏

### Required Sample Size

```python
from statsmodels.stats.power import zt_ind_solve_power

# Parameters
baseline_rate = 0.10
mde = 0.02  # Minimum detectable effect (2% absolute increase)
alpha = 0.05  # Significance level
power = 0.80  # Statistical power

# Effect size
effect_size = mde / np.sqrt(baseline_rate * (1 - baseline_rate))

# Calculate sample size per group
n_per_group = zt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='larger'
)

print(f"Required sample size per group: {int(n_per_group)}")
print(f"Total sample size: {int(n_per_group * 2)}")
```

### Duration Estimation

```python
daily_users = 10000
traffic_split = 0.5  # 50% to each group

users_per_group_per_day = daily_users * traffic_split
days_needed = n_per_group / users_per_group_per_day

print(f"Days needed: {days_needed:.1f}")
```

---

## 3. Experiment Design 🎯

### Random Assignment

```python
def assign_to_variant(user_id, num_variants=2):
    """Deterministic random assignment based on user_id"""
    import hashlib

    # Hash user_id
    hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)

    # Assign to variant
    variant = hash_value % num_variants

    return variant

# Test
users = [f"user_{i}" for i in range(10)]
for user in users:
    variant = assign_to_variant(user)
    print(f"{user}: Variant {variant}")
```

### Stratified Assignment

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Stratify by important feature (e.g., country)
df = pd.DataFrame({
    'user_id': range(1000),
    'country': np.random.choice(['US', 'UK', 'DE'], 1000)
})

# Stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for control_idx, treatment_idx in sss.split(df, df['country']):
    df.loc[control_idx, 'variant'] = 'control'
    df.loc[treatment_idx, 'variant'] = 'treatment'

# Verify balance
print(df.groupby(['variant', 'country']).size())
```

---

## 4. Statistical Tests 📈

### T-Test (Continuous Metrics)

```python
# Experiment: Average order value
control_aov = np.random.normal(50, 10, 1000)
treatment_aov = np.random.normal(52, 10, 1000)

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(treatment_aov, control_aov)

print(f"Control mean: ${control_aov.mean():.2f}")
print(f"Treatment mean: ${treatment_aov.mean():.2f}")
print(f"P-value: {p_value:.4f}")
```

### Chi-Square Test (Categorical)

```python
# Contingency table
data = pd.DataFrame({
    'variant': ['control']*500 + ['treatment']*500,
    'converted': [1]*50 + [0]*450 + [1]*70 + [0]*430
})

contingency_table = pd.crosstab(data['variant'], data['converted'])
print(contingency_table)

# Chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-square: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")
```

### Mann-Whitney U (Non-Parametric)

```python
# When data is not normal
from scipy.stats import mannwhitneyu

u_stat, p_value = mannwhitneyu(treatment_aov, control_aov, alternative='greater')

print(f"U-statistic: {u_stat:.3f}")
print(f"P-value: {p_value:.4f}")
```

---

## 5. Confidence Intervals 📊

```python
def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval"""
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

    return mean - ci, mean + ci

# Calculate CIs
control_ci = confidence_interval(control_aov)
treatment_ci = confidence_interval(treatment_aov)

print(f"Control 95% CI: ${control_ci[0]:.2f} - ${control_ci[1]:.2f}")
print(f"Treatment 95% CI: ${treatment_ci[0]:.2f} - ${treatment_ci[1]:.2f}")

# Visualize
import matplotlib.pyplot as plt

means = [control_aov.mean(), treatment_aov.mean()]
cis = [
    [control_aov.mean() - control_ci[0], treatment_aov.mean() - treatment_ci[0]],
    [control_ci[1] - control_aov.mean(), treatment_ci[1] - treatment_aov.mean()]
]

plt.bar(['Control', 'Treatment'], means, yerr=cis, capsize=5)
plt.ylabel('Average Order Value ($)')
plt.title('A/B Test Results with 95% CI')
plt.show()
```

---

## 6. Common Pitfalls 🚨

### Multiple Testing Problem

```python
# Running 20 tests at α=0.05 → 1 false positive expected!

# Solution: Bonferroni correction
n_tests = 20
alpha = 0.05
corrected_alpha = alpha / n_tests

print(f"Original α: {alpha}")
print(f"Bonferroni-corrected α: {corrected_alpha}")

# Or use False Discovery Rate
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.04, 0.03, 0.06, 0.10]
rejected, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print(f"Adjusted p-values: {p_adjusted}")
print(f"Rejected: {rejected}")
```

### Peeking (Early Stopping)

```python
# WRONG: Stop when p < 0.05 first time
# RIGHT: Wait for planned sample size

# Sequential testing (advanced)
# Use Sequential Probability Ratio Test (SPRT) or
# Alpha spending functions
```

### Sample Ratio Mismatch

```python
def check_srm(control_size, treatment_size, expected_ratio=0.5):
    """Check for Sample Ratio Mismatch"""
    observed = np.array([control_size, treatment_size])
    expected = np.array([expected_ratio, 1 - expected_ratio]) * (control_size + treatment_size)

    chi2, p_value = stats.chisquare(observed, expected)

    print(f"Chi-square: {chi2:.3f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.001:
        print("⚠️ WARNING: Sample Ratio Mismatch detected!")
        print("Check randomization logic")
    else:
        print("✅ Sample ratio OK")

check_srm(5000, 5100)  # Should be 50/50
```

---

## 7. Multi-Armed Bandits (Advanced) 🎰

**When you want to optimize during experiment**

### Epsilon-Greedy

```python
class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        else:
            return np.argmax(self.values)  # Exploit

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

# Simulate
bandit = EpsilonGreedy(n_arms=3, epsilon=0.1)

for i in range(1000):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, [0.10, 0.12, 0.09][arm])
    bandit.update(arm, reward)

print(f"Arm selection counts: {bandit.counts}")
print(f"Estimated values: {bandit.values}")
print(f"Best arm: {np.argmax(bandit.values)}")
```

### Thompson Sampling

```python
class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Successes
        self.beta = np.ones(n_arms)   # Failures

    def select_arm(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i])
                  for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Simulate
thompson = ThompsonSampling(n_arms=3)

for i in range(1000):
    arm = thompson.select_arm()
    reward = np.random.binomial(1, [0.10, 0.12, 0.09][arm])
    thompson.update(arm, reward)

print(f"Alpha (successes): {thompson.alpha}")
print(f"Beta (failures): {thompson.beta}")
print(f"Expected values: {thompson.alpha / (thompson.alpha + thompson.beta)}")
```

---

## Complete A/B Test Framework 🎯

```python
class ABTest:
    def __init__(self, alpha=0.05, power=0.80):
        self.alpha = alpha
        self.power = power
        self.control = []
        self.treatment = []

    def add_observation(self, variant, value):
        if variant == 'control':
            self.control.append(value)
        else:
            self.treatment.append(value)

    def analyze(self):
        """Run analysis"""
        # Convert to arrays
        control = np.array(self.control)
        treatment = np.array(self.treatment)

        # Calculate metrics
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        lift = (treatment_mean - control_mean) / control_mean * 100

        # Statistical test
        t_stat, p_value = stats.ttest_ind(treatment, control)

        # Confidence intervals
        control_ci = self.confidence_interval(control)
        treatment_ci = self.confidence_interval(treatment)

        # Report
        report = {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift_%': lift,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'control_ci': control_ci,
            'treatment_ci': treatment_ci,
            'n_control': len(control),
            'n_treatment': len(treatment)
        }

        return report

    def confidence_interval(self, data, confidence=0.95):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return (mean - ci, mean + ci)

    def print_report(self):
        report = self.analyze()

        print("=" * 50)
        print("A/B TEST RESULTS")
        print("=" * 50)
        print(f"Control mean: {report['control_mean']:.3f}")
        print(f"Treatment mean: {report['treatment_mean']:.3f}")
        print(f"Lift: {report['lift_%']:.2f}%")
        print(f"P-value: {report['p_value']:.4f}")
        print(f"Significant: {'✅ YES' if report['significant'] else '❌ NO'}")
        print(f"\nSample sizes:")
        print(f"  Control: {report['n_control']}")
        print(f"  Treatment: {report['n_treatment']}")
        print("=" * 50)

# Usage
test = ABTest()

# Simulate experiment
for i in range(1000):
    test.add_observation('control', np.random.normal(50, 10))
    test.add_observation('treatment', np.random.normal(52, 10))

test.print_report()
```

---

## Quick Reference 📖

**Experiment Checklist:**

```
✅ Define hypothesis
✅ Choose primary metric
✅ Calculate required sample size
✅ Randomize assignment
✅ Check sample ratio match
✅ Wait for planned duration
✅ Run statistical test
✅ Calculate confidence intervals
✅ Consider practical significance
✅ Document decision
```

**Common Tests:**

| Metric Type | Test | Use Case |
|-------------|------|----------|
| Binary (conversion) | Z-test for proportions | Click rates, conversions |
| Continuous (normal) | T-test | Revenue, time spent |
| Continuous (non-normal) | Mann-Whitney U | Skewed distributions |
| Categorical | Chi-square | Multiple categories |
| Count data | Poisson test | Events per user |

**Key Formulas:**

```python
# Lift
lift = (treatment_mean - control_mean) / control_mean * 100

# Standard error for proportions
se = np.sqrt(p * (1-p) * (1/n1 + 1/n2))

# Effect size
d = (mean1 - mean2) / pooled_std
```

---

## Key Takeaways 💡

1. **Always A/B test** production models
2. **Calculate sample size** before starting
3. **Don't peek** at results early
4. **Check sample ratio** match
5. **Practical significance ≠ statistical significance**
6. **Consider multiple metrics** (guardrails)
7. **Document everything** for reproducibility
8. **Multi-armed bandits** when optimizing during experiment

---

**Module 4 Complete!** 🎉

You now master feature engineering, evaluation, tuning, and experimentation!

**Next Module:** [Module 5 - Neural Networks →](../Module%205%20-%20Neural%20Networks%20Foundations/Lesson%201%20-%20Neural%20Network%20Basics.md)
