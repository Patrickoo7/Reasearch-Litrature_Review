# Lesson 1: AI Bias, Fairness, and Mitigation Strategies ⚖️

**Module 17: AI Safety, Ethics & Responsible AI | Lesson 1 of 4**

Master bias detection, fairness metrics, and practical mitigation techniques for building equitable AI systems!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand different types of bias in AI systems (selection, measurement, algorithmic)
2. ✅ Implement fairness metrics: demographic parity, equalized odds, calibration
3. ✅ Detect bias in datasets and trained models
4. ✅ Apply pre-processing, in-processing, and post-processing mitigation techniques
5. ✅ Use fairness libraries (AIF360, Fairlearn) in production
6. ✅ Analyze real-world case studies (COMPAS, hiring algorithms, facial recognition)
7. ✅ Build fairness-aware ML pipelines
8. ✅ Evaluate fairness-performance trade-offs

---

## Prerequisites

- **Required**: Module 4 (Supervised Learning), Module 10 (Production ML)
- **Helpful**: Module 7 (NLP), Module 15 (LLMs)
- **Libraries**: `fairlearn`, `aif360`, `scikit-learn`, `pandas`, `shap`

```bash
pip install fairlearn aif360 scikit-learn pandas numpy shap matplotlib seaborn
```

---

## 1. Understanding AI Bias - Types and Sources

### The Bias Taxonomy

AI bias can enter systems at multiple stages. Understanding the types is critical for mitigation.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Example 1: Selection Bias in Data Collection
def demonstrate_selection_bias():
    """
    Selection bias occurs when training data is not representative
    of the population where the model will be deployed.

    Example: A hiring model trained only on historical hires
    (who were predominantly one demographic).
    """
    np.random.seed(42)

    # Simulate a population with 50/50 gender distribution
    n_population = 10000
    population_data = {
        'skill_score': np.random.normal(70, 15, n_population),
        'experience_years': np.random.gamma(3, 2, n_population),
        'gender': np.random.choice(['M', 'F'], n_population),
    }
    population_df = pd.DataFrame(population_data)

    # True hiring criterion: skill_score > 65
    population_df['should_hire'] = (population_df['skill_score'] > 65).astype(int)

    # Selection bias: Historical data only includes 80% male hires
    # due to past discrimination
    hired_male = population_df[
        (population_df['gender'] == 'M') &
        (population_df['should_hire'] == 1)
    ].sample(frac=0.8)

    hired_female = population_df[
        (population_df['gender'] == 'F') &
        (population_df['should_hire'] == 1)
    ].sample(frac=0.2)  # Only 20% of qualified females were hired

    # Biased training dataset
    biased_train = pd.concat([hired_male, hired_female])

    print("="*70)
    print("SELECTION BIAS DEMONSTRATION")
    print("="*70)
    print(f"\nPopulation distribution:")
    print(population_df['gender'].value_counts(normalize=True))
    print(f"\nBiased training data distribution:")
    print(biased_train['gender'].value_counts(normalize=True))
    print("\nThis selection bias will cause the model to learn biased patterns!")

    return population_df, biased_train


# Example 2: Measurement Bias
def demonstrate_measurement_bias():
    """
    Measurement bias occurs when features are measured differently
    across groups or proxies are used that correlate with protected attributes.
    """
    np.random.seed(42)

    n = 1000
    data = {
        'group': np.random.choice(['A', 'B'], n),
    }

    # True ability is the same for both groups
    true_ability = np.random.normal(100, 15, n)

    # Measurement bias: Group B's ability is systematically undermeasured
    # (e.g., standardized tests with cultural bias)
    measured_score = np.where(
        data['group'] == 'A',
        true_ability + np.random.normal(0, 3, n),  # Small measurement error
        true_ability - 5 + np.random.normal(0, 3, n)  # Systematic bias
    )

    df = pd.DataFrame({
        'group': data['group'],
        'true_ability': true_ability,
        'measured_score': measured_score
    })

    print("\n" + "="*70)
    print("MEASUREMENT BIAS DEMONSTRATION")
    print("="*70)
    print("\nTrue ability (same for both groups):")
    print(df.groupby('group')['true_ability'].mean())
    print("\nMeasured score (biased against Group B):")
    print(df.groupby('group')['measured_score'].mean())
    print("\nGroup B's ability is systematically undermeasured by ~5 points!")

    return df


# Example 3: Aggregation Bias
def demonstrate_aggregation_bias():
    """
    Aggregation bias occurs when a one-size-fits-all model
    is applied to groups with different data distributions.
    """
    np.random.seed(42)

    # Two groups with different relationships between features and outcome
    n_per_group = 500

    # Group A: Linear relationship
    X_a = np.random.normal(50, 10, n_per_group)
    y_a = 2 * X_a + np.random.normal(0, 5, n_per_group)

    # Group B: Different linear relationship
    X_b = np.random.normal(50, 10, n_per_group)
    y_b = 3 * X_b - 20 + np.random.normal(0, 5, n_per_group)

    # Combined dataset
    X = np.concatenate([X_a, X_b])
    y = np.concatenate([y_a, y_b])
    groups = np.array(['A']*n_per_group + ['B']*n_per_group)

    # Single model for all (aggregation bias)
    from sklearn.linear_model import LinearRegression
    model_all = LinearRegression()
    model_all.fit(X.reshape(-1, 1), y)

    # Group-specific models
    model_a = LinearRegression()
    model_a.fit(X_a.reshape(-1, 1), y_a)

    model_b = LinearRegression()
    model_b.fit(X_b.reshape(-1, 1), y_b)

    # Calculate errors
    from sklearn.metrics import mean_squared_error

    pred_all_a = model_all.predict(X_a.reshape(-1, 1))
    pred_all_b = model_all.predict(X_b.reshape(-1, 1))
    pred_specific_a = model_a.predict(X_a.reshape(-1, 1))
    pred_specific_b = model_b.predict(X_b.reshape(-1, 1))

    print("\n" + "="*70)
    print("AGGREGATION BIAS DEMONSTRATION")
    print("="*70)
    print("\nSingle model MSE:")
    print(f"  Group A: {mean_squared_error(y_a, pred_all_a):.2f}")
    print(f"  Group B: {mean_squared_error(y_b, pred_all_b):.2f}")
    print("\nGroup-specific models MSE:")
    print(f"  Group A: {mean_squared_error(y_a, pred_specific_a):.2f}")
    print(f"  Group B: {mean_squared_error(y_b, pred_specific_b):.2f}")
    print("\nAggregation bias hurts both groups!")


# Run demonstrations
pop_df, biased_df = demonstrate_selection_bias()
measurement_df = demonstrate_measurement_bias()
demonstrate_aggregation_bias()
```

### Historical Bias and Feedback Loops

```python
# Example 4: Feedback Loops in Recidivism Prediction
def simulate_feedback_loop():
    """
    Demonstrate how biased predictions can create feedback loops
    that amplify bias over time.

    Scenario: Recidivism prediction model
    """
    np.random.seed(42)

    n = 1000

    # Initial population
    data = {
        'group': np.random.choice(['Majority', 'Minority'], n, p=[0.7, 0.3]),
        'risk_score': np.random.uniform(0, 1, n),
    }

    # True recidivism rate is the same across groups (20%)
    data['true_recidivism'] = (data['risk_score'] > 0.8).astype(int)

    df = pd.DataFrame(data)

    # Biased model: Predicts higher risk for minority group
    df['predicted_risk'] = np.where(
        df['group'] == 'Minority',
        np.clip(df['risk_score'] + 0.15, 0, 1),  # Bias
        df['risk_score']
    )

    # High predicted risk → More surveillance → More arrests
    # Even for the same underlying behavior
    df['gets_arrested'] = (
        (df['true_recidivism'] == 1) |  # True recidivists
        ((df['predicted_risk'] > 0.7) & (np.random.random(n) < 0.1))  # False positives
    ).astype(int)

    print("\n" + "="*70)
    print("FEEDBACK LOOP DEMONSTRATION")
    print("="*70)
    print("\nTrue recidivism rate (equal across groups):")
    print(df.groupby('group')['true_recidivism'].mean())
    print("\nObserved arrest rate (biased due to predictions):")
    print(df.groupby('group')['gets_arrested'].mean())
    print("\nThe biased model creates a feedback loop!")
    print("Minority group has higher arrest rate despite same true risk.")

    return df


feedback_df = simulate_feedback_loop()
```

---

## 2. Fairness Metrics - Mathematical Definitions

### Core Fairness Metrics

```python
# Example 5: Implementing Fairness Metrics from Scratch
class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator.

    Supports multiple fairness definitions:
    - Demographic Parity
    - Equalized Odds
    - Equal Opportunity
    - Calibration
    - Predictive Parity
    """

    def __init__(self, y_true, y_pred, y_prob, sensitive_attr):
        """
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_prob: Predicted probabilities
            sensitive_attr: Protected attribute (e.g., race, gender)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob)
        self.sensitive_attr = np.array(sensitive_attr)
        self.groups = np.unique(sensitive_attr)

    def demographic_parity(self):
        """
        Demographic Parity (Statistical Parity):
        P(Ŷ=1 | A=a) = P(Ŷ=1 | A=b) for all groups a, b

        The probability of positive prediction should be the same
        across all groups.
        """
        results = {}
        for group in self.groups:
            mask = self.sensitive_attr == group
            positive_rate = self.y_pred[mask].mean()
            results[group] = positive_rate

        # Calculate disparity
        rates = list(results.values())
        disparity = max(rates) - min(rates)
        ratio = min(rates) / max(rates) if max(rates) > 0 else 0

        return {
            'group_rates': results,
            'disparity': disparity,
            'ratio': ratio,
            'satisfied': disparity < 0.1  # 80% rule: ratio > 0.8
        }

    def equalized_odds(self):
        """
        Equalized Odds:
        P(Ŷ=1 | Y=y, A=a) = P(Ŷ=1 | Y=y, A=b) for y ∈ {0,1}

        True positive rates and false positive rates should be
        equal across groups.
        """
        results = {}

        for group in self.groups:
            mask = self.sensitive_attr == group

            # True Positive Rate (Sensitivity)
            y_true_pos = self.y_true[mask] == 1
            if y_true_pos.sum() > 0:
                tpr = self.y_pred[mask][y_true_pos].mean()
            else:
                tpr = 0

            # False Positive Rate
            y_true_neg = self.y_true[mask] == 0
            if y_true_neg.sum() > 0:
                fpr = self.y_pred[mask][y_true_neg].mean()
            else:
                fpr = 0

            results[group] = {'tpr': tpr, 'fpr': fpr}

        # Calculate disparities
        tprs = [r['tpr'] for r in results.values()]
        fprs = [r['fpr'] for r in results.values()]

        tpr_disparity = max(tprs) - min(tprs)
        fpr_disparity = max(fprs) - min(fprs)

        return {
            'group_rates': results,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'satisfied': (tpr_disparity < 0.1) and (fpr_disparity < 0.1)
        }

    def equal_opportunity(self):
        """
        Equal Opportunity:
        P(Ŷ=1 | Y=1, A=a) = P(Ŷ=1 | Y=1, A=b)

        True positive rates should be equal (relaxed version of equalized odds).
        """
        results = {}

        for group in self.groups:
            mask = (self.sensitive_attr == group) & (self.y_true == 1)
            if mask.sum() > 0:
                tpr = self.y_pred[mask].mean()
            else:
                tpr = 0
            results[group] = tpr

        tprs = list(results.values())
        disparity = max(tprs) - min(tprs)

        return {
            'group_tprs': results,
            'disparity': disparity,
            'satisfied': disparity < 0.1
        }

    def calibration(self, n_bins=10):
        """
        Calibration:
        P(Y=1 | Ŷ=p, A=a) = p for all groups a and probabilities p

        Among individuals predicted to have probability p,
        approximately p fraction should have positive outcome.
        """
        results = {}

        for group in self.groups:
            mask = self.sensitive_attr == group

            # Bin probabilities
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(self.y_prob[mask], bins) - 1

            calibration_error = 0
            bin_count = 0

            for i in range(n_bins):
                in_bin = bin_indices == i
                if in_bin.sum() > 0:
                    predicted_prob = self.y_prob[mask][in_bin].mean()
                    true_prob = self.y_true[mask][in_bin].mean()
                    calibration_error += abs(predicted_prob - true_prob)
                    bin_count += 1

            if bin_count > 0:
                calibration_error /= bin_count

            results[group] = calibration_error

        errors = list(results.values())
        max_error = max(errors)

        return {
            'group_errors': results,
            'max_calibration_error': max_error,
            'satisfied': max_error < 0.1
        }

    def predictive_parity(self):
        """
        Predictive Parity (PPV Parity):
        P(Y=1 | Ŷ=1, A=a) = P(Y=1 | Ŷ=1, A=b)

        Precision should be equal across groups.
        """
        results = {}

        for group in self.groups:
            mask = (self.sensitive_attr == group) & (self.y_pred == 1)
            if mask.sum() > 0:
                precision = self.y_true[mask].mean()
            else:
                precision = 0
            results[group] = precision

        precisions = list(results.values())
        disparity = max(precisions) - min(precisions)

        return {
            'group_precisions': results,
            'disparity': disparity,
            'satisfied': disparity < 0.1
        }

    def generate_report(self):
        """Generate comprehensive fairness report."""
        print("="*70)
        print("FAIRNESS METRICS REPORT")
        print("="*70)

        # Demographic Parity
        dp = self.demographic_parity()
        print("\n1. DEMOGRAPHIC PARITY")
        print(f"   Positive prediction rates: {dp['group_rates']}")
        print(f"   Disparity: {dp['disparity']:.3f}")
        print(f"   Ratio: {dp['ratio']:.3f}")
        print(f"   Satisfied: {'✓' if dp['satisfied'] else '✗'}")

        # Equalized Odds
        eo = self.equalized_odds()
        print("\n2. EQUALIZED ODDS")
        for group, rates in eo['group_rates'].items():
            print(f"   {group}: TPR={rates['tpr']:.3f}, FPR={rates['fpr']:.3f}")
        print(f"   TPR Disparity: {eo['tpr_disparity']:.3f}")
        print(f"   FPR Disparity: {eo['fpr_disparity']:.3f}")
        print(f"   Satisfied: {'✓' if eo['satisfied'] else '✗'}")

        # Equal Opportunity
        eop = self.equal_opportunity()
        print("\n3. EQUAL OPPORTUNITY")
        print(f"   Group TPRs: {eop['group_tprs']}")
        print(f"   Disparity: {eop['disparity']:.3f}")
        print(f"   Satisfied: {'✓' if eop['satisfied'] else '✗'}")

        # Calibration
        cal = self.calibration()
        print("\n4. CALIBRATION")
        print(f"   Calibration errors: {cal['group_errors']}")
        print(f"   Max error: {cal['max_calibration_error']:.3f}")
        print(f"   Satisfied: {'✓' if cal['satisfied'] else '✗'}")

        # Predictive Parity
        pp = self.predictive_parity()
        print("\n5. PREDICTIVE PARITY")
        print(f"   Group precisions: {pp['group_precisions']}")
        print(f"   Disparity: {pp['disparity']:.3f}")
        print(f"   Satisfied: {'✓' if pp['satisfied'] else '✗'}")


# Example 6: Test Fairness Metrics on Synthetic Data
np.random.seed(42)
n = 1000

# Create synthetic dataset with bias
sensitive_attr = np.random.choice(['Group_A', 'Group_B'], n, p=[0.6, 0.4])
y_true = np.random.choice([0, 1], n, p=[0.7, 0.3])

# Biased predictions: Group_B has lower positive prediction rate
y_pred = np.where(
    sensitive_attr == 'Group_A',
    (np.random.random(n) < 0.35).astype(int),
    (np.random.random(n) < 0.20).astype(int)  # Bias against Group_B
)

# Probabilities (for calibration)
y_prob = np.where(
    y_pred == 1,
    np.random.beta(8, 2, n),
    np.random.beta(2, 8, n)
)

# Calculate fairness metrics
metrics = FairnessMetrics(y_true, y_pred, y_prob, sensitive_attr)
metrics.generate_report()
```

---

## 3. Detecting Bias in Datasets

### Pre-Training Bias Detection

```python
# Example 7: Dataset Bias Detection Tool
class DatasetBiasDetector:
    """
    Detect various forms of bias in datasets before training.
    """

    def __init__(self, df, target_col, sensitive_cols):
        """
        Args:
            df: DataFrame
            target_col: Name of target column
            sensitive_cols: List of sensitive attribute columns
        """
        self.df = df
        self.target_col = target_col
        self.sensitive_cols = sensitive_cols

    def check_representation_bias(self):
        """Check if sensitive groups are underrepresented."""
        print("\n" + "="*70)
        print("REPRESENTATION BIAS CHECK")
        print("="*70)

        for col in self.sensitive_cols:
            print(f"\n{col} distribution:")
            counts = self.df[col].value_counts()
            percentages = self.df[col].value_counts(normalize=True) * 100

            for val in counts.index:
                print(f"  {val}: {counts[val]} ({percentages[val]:.1f}%)")

            # Check if any group is < 10% (underrepresented)
            min_pct = percentages.min()
            if min_pct < 10:
                print(f"  ⚠️  Warning: Smallest group is only {min_pct:.1f}%")

    def check_label_bias(self):
        """Check if target labels are distributed differently across groups."""
        print("\n" + "="*70)
        print("LABEL BIAS CHECK")
        print("="*70)

        for col in self.sensitive_cols:
            print(f"\n{col} vs {self.target_col}:")

            # Cross-tabulation
            ct = pd.crosstab(
                self.df[col],
                self.df[self.target_col],
                normalize='index'
            ) * 100

            print(ct)

            # Chi-square test for independence
            from scipy.stats import chi2_contingency
            chi2, p_value, _, _ = chi2_contingency(
                pd.crosstab(self.df[col], self.df[self.target_col])
            )

            print(f"\nChi-square test: χ²={chi2:.2f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                print("  ⚠️  Significant association detected (potential bias)")

    def check_feature_correlation_with_sensitive(self):
        """Check if features are correlated with sensitive attributes."""
        print("\n" + "="*70)
        print("FEATURE-SENSITIVE CORRELATION CHECK")
        print("="*70)

        # Numeric features only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_col]

        for sens_col in self.sensitive_cols:
            if self.df[sens_col].dtype == 'object':
                # Encode categorical
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                sens_encoded = le.fit_transform(self.df[sens_col])
            else:
                sens_encoded = self.df[sens_col]

            print(f"\nCorrelations with {sens_col}:")

            correlations = []
            for feat_col in numeric_cols[:10]:  # Limit to 10 features
                corr = np.corrcoef(self.df[feat_col], sens_encoded)[0, 1]
                correlations.append((feat_col, abs(corr)))

            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)

            for feat, corr in correlations[:5]:
                print(f"  {feat}: {corr:.3f}")
                if corr > 0.3:
                    print(f"    ⚠️  High correlation - potential proxy!")

    def check_class_imbalance_per_group(self):
        """Check if class imbalance varies across groups."""
        print("\n" + "="*70)
        print("CLASS IMBALANCE PER GROUP")
        print("="*70)

        overall_imbalance = self.df[self.target_col].value_counts(normalize=True)
        print(f"\nOverall class distribution:")
        print(overall_imbalance)

        for col in self.sensitive_cols:
            print(f"\n{col}-specific class distributions:")
            for group in self.df[col].unique():
                group_df = self.df[self.df[col] == group]
                pos_rate = (group_df[self.target_col] == 1).mean()
                print(f"  {group}: {pos_rate:.1%} positive")


# Example 8: Apply Bias Detection to Adult Income Dataset
from sklearn.datasets import fetch_openml

# Load Adult Income dataset
adult = fetch_openml('adult', version=2, as_frame=True, parser='auto')
df_adult = adult.frame

# Prepare data
df_adult = df_adult.dropna()
df_adult['income'] = (df_adult['income'] == '>50K').astype(int)

# Detect bias
detector = DatasetBiasDetector(
    df_adult.sample(5000),  # Sample for speed
    target_col='income',
    sensitive_cols=['sex', 'race']
)

detector.check_representation_bias()
detector.check_label_bias()
detector.check_class_imbalance_per_group()
```

---

## 4. Bias Mitigation - Pre-processing Techniques

### Reweighting and Resampling

```python
# Example 9: Reweighting for Fairness
from sklearn.utils.class_weight import compute_sample_weight

def reweight_for_fairness(X, y, sensitive_attr):
    """
    Reweight samples to achieve demographic parity.

    Give higher weights to underrepresented group-label combinations.
    """
    # Create group-label combinations
    groups = pd.DataFrame({
        'sensitive': sensitive_attr,
        'label': y
    })
    groups['combination'] = groups['sensitive'].astype(str) + '_' + groups['label'].astype(str)

    # Calculate target distribution (uniform across combinations)
    unique_combinations = groups['combination'].unique()
    target_weight = 1.0 / len(unique_combinations)

    # Calculate current distribution
    current_dist = groups['combination'].value_counts(normalize=True)

    # Calculate weights
    weights = np.zeros(len(y))
    for combo in unique_combinations:
        mask = groups['combination'] == combo
        current_prob = current_dist.get(combo, 0)
        if current_prob > 0:
            weights[mask] = target_weight / current_prob

    # Normalize weights
    weights = weights / weights.mean()

    print("Sample weights statistics:")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Std: {weights.std():.3f}")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")

    return weights


# Example 10: Disparate Impact Remover (AIF360)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover

def apply_disparate_impact_remover(df, sensitive_col, label_col, repair_level=1.0):
    """
    Transform features to remove disparate impact while preserving rank order.

    Args:
        df: DataFrame
        sensitive_col: Name of sensitive attribute
        label_col: Name of label column
        repair_level: 0 (no repair) to 1 (full repair)
    """
    # Prepare for AIF360
    df_encoded = df.copy()

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    # Create BinaryLabelDataset
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_encoded,
        label_names=[label_col],
        protected_attribute_names=[sensitive_col]
    )

    # Apply Disparate Impact Remover
    di_remover = DisparateImpactRemover(repair_level=repair_level)
    dataset_transformed = di_remover.fit_transform(dataset)

    # Convert back to DataFrame
    df_repaired = dataset_transformed.convert_to_dataframe()[0]

    print(f"\nDisparate Impact Remover applied (repair_level={repair_level})")
    print(f"Original dataset shape: {df.shape}")
    print(f"Repaired dataset shape: {df_repaired.shape}")

    return df_repaired, dataset_transformed


# Example 11: Fair Resampling
def fair_resample(X, y, sensitive_attr, strategy='oversample'):
    """
    Resample to balance representation across sensitive groups.

    Args:
        strategy: 'oversample' or 'undersample'
    """
    from sklearn.utils import resample

    # Combine into DataFrame
    df = pd.DataFrame(X)
    df['target'] = y
    df['sensitive'] = sensitive_attr

    # Get minimum/maximum group size
    group_sizes = df.groupby('sensitive').size()

    if strategy == 'oversample':
        target_size = group_sizes.max()
    else:  # undersample
        target_size = group_sizes.min()

    # Resample each group
    resampled_dfs = []
    for group in df['sensitive'].unique():
        group_df = df[df['sensitive'] == group]

        if len(group_df) != target_size:
            group_df_resampled = resample(
                group_df,
                n_samples=target_size,
                replace=(strategy == 'oversample'),
                random_state=42
            )
        else:
            group_df_resampled = group_df

        resampled_dfs.append(group_df_resampled)

    # Combine
    df_resampled = pd.concat(resampled_dfs)

    X_resampled = df_resampled.drop(['target', 'sensitive'], axis=1).values
    y_resampled = df_resampled['target'].values
    sensitive_resampled = df_resampled['sensitive'].values

    print(f"\n{strategy.capitalize()} Resampling:")
    print(f"  Original size: {len(y)}")
    print(f"  Resampled size: {len(y_resampled)}")
    print("\nGroup distribution:")
    print(pd.Series(sensitive_resampled).value_counts())

    return X_resampled, y_resampled, sensitive_resampled
```

---

## 5. Bias Mitigation - In-processing Techniques

### Fairness-Aware Training

```python
# Example 12: Adversarial Debiasing (AIF360)
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf

def train_adversarial_debiased_model(dataset_train, sensitive_col):
    """
    Train a model with adversarial debiasing.

    The model learns to predict the target while an adversary
    tries to predict the sensitive attribute from the predictions.
    This forces the model to make predictions independent of the sensitive attribute.
    """
    # Suppress TF warnings
    tf.compat.v1.disable_eager_execution()

    # Create session
    sess = tf.compat.v1.Session()

    # Train adversarial debiasing model
    debiased_model = AdversarialDebiasing(
        privileged_groups=[{sensitive_col: 1}],
        unprivileged_groups=[{sensitive_col: 0}],
        scope_name='debiased_classifier',
        debias=True,
        sess=sess
    )

    debiased_model.fit(dataset_train)

    print("Adversarial debiasing model trained!")
    print("The model learned to predict while being fair to protected groups.")

    return debiased_model


# Example 13: Fairness Constraints with Fairlearn
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.ensemble import RandomForestClassifier

def train_with_fairness_constraints(X, y, sensitive_features, constraint_type='demographic_parity'):
    """
    Train model with fairness constraints using Fairlearn's reduction approach.

    Args:
        constraint_type: 'demographic_parity', 'equalized_odds', 'equal_opportunity'
    """
    # Choose constraint
    if constraint_type == 'demographic_parity':
        constraint = DemographicParity()
    elif constraint_type == 'equalized_odds':
        constraint = EqualizedOdds()
    else:
        from fairlearn.reductions import EqualizedOdds
        constraint = EqualizedOdds()

    # Base estimator
    base_estimator = LogisticRegression(max_iter=1000)

    # Train with constraint
    mitigator = ExponentiatedGradient(
        base_estimator,
        constraint,
        max_iter=50
    )

    mitigator.fit(X, y, sensitive_features=sensitive_features)

    print(f"\nModel trained with {constraint_type} constraint")
    print(f"Number of predictors: {len(mitigator.predictors_)}")

    return mitigator


# Example 14: Fairness-Aware Loss Function
import torch
import torch.nn as nn

class FairnessAwareLoss(nn.Module):
    """
    Custom loss combining prediction accuracy and fairness.

    Loss = BCE_loss + λ * fairness_penalty
    """

    def __init__(self, lambda_fairness=0.5):
        super().__init__()
        self.lambda_fairness = lambda_fairness
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, sensitive_attr):
        """
        Args:
            logits: Model predictions (before sigmoid)
            targets: True labels
            sensitive_attr: Binary sensitive attribute (0 or 1)
        """
        # Standard BCE loss
        bce = self.bce_loss(logits, targets.float())

        # Fairness penalty: Demographic parity violation
        # We want P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1)
        probs = torch.sigmoid(logits)

        # Positive rates per group
        group_0_mask = sensitive_attr == 0
        group_1_mask = sensitive_attr == 1

        if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
            pos_rate_0 = probs[group_0_mask].mean()
            pos_rate_1 = probs[group_1_mask].mean()
            fairness_penalty = torch.abs(pos_rate_0 - pos_rate_1)
        else:
            fairness_penalty = torch.tensor(0.0)

        # Combined loss
        total_loss = bce + self.lambda_fairness * fairness_penalty

        return total_loss, bce, fairness_penalty


# Example usage
def train_with_fairness_aware_loss():
    """Example of training with fairness-aware loss."""

    # Simple neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 10)
    sensitive = np.random.choice([0, 1], n)
    y = ((X[:, 0] + 0.5 * sensitive) > 0).astype(float)

    # Convert to PyTorch
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    sensitive_tensor = torch.LongTensor(sensitive)

    # Model and loss
    model = SimpleNN(input_dim=10)
    criterion = FairnessAwareLoss(lambda_fairness=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        logits = model(X_tensor).squeeze()
        loss, bce, fairness = criterion(logits, y_tensor, sensitive_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
                  f"BCE={bce.item():.4f}, Fairness={fairness.item():.4f}")

    print("\nTraining complete with fairness-aware loss!")


train_with_fairness_aware_loss()
```

---

## 6. Bias Mitigation - Post-processing Techniques

### Threshold Optimization

```python
# Example 15: Group-Specific Thresholds
from fairlearn.postprocessing import ThresholdOptimizer

def optimize_thresholds_per_group(estimator, X, y, sensitive_features):
    """
    Learn optimal decision thresholds per group to satisfy fairness constraints.
    """
    # Create threshold optimizer
    threshold_optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints='demographic_parity',
        objective='accuracy_score',
        prefit=False
    )

    # Fit
    threshold_optimizer.fit(X, y, sensitive_features=sensitive_features)

    # Get predictions with optimized thresholds
    y_pred_fair = threshold_optimizer.predict(X, sensitive_features=sensitive_features)

    print("\nThreshold optimization complete!")
    print("Learned group-specific thresholds to achieve fairness.")

    # Show thresholds
    print(f"\nInterpolated thresholds:")
    print(threshold_optimizer.interpolated_thresholder_)

    return threshold_optimizer, y_pred_fair


# Example 16: Calibrated Equalized Odds Post-processing
def calibrated_equalized_odds_postprocessing(y_true, y_prob, sensitive_attr):
    """
    Adjust predictions to satisfy equalized odds while maintaining calibration.

    Based on "Equality of Opportunity in Supervised Learning" (Hardt et al., 2016)
    """
    # Calculate optimal thresholds per group
    groups = np.unique(sensitive_attr)
    thresholds = {}

    # Find thresholds that equalize TPR and FPR across groups
    from scipy.optimize import minimize

    def objective(thresholds_array):
        """Minimize disparity in TPR and FPR."""
        tprs = []
        fprs = []

        for i, group in enumerate(groups):
            mask = sensitive_attr == group
            y_pred = (y_prob[mask] > thresholds_array[i]).astype(int)

            # TPR
            tpr = y_pred[y_true[mask] == 1].mean() if (y_true[mask] == 1).sum() > 0 else 0
            # FPR
            fpr = y_pred[y_true[mask] == 0].mean() if (y_true[mask] == 0).sum() > 0 else 0

            tprs.append(tpr)
            fprs.append(fpr)

        # Minimize variance in TPR and FPR
        tpr_disparity = np.var(tprs)
        fpr_disparity = np.var(fprs)

        return tpr_disparity + fpr_disparity

    # Optimize
    initial_thresholds = np.ones(len(groups)) * 0.5
    result = minimize(
        objective,
        initial_thresholds,
        bounds=[(0, 1)] * len(groups),
        method='L-BFGS-B'
    )

    optimal_thresholds = result.x

    # Apply thresholds
    y_pred_fair = np.zeros_like(y_true)
    for i, group in enumerate(groups):
        mask = sensitive_attr == group
        y_pred_fair[mask] = (y_prob[mask] > optimal_thresholds[i]).astype(int)
        thresholds[group] = optimal_thresholds[i]

    print("\nCalibrated Equalized Odds Post-processing:")
    print("Optimal thresholds per group:")
    for group, thresh in thresholds.items():
        print(f"  {group}: {thresh:.3f}")

    return y_pred_fair, thresholds


# Example 17: Reject Option Classification
def reject_option_classification(y_true, y_prob, sensitive_attr, theta=0.05):
    """
    For predictions near the decision boundary, flip labels to improve fairness.

    Args:
        theta: Margin around 0.5 threshold (0.5 ± theta is the critical region)
    """
    y_pred = (y_prob > 0.5).astype(int)
    y_pred_roc = y_pred.copy()

    # Find critical region (near decision boundary)
    critical_region = (y_prob > 0.5 - theta) & (y_prob < 0.5 + theta)

    # Calculate discrimination metrics
    groups = np.unique(sensitive_attr)
    privileged_group = groups[0]  # Assume first group is privileged

    # For instances in critical region:
    # - If from unprivileged group and predicted 0, flip to 1
    # - If from privileged group and predicted 1, flip to 0
    for group in groups:
        if group != privileged_group:  # Unprivileged group
            mask = critical_region & (sensitive_attr == group) & (y_pred == 0)
            y_pred_roc[mask] = 1
        else:  # Privileged group
            mask = critical_region & (sensitive_attr == group) & (y_pred == 1)
            y_pred_roc[mask] = 0

    # Count flips
    n_flips = (y_pred != y_pred_roc).sum()

    print(f"\nReject Option Classification:")
    print(f"  Critical region: [{0.5-theta:.2f}, {0.5+theta:.2f}]")
    print(f"  Instances in critical region: {critical_region.sum()}")
    print(f"  Labels flipped: {n_flips}")

    return y_pred_roc
```

---

## 7. Using Fairlearn in Production

### Comprehensive Fairlearn Pipeline

```python
# Example 18: Complete Fairlearn Pipeline
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import GridSearch

def build_fair_model_pipeline(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test):
    """
    Complete pipeline for building and evaluating fair models with Fairlearn.
    """
    print("="*70)
    print("FAIRLEARN COMPLETE PIPELINE")
    print("="*70)

    # Step 1: Train baseline model
    print("\n[1/5] Training baseline model...")
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    # Step 2: Evaluate baseline fairness
    print("\n[2/5] Evaluating baseline model fairness...")

    metric_frame_baseline = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'selection_rate': selection_rate,
            'tpr': true_positive_rate,
            'fpr': false_positive_rate
        },
        y_true=y_test,
        y_pred=y_pred_baseline,
        sensitive_features=sensitive_test
    )

    print("\nBaseline metrics by group:")
    print(metric_frame_baseline.by_group)

    # Step 3: Train fair model with GridSearch
    print("\n[3/5] Training fair model with GridSearch...")

    sweep = GridSearch(
        LogisticRegression(max_iter=1000),
        constraints=DemographicParity(),
        grid_size=20
    )

    sweep.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred_fair = sweep.predict(X_test)

    # Step 4: Evaluate fair model
    print("\n[4/5] Evaluating fair model...")

    metric_frame_fair = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'selection_rate': selection_rate,
            'tpr': true_positive_rate,
            'fpr': false_positive_rate
        },
        y_true=y_test,
        y_pred=y_pred_fair,
        sensitive_features=sensitive_test
    )

    print("\nFair model metrics by group:")
    print(metric_frame_fair.by_group)

    # Step 5: Compare models
    print("\n[5/5] Comparison:")
    print("\nBaseline vs Fair Model:")
    print(f"  Baseline accuracy: {metric_frame_baseline.overall['accuracy']:.3f}")
    print(f"  Fair model accuracy: {metric_frame_fair.overall['accuracy']:.3f}")
    print(f"\n  Baseline selection rate disparity: "
          f"{metric_frame_baseline.difference()['selection_rate']:.3f}")
    print(f"  Fair model selection rate disparity: "
          f"{metric_frame_fair.difference()['selection_rate']:.3f}")

    return baseline_model, sweep, metric_frame_baseline, metric_frame_fair


# Example 19: Fairlearn Dashboard Visualization
from fairlearn.metrics import MetricFrame

def create_fairness_dashboard(y_true, y_pred_models, sensitive_features, model_names):
    """
    Create comprehensive fairness comparison dashboard.

    Args:
        y_pred_models: Dict of {model_name: predictions}
    """
    import matplotlib.pyplot as plt

    metrics_dict = {
        'Accuracy': accuracy_score,
        'Selection Rate': selection_rate,
        'TPR': true_positive_rate,
        'FPR': false_positive_rate
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_name, metric_func) in enumerate(metrics_dict.items()):
        ax = axes[idx]

        # Calculate metric for each model
        for model_name, y_pred in y_pred_models.items():
            metric_frame = MetricFrame(
                metrics=metric_func,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features
            )

            # Plot by group
            groups = metric_frame.by_group.index
            values = metric_frame.by_group.values

            x_pos = np.arange(len(groups))
            ax.bar(x_pos + 0.2 * list(y_pred_models.keys()).index(model_name),
                   values, width=0.2, label=model_name, alpha=0.8)

        ax.set_xlabel('Sensitive Group')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Group')
        ax.set_xticks(x_pos + 0.2)
        ax.set_xticklabels(groups)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/fairness_dashboard.png', dpi=150, bbox_inches='tight')
    print("\nFairness dashboard saved to /tmp/fairness_dashboard.png")

    return fig
```

---

## 8. Case Study: COMPAS Recidivism Algorithm

### Analyzing ProPublica's COMPAS Data

```python
# Example 20: COMPAS Bias Analysis
def analyze_compas_bias():
    """
    Analyze bias in COMPAS recidivism prediction algorithm.

    Based on ProPublica's analysis showing racial bias.
    """
    # Note: This uses simulated data for demonstration
    # Real COMPAS data: https://github.com/propublica/compas-analysis

    np.random.seed(42)
    n = 7000

    # Simulate COMPAS-like data
    race = np.random.choice(['African-American', 'Caucasian'], n, p=[0.51, 0.49])

    # True recidivism (same rate across races: ~45%)
    true_recidivism = np.random.choice([0, 1], n, p=[0.55, 0.45])

    # COMPAS risk scores (biased)
    # African-American defendants get higher risk scores on average
    compas_score = np.where(
        race == 'African-American',
        np.random.beta(6, 4, n) * 10,  # Higher scores
        np.random.beta(4, 6, n) * 10   # Lower scores
    )

    # High risk if score > 5
    predicted_high_risk = (compas_score > 5).astype(int)

    # Create DataFrame
    df_compas = pd.DataFrame({
        'race': race,
        'compas_score': compas_score,
        'predicted_high_risk': predicted_high_risk,
        'two_year_recid': true_recidivism
    })

    print("="*70)
    print("COMPAS BIAS ANALYSIS")
    print("="*70)

    # Analysis 1: Score distribution by race
    print("\n1. Risk Score Distribution by Race:")
    print(df_compas.groupby('race')['compas_score'].describe())

    # Analysis 2: False positive rates (ProPublica's key finding)
    print("\n2. False Positive Rates (Predicted high risk but didn't recidivate):")
    for r in df_compas['race'].unique():
        race_df = df_compas[df_compas['race'] == r]

        # False positive rate: Predicted high risk but no recidivism
        fp_rate = (
            (race_df['predicted_high_risk'] == 1) &
            (race_df['two_year_recid'] == 0)
        ).sum() / (race_df['two_year_recid'] == 0).sum()

        print(f"  {r}: {fp_rate:.1%}")

    # Analysis 3: False negative rates
    print("\n3. False Negative Rates (Predicted low risk but did recidivate):")
    for r in df_compas['race'].unique():
        race_df = df_compas[df_compas['race'] == r]

        # False negative rate
        fn_rate = (
            (race_df['predicted_high_risk'] == 0) &
            (race_df['two_year_recid'] == 1)
        ).sum() / (race_df['two_year_recid'] == 1).sum()

        print(f"  {r}: {fn_rate:.1%}")

    # Fairness metrics
    print("\n4. Comprehensive Fairness Analysis:")
    metrics = FairnessMetrics(
        y_true=df_compas['two_year_recid'].values,
        y_pred=df_compas['predicted_high_risk'].values,
        y_prob=df_compas['compas_score'].values / 10,
        sensitive_attr=df_compas['race'].values
    )
    metrics.generate_report()

    return df_compas


compas_df = analyze_compas_bias()
```

---

## 9. Case Study: Hiring Algorithm Fairness

```python
# Example 21: Hiring Algorithm with Multiple Protected Attributes
def simulate_hiring_scenario():
    """
    Simulate a hiring scenario with intersectional fairness considerations.
    """
    np.random.seed(42)
    n = 2000

    # Multiple protected attributes
    gender = np.random.choice(['Male', 'Female'], n, p=[0.6, 0.4])
    race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n,
                           p=[0.6, 0.15, 0.15, 0.1])
    age_group = np.random.choice(['<30', '30-50', '>50'], n, p=[0.3, 0.5, 0.2])

    # Skills (true hiring criterion)
    technical_skill = np.random.normal(70, 15, n)
    communication_skill = np.random.normal(70, 15, n)
    experience_years = np.random.gamma(3, 2, n)

    # True hiring decision (based on skills)
    hiring_score = (
        0.4 * technical_skill +
        0.3 * communication_skill +
        0.3 * np.minimum(experience_years * 10, 100)
    )
    should_hire = (hiring_score > np.percentile(hiring_score, 80)).astype(int)

    # Biased algorithm (proxies for protected attributes)
    # Penalizes women, older candidates, and minorities
    biased_score = hiring_score.copy()
    biased_score -= (gender == 'Female') * 5
    biased_score -= (age_group == '>50') * 7
    biased_score -= (race != 'White') * 4

    predicted_hire = (biased_score > np.percentile(biased_score, 80)).astype(int)

    # Create DataFrame
    df_hiring = pd.DataFrame({
        'gender': gender,
        'race': race,
        'age_group': age_group,
        'technical_skill': technical_skill,
        'communication_skill': communication_skill,
        'experience_years': experience_years,
        'should_hire': should_hire,
        'predicted_hire': predicted_hire,
        'hiring_score': hiring_score,
        'biased_score': biased_score
    })

    print("="*70)
    print("HIRING ALGORITHM FAIRNESS ANALYSIS")
    print("="*70)

    # Analysis by each protected attribute
    for attr in ['gender', 'race', 'age_group']:
        print(f"\n{attr.upper()} Analysis:")
        print(f"\nHiring rates by {attr}:")

        for group in df_hiring[attr].unique():
            group_df = df_hiring[df_hiring[attr] == group]
            hire_rate = group_df['predicted_hire'].mean()
            print(f"  {group}: {hire_rate:.1%}")

        # Fairness metrics
        print(f"\nFairness metrics for {attr}:")
        metrics = FairnessMetrics(
            y_true=df_hiring['should_hire'].values,
            y_pred=df_hiring['predicted_hire'].values,
            y_prob=df_hiring['biased_score'].values / 100,
            sensitive_attr=df_hiring[attr].values
        )

        dp = metrics.demographic_parity()
        print(f"  Demographic parity disparity: {dp['disparity']:.3f}")
        print(f"  Ratio: {dp['ratio']:.3f}")

    # Intersectional analysis
    print("\n" + "="*70)
    print("INTERSECTIONAL ANALYSIS")
    print("="*70)

    df_hiring['intersection'] = (
        df_hiring['gender'] + ' + ' + df_hiring['race']
    )

    print("\nHiring rates by gender + race intersection:")
    intersection_rates = df_hiring.groupby('intersection')['predicted_hire'].mean().sort_values(ascending=False)
    for intersection, rate in intersection_rates.items():
        print(f"  {intersection}: {rate:.1%}")

    max_rate = intersection_rates.max()
    min_rate = intersection_rates.min()
    print(f"\nIntersectional disparity: {max_rate - min_rate:.1%}")

    return df_hiring


hiring_df = simulate_hiring_scenario()
```

---

## 10. Production Fairness Monitoring

```python
# Example 22: Real-time Fairness Monitoring System
class FairnessMonitor:
    """
    Production monitoring system for ML fairness.

    Tracks fairness metrics over time and alerts on violations.
    """

    def __init__(self, sensitive_attributes, fairness_thresholds=None):
        """
        Args:
            sensitive_attributes: List of sensitive attribute names
            fairness_thresholds: Dict of {metric: threshold}
        """
        self.sensitive_attributes = sensitive_attributes
        self.fairness_thresholds = fairness_thresholds or {
            'demographic_parity': 0.1,
            'equalized_odds_tpr': 0.1,
            'equalized_odds_fpr': 0.1
        }
        self.history = []

    def monitor_batch(self, y_true, y_pred, y_prob, sensitive_data, timestamp=None):
        """
        Monitor a batch of predictions.

        Args:
            y_true: True labels
            y_pred: Predictions
            y_prob: Prediction probabilities
            sensitive_data: DataFrame with sensitive attributes
            timestamp: Timestamp for this batch
        """
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now()

        violations = []
        metrics_summary = {'timestamp': timestamp}

        # Check each sensitive attribute
        for attr in self.sensitive_attributes:
            if attr not in sensitive_data.columns:
                continue

            # Calculate fairness metrics
            fm = FairnessMetrics(y_true, y_pred, y_prob, sensitive_data[attr])

            # Demographic parity
            dp = fm.demographic_parity()
            metrics_summary[f'{attr}_dp_disparity'] = dp['disparity']

            if dp['disparity'] > self.fairness_thresholds['demographic_parity']:
                violations.append({
                    'timestamp': timestamp,
                    'attribute': attr,
                    'metric': 'demographic_parity',
                    'value': dp['disparity'],
                    'threshold': self.fairness_thresholds['demographic_parity']
                })

            # Equalized odds
            eo = fm.equalized_odds()
            metrics_summary[f'{attr}_tpr_disparity'] = eo['tpr_disparity']
            metrics_summary[f'{attr}_fpr_disparity'] = eo['fpr_disparity']

            if eo['tpr_disparity'] > self.fairness_thresholds['equalized_odds_tpr']:
                violations.append({
                    'timestamp': timestamp,
                    'attribute': attr,
                    'metric': 'equalized_odds_tpr',
                    'value': eo['tpr_disparity'],
                    'threshold': self.fairness_thresholds['equalized_odds_tpr']
                })

        # Store in history
        self.history.append(metrics_summary)

        # Alert on violations
        if violations:
            self._alert_violations(violations)

        return metrics_summary, violations

    def _alert_violations(self, violations):
        """Send alerts for fairness violations."""
        print("\n⚠️  FAIRNESS VIOLATIONS DETECTED ⚠️")
        print("="*70)
        for v in violations:
            print(f"Time: {v['timestamp']}")
            print(f"Attribute: {v['attribute']}")
            print(f"Metric: {v['metric']}")
            print(f"Value: {v['value']:.3f} (threshold: {v['threshold']:.3f})")
            print("-"*70)

    def generate_report(self):
        """Generate fairness monitoring report."""
        if not self.history:
            print("No monitoring data available.")
            return

        df_history = pd.DataFrame(self.history)

        print("\n" + "="*70)
        print("FAIRNESS MONITORING REPORT")
        print("="*70)

        print(f"\nMonitoring period: {df_history['timestamp'].min()} to {df_history['timestamp'].max()}")
        print(f"Total batches monitored: {len(self.history)}")

        # Summary statistics
        print("\nFairness Metrics Summary:")
        for col in df_history.columns:
            if col != 'timestamp':
                print(f"\n{col}:")
                print(f"  Mean: {df_history[col].mean():.3f}")
                print(f"  Max: {df_history[col].max():.3f}")
                print(f"  Min: {df_history[col].min():.3f}")
                print(f"  Std: {df_history[col].std():.3f}")

        return df_history


# Example usage
monitor = FairnessMonitor(
    sensitive_attributes=['gender', 'race'],
    fairness_thresholds={
        'demographic_parity': 0.1,
        'equalized_odds_tpr': 0.1,
        'equalized_odds_fpr': 0.1
    }
)

# Simulate monitoring over time
for batch_id in range(5):
    # Generate batch data
    n_batch = 500
    y_true_batch = np.random.choice([0, 1], n_batch)
    y_pred_batch = np.random.choice([0, 1], n_batch)
    y_prob_batch = np.random.random(n_batch)

    sensitive_batch = pd.DataFrame({
        'gender': np.random.choice(['M', 'F'], n_batch),
        'race': np.random.choice(['A', 'B', 'C'], n_batch)
    })

    # Monitor
    metrics, violations = monitor.monitor_batch(
        y_true_batch, y_pred_batch, y_prob_batch, sensitive_batch
    )

# Generate report
report_df = monitor.generate_report()
```

---

## Practice Exercises

### Exercise 1: Implement Fairness Constraint

```python
"""
Exercise: Implement a custom fairness constraint for a classification model.

Task:
1. Create a dataset with bias against a protected group
2. Train a baseline model
3. Implement a custom fairness constraint (choose one):
   - Demographic parity
   - Equalized odds
   - Predictive rate parity
4. Train a model with your constraint
5. Compare fairness and accuracy metrics
"""

def exercise_custom_fairness_constraint():
    # TODO: Implement
    # Hint: Use Fairlearn's reduction approach or custom loss function
    pass
```

### Exercise 2: Multi-Attribute Fairness

```python
"""
Exercise: Handle fairness for multiple protected attributes simultaneously.

Task:
1. Create a dataset with multiple protected attributes (gender, race, age)
2. Implement fairness evaluation for all attributes
3. Implement mitigation that addresses all attributes
4. Analyze trade-offs between different fairness criteria
"""

def exercise_multi_attribute_fairness():
    # TODO: Implement
    pass
```

### Exercise 3: Fairness-Accuracy Trade-off Analysis

```python
"""
Exercise: Analyze the Pareto frontier of fairness vs accuracy.

Task:
1. Train models with varying fairness constraint strengths
2. Plot accuracy vs fairness metric
3. Identify the Pareto-optimal models
4. Recommend a model based on stakeholder requirements
"""

def exercise_fairness_accuracy_tradeoff():
    # TODO: Implement
    # Hint: Vary lambda in fairness-aware loss or grid_size in Fairlearn
    pass
```

---

## Key Takeaways

1. **Types of Bias**:
   - **Selection bias**: Training data not representative of deployment population
   - **Measurement bias**: Features measured differently across groups
   - **Aggregation bias**: One model for groups with different distributions
   - **Feedback loops**: Biased predictions reinforce historical bias

2. **Fairness Metrics** (often incompatible):
   - **Demographic Parity**: Equal positive prediction rates
   - **Equalized Odds**: Equal TPR and FPR across groups
   - **Equal Opportunity**: Equal TPR only
   - **Calibration**: Predictions match true probabilities per group
   - **Predictive Parity**: Equal precision across groups

3. **Mitigation Strategies**:
   - **Pre-processing**: Reweighting, resampling, disparate impact remover
   - **In-processing**: Fairness constraints, adversarial debiasing, custom loss
   - **Post-processing**: Threshold optimization, reject option classification

4. **Production Considerations**:
   - Monitor fairness metrics continuously
   - Use intersectional analysis (multiple attributes)
   - Document fairness-accuracy trade-offs
   - Establish alert thresholds for violations

5. **Fairlearn** is production-ready:
   - `fairlearn.reductions` for in-processing
   - `fairlearn.postprocessing` for threshold optimization
   - `fairlearn.metrics.MetricFrame` for evaluation

6. **Important Impossibility Results**:
   - Cannot simultaneously satisfy all fairness criteria
   - Must choose based on context and stakeholders
   - Document which criteria are prioritized and why

---

## Further Reading

### Papers
1. **Fairness Definitions**: "Fairness and Machine Learning" (Barocas, Hardt, Narayanan, 2019)
2. **Equalized Odds**: "Equality of Opportunity in Supervised Learning" (Hardt et al., 2016)
3. **Impossibility Theorem**: "Inherent Trade-Offs in the Fair Determination of Risk Scores" (Kleinberg et al., 2017)
4. **COMPAS Analysis**: "Machine Bias" (ProPublica, 2016)
5. **Adversarial Debiasing**: "Mitigating Unwanted Biases with Adversarial Learning" (Zhang et al., 2018)
6. **Calibration**: "On Fairness and Calibration" (Pleiss et al., 2017)

### Libraries & Tools
- **Fairlearn**: https://fairlearn.org/
- **AIF360** (IBM): https://aif360.mybluemix.net/
- **What-If Tool** (Google): https://pair-code.github.io/what-if-tool/
- **Aequitas** (UChicago): http://aequitas.dssg.io/

### Resources
- Fairness and Machine Learning book (free online)
- Google's ML Fairness resources
- Microsoft's Responsible AI resources
- ProPublica's COMPAS analysis

### Related Modules
- **Module 7**: NLP (bias in language models)
- **Module 10**: Production ML (monitoring)
- **Module 15 Lesson 5**: LLM Evaluation (bias in LLMs)
- **Module 17 Lesson 2**: AI Safety and Alignment
- **Module 17 Lesson 4**: Model Cards and Documentation

---

**Next Lesson**: AI Safety, Alignment, and Robustness (RLHF, adversarial attacks, jailbreak prevention)
