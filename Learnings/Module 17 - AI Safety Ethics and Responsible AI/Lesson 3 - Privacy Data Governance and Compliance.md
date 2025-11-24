# Lesson 3: Privacy, Data Governance, and Compliance 🔒

**Module 17: AI Safety, Ethics & Responsible AI | Lesson 3 of 4**

Master privacy-preserving ML, differential privacy, federated learning, and regulatory compliance for production AI!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand data privacy principles (GDPR, CCPA, HIPAA)
2. ✅ Implement differential privacy in ML pipelines
3. ✅ Build federated learning systems for privacy-preserving training
4. ✅ Detect and scrub PII (Personally Identifiable Information)
5. ✅ Apply data minimization and purpose limitation
6. ✅ Implement secure multi-party computation (SMPC) techniques
7. ✅ Conduct privacy impact assessments and audits
8. ✅ Build compliant data governance frameworks

---

## Prerequisites

- **Required**: Module 4 (Supervised Learning), Module 10 (Production ML)
- **Helpful**: Module 7 (NLP for PII detection), Module 15 (LLMs)
- **Libraries**: `opacus`, `presidio`, `crypten`, `pysyft`, `transformers`

```bash
pip install opacus presidio-analyzer presidio-anonymizer
pip install pysyft crypten
pip install faker  # For generating synthetic data
```

---

## 1. Data Privacy Principles and Regulations

### GDPR, CCPA, and HIPAA Compliance

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Example 1: Understanding Privacy Regulations
class PrivacyRegulationChecker:
    """
    Check ML workflows for privacy regulation compliance.

    Supports:
    - GDPR (EU General Data Protection Regulation)
    - CCPA (California Consumer Privacy Act)
    - HIPAA (Health Insurance Portability and Accountability Act)
    """

    def __init__(self, regulation='GDPR'):
        """
        Args:
            regulation: 'GDPR', 'CCPA', or 'HIPAA'
        """
        self.regulation = regulation
        self.requirements = self._get_requirements()

    def _get_requirements(self):
        """Get requirements for each regulation."""

        requirements = {
            'GDPR': {
                'lawful_basis': 'Must have legal basis for processing (consent, contract, etc.)',
                'data_minimization': 'Collect only necessary data',
                'purpose_limitation': 'Use data only for stated purpose',
                'storage_limitation': 'Delete data when no longer needed',
                'right_to_access': 'Provide data subject access to their data',
                'right_to_erasure': 'Delete data upon request (right to be forgotten)',
                'right_to_portability': 'Provide data in machine-readable format',
                'privacy_by_design': 'Build privacy into systems from the start',
                'data_protection_impact_assessment': 'Required for high-risk processing',
                'dpo_required': 'Data Protection Officer for large-scale processing',
            },
            'CCPA': {
                'right_to_know': 'Disclose what personal info is collected',
                'right_to_delete': 'Delete personal info upon request',
                'right_to_opt_out': 'Opt-out of sale of personal info',
                'non_discrimination': 'Cannot discriminate for exercising rights',
                'notice_at_collection': 'Inform users about data collection',
                'verifiable_requests': 'Verify identity for requests',
            },
            'HIPAA': {
                'minimum_necessary': 'Use/disclose minimum necessary PHI',
                'patient_rights': 'Right to access and amend health records',
                'safeguards': 'Administrative, physical, technical safeguards',
                'business_associate_agreements': 'Contracts with third parties',
                'breach_notification': 'Notify of data breaches',
                'de_identification': 'Properly de-identify PHI',
            }
        }

        return requirements.get(self.regulation, {})

    def check_compliance(self, ml_workflow_config):
        """
        Check if ML workflow meets regulatory requirements.

        Args:
            ml_workflow_config: Dict with workflow configuration

        Returns:
            Compliance report
        """
        print("="*70)
        print(f"{self.regulation} COMPLIANCE CHECK")
        print("="*70)

        compliance_status = {}

        for requirement, description in self.requirements.items():
            # Check each requirement
            is_compliant = self._check_requirement(requirement, ml_workflow_config)

            compliance_status[requirement] = {
                'compliant': is_compliant,
                'description': description
            }

            status_symbol = "✓" if is_compliant else "✗"
            print(f"\n{status_symbol} {requirement}")
            print(f"   {description}")
            if not is_compliant:
                print(f"   ⚠️  NOT COMPLIANT - Action required")

        overall_compliant = all(v['compliant'] for v in compliance_status.values())

        print("\n" + "="*70)
        print(f"Overall compliance: {'✓ COMPLIANT' if overall_compliant else '✗ NON-COMPLIANT'}")

        return compliance_status

    def _check_requirement(self, requirement, config):
        """Check specific requirement against config."""

        # Simplified checks (in production, these would be more thorough)
        checks = {
            'lawful_basis': config.get('consent_obtained', False),
            'data_minimization': config.get('minimal_data_collection', False),
            'purpose_limitation': config.get('purpose_documented', False),
            'storage_limitation': config.get('retention_policy', False),
            'right_to_access': config.get('data_access_api', False),
            'right_to_erasure': config.get('deletion_api', False),
            'privacy_by_design': config.get('privacy_by_design', False),
            'minimum_necessary': config.get('minimal_data_collection', False),
            'safeguards': config.get('encryption_enabled', False),
        }

        return checks.get(requirement, False)


# Example 2: ML Workflow Compliance Check
def check_ml_workflow_compliance():
    """Check example ML workflow for GDPR compliance."""

    # Example ML workflow configuration
    workflow_config = {
        'consent_obtained': True,
        'minimal_data_collection': True,
        'purpose_documented': True,
        'retention_policy': False,  # Missing!
        'data_access_api': True,
        'deletion_api': False,  # Missing!
        'privacy_by_design': True,
        'encryption_enabled': True,
    }

    checker = PrivacyRegulationChecker(regulation='GDPR')
    compliance_report = checker.check_compliance(workflow_config)

    print("\n📋 Recommendations:")
    print("  1. Implement data retention and deletion policies")
    print("  2. Build API for user data deletion requests")
    print("  3. Document compliance in privacy policy")


check_ml_workflow_compliance()


# Example 3: Data Retention Policy Implementation
class DataRetentionPolicy:
    """
    Implement data retention policy for compliance.

    Automatically deletes data after specified retention period.
    """

    def __init__(self, retention_days=90):
        """
        Args:
            retention_days: Number of days to retain data
        """
        self.retention_days = retention_days

    def should_delete(self, data_timestamp):
        """
        Check if data should be deleted based on age.

        Args:
            data_timestamp: Timestamp when data was collected

        Returns:
            True if should delete
        """
        age = datetime.now() - data_timestamp
        return age.days > self.retention_days

    def enforce_retention(self, dataset):
        """
        Remove data older than retention period.

        Args:
            dataset: DataFrame with 'collected_at' column

        Returns:
            Filtered dataset
        """
        print(f"\nEnforcing {self.retention_days}-day retention policy...")
        print(f"Original dataset size: {len(dataset)}")

        # Filter out old data
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        dataset_filtered = dataset[dataset['collected_at'] >= cutoff_date]

        deleted_count = len(dataset) - len(dataset_filtered)
        print(f"Deleted {deleted_count} records older than {self.retention_days} days")
        print(f"Remaining dataset size: {len(dataset_filtered)}")

        return dataset_filtered


# Test retention policy
dates = [datetime.now() - timedelta(days=d) for d in [10, 50, 100, 150, 200]]
test_df = pd.DataFrame({
    'user_id': range(5),
    'data': ['data1', 'data2', 'data3', 'data4', 'data5'],
    'collected_at': dates
})

policy = DataRetentionPolicy(retention_days=90)
filtered_df = policy.enforce_retention(test_df)
```

---

## 2. Differential Privacy Fundamentals

### Understanding ε-Differential Privacy

```python
# Example 4: Differential Privacy Basics
def demonstrate_differential_privacy():
    """
    Demonstrate the concept of differential privacy.

    ε-differential privacy: Adding/removing one person's data
    changes query result probability by at most e^ε.
    """
    print("="*70)
    print("DIFFERENTIAL PRIVACY DEMONSTRATION")
    print("="*70)

    # Dataset with salaries
    salaries = np.array([50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000])

    # True mean
    true_mean = salaries.mean()

    print(f"\nTrue mean salary: ${true_mean:,.0f}")

    # Non-private query (exact answer)
    print(f"\nNon-private query result: ${true_mean:,.0f}")
    print("Privacy risk: Individual salaries can be inferred!")

    # Differentially private query (add Laplace noise)
    def laplace_mechanism(true_value, sensitivity, epsilon):
        """
        Add Laplace noise for differential privacy.

        Args:
            true_value: True query result
            sensitivity: Maximum change from adding/removing one person
            epsilon: Privacy budget (smaller = more private)

        Returns:
            Noisy result
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise

    # Sensitivity of mean: max_value / n
    sensitivity = (salaries.max() - salaries.min()) / len(salaries)

    # Different privacy levels
    epsilons = [0.1, 0.5, 1.0, 5.0]

    print("\n" + "-"*70)
    print("Differentially Private Queries:")
    print("-"*70)

    for eps in epsilons:
        noisy_mean = laplace_mechanism(true_mean, sensitivity, eps)
        error = abs(noisy_mean - true_mean)

        print(f"\nε = {eps}:")
        print(f"  Result: ${noisy_mean:,.0f}")
        print(f"  Error: ${error:,.0f}")
        print(f"  Privacy level: {'High' if eps < 1 else 'Medium' if eps < 5 else 'Low'}")

    print("\n" + "="*70)
    print("Key insight: Lower ε = more privacy but more noise")


demonstrate_differential_privacy()


# Example 5: Implementing Differential Privacy with Opacus
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_with_differential_privacy():
    """
    Train a neural network with differential privacy using Opacus.

    Opacus adds noise to gradients during training to provide
    formal privacy guarantees.
    """
    print("\n" + "="*70)
    print("DIFFERENTIALLY PRIVATE DEEP LEARNING")
    print("="*70)

    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    # Create synthetic dataset
    n_samples = 1000
    n_features = 10

    X = torch.randn(n_samples, n_features)
    y = (X.sum(dim=1) > 0).long()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Simple model
    model = nn.Sequential(
        nn.Linear(n_features, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Privacy parameters
    epsilon = 1.0  # Privacy budget
    delta = 1e-5  # Probability of privacy breach
    max_grad_norm = 1.0  # Gradient clipping threshold

    # Attach privacy engine
    privacy_engine = PrivacyEngine()

    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.1,  # Amount of noise to add
        max_grad_norm=max_grad_norm,
    )

    print(f"\nPrivacy parameters:")
    print(f"  Target ε: {epsilon}")
    print(f"  δ: {delta}")
    print(f"  Max gradient norm: {max_grad_norm}")

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate privacy spent
        epsilon_spent = privacy_engine.get_epsilon(delta)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={total_loss/len(dataloader):.4f}, "
              f"ε={epsilon_spent:.2f}")

    print(f"\nFinal privacy guarantee: (ε={epsilon_spent:.2f}, δ={delta})")
    print("This means the model provides differential privacy!")


train_with_differential_privacy()


# Example 6: Private Aggregation
class PrivateAggregator:
    """
    Perform differentially private aggregation queries.

    Useful for reporting statistics without revealing individual data.
    """

    def __init__(self, epsilon=1.0):
        """
        Args:
            epsilon: Privacy budget
        """
        self.epsilon = epsilon

    def private_count(self, data):
        """
        Count with differential privacy.

        Sensitivity = 1 (adding/removing one person changes count by 1)
        """
        true_count = len(data)
        noise = np.random.laplace(0, 1 / self.epsilon)
        return max(0, true_count + noise)  # Ensure non-negative

    def private_sum(self, data, value_range):
        """
        Sum with differential privacy.

        Args:
            data: Array of values
            value_range: (min_value, max_value) for sensitivity calculation
        """
        true_sum = np.sum(data)
        sensitivity = value_range[1] - value_range[0]
        noise = np.random.laplace(0, sensitivity / self.epsilon)
        return true_sum + noise

    def private_mean(self, data, value_range):
        """Mean with differential privacy."""
        private_sum = self.private_sum(data, value_range)
        private_count = self.private_count(data)

        if private_count > 0:
            return private_sum / private_count
        return 0

    def private_histogram(self, data, bins, value_range):
        """
        Histogram with differential privacy.

        Each bin count is privatized independently.
        """
        # Create histogram
        hist, bin_edges = np.histogram(data, bins=bins, range=value_range)

        # Add noise to each bin
        epsilon_per_bin = self.epsilon / len(hist)  # Split privacy budget
        noisy_hist = []

        for count in hist:
            noise = np.random.laplace(0, 1 / epsilon_per_bin)
            noisy_count = max(0, count + noise)
            noisy_hist.append(noisy_count)

        return np.array(noisy_hist), bin_edges


# Example usage
ages = np.random.randint(18, 80, 500)

aggregator = PrivateAggregator(epsilon=1.0)

print("\n" + "="*70)
print("PRIVATE AGGREGATION")
print("="*70)

print(f"\nTrue count: {len(ages)}")
print(f"Private count: {aggregator.private_count(ages):.0f}")

print(f"\nTrue mean age: {ages.mean():.1f}")
print(f"Private mean age: {aggregator.private_mean(ages, (18, 80)):.1f}")

hist, bins = aggregator.private_histogram(ages, bins=5, value_range=(18, 80))
print(f"\nPrivate age histogram:")
for i, count in enumerate(hist):
    print(f"  {bins[i]:.0f}-{bins[i+1]:.0f}: {count:.0f}")
```

---

## 3. Federated Learning

### Privacy-Preserving Distributed Training

```python
# Example 7: Federated Learning Basics
class FederatedLearningServer:
    """
    Server for federated learning.

    Aggregates model updates from clients without accessing raw data.
    """

    def __init__(self, global_model):
        """
        Args:
            global_model: PyTorch model to be trained federatively
        """
        self.global_model = global_model
        self.round = 0

    def aggregate_updates(self, client_updates, client_data_sizes):
        """
        Aggregate client model updates using weighted averaging.

        Args:
            client_updates: List of client model state dicts
            client_data_sizes: List of dataset sizes for each client

        Returns:
            Aggregated model state dict
        """
        # FedAvg: Weighted average based on dataset size
        total_data = sum(client_data_sizes)

        # Initialize aggregated state
        aggregated_state = {}

        # Get parameter names from first client
        param_names = client_updates[0].keys()

        for param_name in param_names:
            # Weighted sum
            weighted_sum = None

            for client_state, data_size in zip(client_updates, client_data_sizes):
                weight = data_size / total_data
                param = client_state[param_name]

                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param

            aggregated_state[param_name] = weighted_sum

        return aggregated_state

    def train_round(self, clients, num_local_epochs=1):
        """
        Perform one round of federated training.

        Args:
            clients: List of FederatedClient instances
            num_local_epochs: Epochs each client trains locally

        Returns:
            Average loss across clients
        """
        print(f"\n--- Federated Learning Round {self.round + 1} ---")

        # Distribute global model to clients
        for client in clients:
            client.receive_global_model(self.global_model.state_dict())

        # Clients train locally
        client_updates = []
        client_data_sizes = []
        losses = []

        for i, client in enumerate(clients):
            print(f"  Client {i+1} training...")
            avg_loss = client.train_local(epochs=num_local_epochs)
            losses.append(avg_loss)

            # Collect updates
            client_updates.append(client.get_model_update())
            client_data_sizes.append(len(client.dataset))

        # Aggregate updates
        aggregated_state = self.aggregate_updates(client_updates, client_data_sizes)

        # Update global model
        self.global_model.load_state_dict(aggregated_state)

        self.round += 1
        avg_loss = np.mean(losses)

        print(f"  Average loss: {avg_loss:.4f}")

        return avg_loss


class FederatedClient:
    """
    Client for federated learning.

    Trains on local data without sharing it.
    """

    def __init__(self, dataset, model_template):
        """
        Args:
            dataset: Local dataset (TensorDataset)
            model_template: Model architecture (not trained)
        """
        self.dataset = dataset
        self.model = model_template
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    def receive_global_model(self, global_state):
        """Receive global model from server."""
        self.model.load_state_dict(global_state)

    def train_local(self, epochs=1):
        """Train on local data."""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0

        for epoch in range(epochs):
            for X_batch, y_batch in self.dataloader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return total_loss / (len(self.dataloader) * epochs)

    def get_model_update(self):
        """Return trained model state."""
        return self.model.state_dict()


# Example 8: Federated Learning Simulation
def simulate_federated_learning():
    """Simulate federated learning with multiple clients."""

    print("="*70)
    print("FEDERATED LEARNING SIMULATION")
    print("="*70)

    # Create model template
    model_template = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    # Create global model
    global_model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    # Create clients with different local datasets
    num_clients = 5
    clients = []

    print(f"\nCreating {num_clients} clients with local data...")

    for i in range(num_clients):
        # Each client has different local data
        n_samples = np.random.randint(100, 500)
        X_local = torch.randn(n_samples, 10)
        y_local = (X_local.sum(dim=1) > 0).long()

        dataset_local = TensorDataset(X_local, y_local)

        client = FederatedClient(dataset_local, model_template)
        clients.append(client)

        print(f"  Client {i+1}: {n_samples} samples")

    # Create server
    server = FederatedLearningServer(global_model)

    # Federated training
    print(f"\nStarting federated training...")

    num_rounds = 5
    for round in range(num_rounds):
        avg_loss = server.train_round(clients, num_local_epochs=2)

    print(f"\n{'='*70}")
    print("Federated learning complete!")
    print("Server learned from all clients without accessing raw data!")


simulate_federated_learning()


# Example 9: Secure Aggregation
def secure_aggregation(client_updates, threshold=0.5):
    """
    Secure aggregation with dropout resilience.

    Only aggregates if enough clients participate.
    Uses cryptographic techniques in production (e.g., secret sharing).

    Args:
        client_updates: List of client updates
        threshold: Minimum fraction of clients required

    Returns:
        Aggregated update or None
    """
    min_clients = int(threshold * len(client_updates))

    if len(client_updates) < min_clients:
        print(f"⚠️  Not enough clients ({len(client_updates)} < {min_clients})")
        return None

    # Simple averaging (in production, use secure multi-party computation)
    aggregated = {}

    for param_name in client_updates[0].keys():
        param_sum = sum(update[param_name] for update in client_updates)
        aggregated[param_name] = param_sum / len(client_updates)

    print(f"✓ Secure aggregation with {len(client_updates)} clients")
    return aggregated
```

---

## 4. PII Detection and Anonymization

### Detecting and Removing Sensitive Information

```python
# Example 10: PII Detection with Presidio
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def detect_and_anonymize_pii(text):
    """
    Detect and anonymize PII in text.

    Uses Microsoft Presidio for NER-based PII detection.

    Args:
        text: Input text potentially containing PII

    Returns:
        Anonymized text, detected entities
    """
    # Initialize Presidio
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # Analyze text for PII
    results = analyzer.analyze(
        text=text,
        language='en',
        entities=[
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
            "LOCATION", "DATE_TIME", "NRP", "MEDICAL_LICENSE"
        ]
    )

    print(f"\n{'='*70}")
    print("PII DETECTION RESULTS")
    print(f"{'='*70}")

    print(f"\nOriginal text:\n{text}")

    print(f"\nDetected PII:")
    for result in results:
        print(f"  - {result.entity_type}: '{text[result.start:result.end]}' "
              f"(confidence: {result.score:.2f})")

    # Anonymize
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )

    print(f"\nAnonymized text:\n{anonymized_result.text}")

    return anonymized_result.text, results


# Example 11: Test PII Detection
test_texts = [
    "John Smith's email is john.smith@example.com and his phone is 555-123-4567.",
    "The patient Sarah Johnson (DOB: 01/15/1980) was seen at the clinic on Main Street.",
    "Credit card number 4532-1234-5678-9010 was used for the transaction.",
]

for text in test_texts:
    anonymized, entities = detect_and_anonymize_pii(text)
    print("\n" + "-"*70)


# Example 12: Custom PII Detector for Structured Data
class StructuredDataPIIDetector:
    """
    Detect PII in structured data (DataFrames).

    Uses pattern matching and heuristics.
    """

    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
        }

        # Column names that likely contain PII
        self.pii_column_names = [
            'name', 'email', 'phone', 'address', 'ssn',
            'credit_card', 'dob', 'birth_date', 'social_security'
        ]

    def detect_pii_columns(self, df):
        """
        Detect which columns contain PII.

        Args:
            df: pandas DataFrame

        Returns:
            Dict of {column: pii_type}
        """
        import re

        pii_columns = {}

        for col in df.columns:
            col_lower = col.lower()

            # Check column name
            for pii_name in self.pii_column_names:
                if pii_name in col_lower:
                    pii_columns[col] = f"column_name_{pii_name}"
                    break

            # Check content patterns
            if col not in pii_columns and df[col].dtype == 'object':
                sample = df[col].dropna().astype(str).head(100)

                for pii_type, pattern in self.pii_patterns.items():
                    matches = sample.str.contains(pattern, regex=True, na=False).sum()

                    if matches / len(sample) > 0.1:  # >10% matches
                        pii_columns[col] = f"pattern_{pii_type}"
                        break

        return pii_columns

    def anonymize_dataframe(self, df, method='mask'):
        """
        Anonymize PII in DataFrame.

        Args:
            df: pandas DataFrame
            method: 'mask', 'hash', 'generalize', or 'remove'

        Returns:
            Anonymized DataFrame
        """
        import hashlib

        df_anon = df.copy()
        pii_columns = self.detect_pii_columns(df)

        print(f"\nDetected PII columns: {list(pii_columns.keys())}")

        for col, pii_type in pii_columns.items():
            print(f"  Anonymizing {col} ({pii_type})...")

            if method == 'mask':
                # Replace with asterisks
                df_anon[col] = '***REDACTED***'

            elif method == 'hash':
                # One-way hash
                df_anon[col] = df_anon[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
                    if pd.notna(x) else x
                )

            elif method == 'generalize':
                # Generalize (e.g., age → age range)
                if 'age' in col.lower():
                    df_anon[col] = pd.cut(
                        df[col],
                        bins=[0, 18, 35, 50, 65, 100],
                        labels=['0-18', '19-35', '36-50', '51-65', '65+']
                    )

            elif method == 'remove':
                # Remove column entirely
                df_anon = df_anon.drop(columns=[col])

        return df_anon


# Example 13: Test Structured PII Detection
from faker import Faker
fake = Faker()

# Create synthetic dataset with PII
n_rows = 100
df_with_pii = pd.DataFrame({
    'user_id': range(n_rows),
    'name': [fake.name() for _ in range(n_rows)],
    'email': [fake.email() for _ in range(n_rows)],
    'phone': [fake.phone_number() for _ in range(n_rows)],
    'age': np.random.randint(18, 80, n_rows),
    'purchase_amount': np.random.uniform(10, 1000, n_rows),
})

print("\n" + "="*70)
print("STRUCTURED DATA PII DETECTION")
print("="*70)

print(f"\nOriginal data (first 3 rows):")
print(df_with_pii.head(3))

detector = StructuredDataPIIDetector()
df_anonymized = detector.anonymize_dataframe(df_with_pii, method='hash')

print(f"\nAnonymized data (first 3 rows):")
print(df_anonymized.head(3))
```

---

## 5. Data Minimization and Purpose Limitation

### Collecting Only Necessary Data

```python
# Example 14: Data Minimization Framework
class DataMinimizationFramework:
    """
    Ensure only necessary data is collected and used.

    Implements purpose limitation and data minimization principles.
    """

    def __init__(self, purpose):
        """
        Args:
            purpose: Stated purpose for data collection
        """
        self.purpose = purpose
        self.allowed_fields = self._get_allowed_fields()

    def _get_allowed_fields(self):
        """
        Define allowed fields based on purpose.

        Returns:
            List of allowed field names
        """
        # Define necessary fields for different purposes
        purpose_fields = {
            'product_recommendation': [
                'user_id', 'product_views', 'purchase_history',
                'product_ratings', 'session_duration'
            ],
            'fraud_detection': [
                'user_id', 'transaction_amount', 'transaction_location',
                'device_id', 'transaction_time', 'ip_address'
            ],
            'customer_support': [
                'user_id', 'support_tickets', 'product_owned',
                'issue_description'
            ],
            'marketing_analytics': [
                'user_id', 'campaign_clicks', 'email_opens',
                'conversion_events'
            ],
        }

        return purpose_fields.get(self.purpose, [])

    def filter_data(self, data):
        """
        Filter data to include only necessary fields.

        Args:
            data: DataFrame or dict

        Returns:
            Filtered data
        """
        if isinstance(data, pd.DataFrame):
            # Filter DataFrame columns
            available_allowed = [col for col in self.allowed_fields if col in data.columns]
            filtered = data[available_allowed]

            removed = set(data.columns) - set(available_allowed)
            if removed:
                print(f"\nData minimization applied:")
                print(f"  Purpose: {self.purpose}")
                print(f"  Kept fields: {available_allowed}")
                print(f"  Removed fields: {list(removed)}")

            return filtered

        elif isinstance(data, dict):
            # Filter dict keys
            filtered = {k: v for k, v in data.items() if k in self.allowed_fields}
            return filtered

    def validate_usage(self, requested_fields):
        """
        Validate that requested fields are allowed for stated purpose.

        Args:
            requested_fields: List of field names

        Returns:
            (is_valid, unauthorized_fields)
        """
        unauthorized = [f for f in requested_fields if f not in self.allowed_fields]

        is_valid = len(unauthorized) == 0

        if not is_valid:
            print(f"\n⚠️  Purpose limitation violation!")
            print(f"  Stated purpose: {self.purpose}")
            print(f"  Unauthorized fields: {unauthorized}")
            print(f"  These fields are not necessary for this purpose.")

        return is_valid, unauthorized


# Example 15: Test Data Minimization
print("\n" + "="*70)
print("DATA MINIMIZATION EXAMPLE")
print("="*70)

# Full user data (overly broad collection)
full_user_data = pd.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
    'age': [25, 35, 45],
    'ssn': ['123-45-6789', '987-65-4321', '456-78-9012'],  # Not needed!
    'product_views': [['A', 'B'], ['C', 'D'], ['E', 'F']],
    'purchase_history': [['A'], ['C', 'D'], ['E']],
})

print("\nFull collected data:")
print(full_user_data)

# Apply data minimization for product recommendations
minimizer = DataMinimizationFramework(purpose='product_recommendation')
minimal_data = minimizer.filter_data(full_user_data)

print("\nMinimized data (only necessary for product recommendation):")
print(minimal_data)

# Validate field usage
print("\n" + "-"*70)
is_valid, unauthorized = minimizer.validate_usage(['user_id', 'product_views', 'ssn'])
```

---

## 6. Privacy Impact Assessment

```python
# Example 16: Privacy Impact Assessment (PIA) Tool
class PrivacyImpactAssessment:
    """
    Conduct Privacy Impact Assessment for ML projects.

    Required by GDPR for high-risk processing.
    """

    def __init__(self, project_name):
        """
        Args:
            project_name: Name of ML project
        """
        self.project_name = project_name
        self.assessment = {
            'project_name': project_name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'risk_level': None,
            'findings': [],
            'recommendations': []
        }

    def assess_data_collection(self, data_types, volume, sensitive=False):
        """
        Assess data collection practices.

        Args:
            data_types: List of data types collected
            volume: Number of records
            sensitive: Whether data includes sensitive categories
        """
        print(f"\n[1/5] Assessing data collection...")

        findings = []
        risk_score = 0

        # Check for sensitive data
        if sensitive:
            findings.append("⚠️  Sensitive data detected (higher risk)")
            risk_score += 3

        # Check volume
        if volume > 100000:
            findings.append(f"⚠️  Large dataset ({volume:,} records)")
            risk_score += 2

        # Check data types
        high_risk_types = ['biometric', 'genetic', 'health', 'financial', 'location']
        detected_high_risk = [t for t in data_types if t.lower() in high_risk_types]

        if detected_high_risk:
            findings.append(f"⚠️  High-risk data types: {detected_high_risk}")
            risk_score += 2

        self.assessment['findings'].extend(findings)
        return risk_score

    def assess_processing_purpose(self, purpose, legal_basis):
        """
        Assess processing purpose and legal basis.

        Args:
            purpose: Description of processing purpose
            legal_basis: GDPR legal basis ('consent', 'contract', 'legitimate_interest', etc.)
        """
        print(f"[2/5] Assessing processing purpose...")

        findings = []
        risk_score = 0

        # Check if purpose is clearly defined
        if len(purpose) < 20:
            findings.append("⚠️  Purpose not clearly defined")
            risk_score += 1

        # Check legal basis
        if legal_basis == 'legitimate_interest':
            findings.append("⚠️  Legitimate interest requires balancing test")
            risk_score += 1

        self.assessment['findings'].extend(findings)
        return risk_score

    def assess_data_sharing(self, third_parties, cross_border=False):
        """
        Assess data sharing practices.

        Args:
            third_parties: List of third parties data is shared with
            cross_border: Whether data crosses borders
        """
        print(f"[3/5] Assessing data sharing...")

        findings = []
        risk_score = 0

        if third_parties:
            findings.append(f"⚠️  Data shared with {len(third_parties)} third parties")
            risk_score += 2

        if cross_border:
            findings.append("⚠️  Cross-border data transfer (check adequacy)")
            risk_score += 2

        self.assessment['findings'].extend(findings)
        return risk_score

    def assess_security_measures(self, encryption, access_control, audit_logging):
        """
        Assess security and privacy measures.

        Args:
            encryption: Whether data is encrypted
            access_control: Whether access controls are in place
            audit_logging: Whether activities are logged
        """
        print(f"[4/5] Assessing security measures...")

        findings = []
        risk_score = 0

        if not encryption:
            findings.append("❌ No encryption - HIGH RISK")
            risk_score += 3

        if not access_control:
            findings.append("❌ No access control - HIGH RISK")
            risk_score += 3

        if not audit_logging:
            findings.append("⚠️  No audit logging")
            risk_score += 1

        self.assessment['findings'].extend(findings)
        return risk_score

    def assess_individual_rights(self, access_api, deletion_api, portability_api):
        """
        Assess support for individual rights.

        Args:
            access_api: API for data access requests
            deletion_api: API for deletion requests
            portability_api: API for data portability
        """
        print(f"[5/5] Assessing individual rights...")

        findings = []
        risk_score = 0

        if not access_api:
            findings.append("⚠️  No API for data access requests")
            risk_score += 2

        if not deletion_api:
            findings.append("❌ No API for deletion requests - GDPR violation")
            risk_score += 3

        if not portability_api:
            findings.append("⚠️  No API for data portability")
            risk_score += 1

        self.assessment['findings'].extend(findings)
        return risk_score

    def generate_report(self):
        """Generate PIA report with recommendations."""

        print("\n" + "="*70)
        print(f"PRIVACY IMPACT ASSESSMENT: {self.project_name}")
        print("="*70)

        print(f"\nDate: {self.assessment['date']}")

        print(f"\nFindings:")
        for finding in self.assessment['findings']:
            print(f"  {finding}")

        # Calculate overall risk
        # (In real PIA, risk_score would be accumulated from all assessments)
        risk_level = "LOW" if len(self.assessment['findings']) < 3 else \
                     "MEDIUM" if len(self.assessment['findings']) < 6 else "HIGH"

        self.assessment['risk_level'] = risk_level

        print(f"\n{'='*70}")
        print(f"Overall Risk Level: {risk_level}")
        print(f"{'='*70}")

        # Generate recommendations
        recommendations = self._generate_recommendations()
        self.assessment['recommendations'] = recommendations

        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

        return self.assessment

    def _generate_recommendations(self):
        """Generate recommendations based on findings."""
        recommendations = []

        findings_text = ' '.join(self.assessment['findings'])

        if 'encryption' in findings_text.lower():
            recommendations.append("Implement end-to-end encryption for data at rest and in transit")

        if 'access control' in findings_text.lower():
            recommendations.append("Implement role-based access control (RBAC)")

        if 'deletion' in findings_text.lower():
            recommendations.append("Build API for GDPR deletion requests (right to be forgotten)")

        if 'sensitive data' in findings_text.lower():
            recommendations.append("Apply additional safeguards for sensitive data processing")

        if 'third parties' in findings_text.lower():
            recommendations.append("Establish data processing agreements with all third parties")

        if not recommendations:
            recommendations.append("Continue monitoring and periodic PIA reviews")

        return recommendations


# Example 17: Conduct PIA
pia = PrivacyImpactAssessment("Customer Churn Prediction Model")

pia.assess_data_collection(
    data_types=['demographics', 'purchase_history', 'location'],
    volume=500000,
    sensitive=False
)

pia.assess_processing_purpose(
    purpose="Predict customer churn to improve retention",
    legal_basis="legitimate_interest"
)

pia.assess_data_sharing(
    third_parties=['Analytics Provider', 'Cloud Storage'],
    cross_border=True
)

pia.assess_security_measures(
    encryption=True,
    access_control=True,
    audit_logging=False
)

pia.assess_individual_rights(
    access_api=True,
    deletion_api=False,
    portability_api=False
)

report = pia.generate_report()
```

---

## 7. Privacy Auditing

```python
# Example 18: Model Privacy Audit
class ModelPrivacyAuditor:
    """
    Audit trained models for privacy leakage.

    Tests for:
    - Membership inference attacks
    - Attribute inference
    - Model inversion
    """

    def __init__(self, model):
        """
        Args:
            model: Trained model to audit
        """
        self.model = model

    def membership_inference_attack(self, train_data, test_data):
        """
        Test if model memorized training data (membership inference).

        Attack: Check if model is more confident on training data.

        Args:
            train_data: Data model was trained on
            test_data: Data model was NOT trained on

        Returns:
            Attack accuracy (higher = more privacy leakage)
        """
        print("\n" + "="*70)
        print("MEMBERSHIP INFERENCE ATTACK")
        print("="*70)

        import torch.nn.functional as F

        self.model.eval()

        # Get confidences for training data
        train_confidences = []
        with torch.no_grad():
            for X_batch, y_batch in train_data:
                outputs = self.model(X_batch)
                probs = F.softmax(outputs, dim=1)

                # Confidence = probability of true class
                for i in range(len(y_batch)):
                    confidence = probs[i, y_batch[i]].item()
                    train_confidences.append(confidence)

        # Get confidences for test data
        test_confidences = []
        with torch.no_grad():
            for X_batch, y_batch in test_data:
                outputs = self.model(X_batch)
                probs = F.softmax(outputs, dim=1)

                for i in range(len(y_batch)):
                    confidence = probs[i, y_batch[i]].item()
                    test_confidences.append(confidence)

        # Attack: Classify as "in training set" if confidence > threshold
        threshold = np.median(train_confidences + test_confidences)

        train_correct = sum(c > threshold for c in train_confidences[:100])
        test_correct = sum(c <= threshold for c in test_confidences[:100])

        attack_accuracy = (train_correct + test_correct) / 200

        print(f"\nMembership Inference Attack Results:")
        print(f"  Attack accuracy: {attack_accuracy:.1%}")
        print(f"  Random guessing: 50%")

        if attack_accuracy > 0.6:
            print(f"  ⚠️  Model may be leaking membership information!")
        else:
            print(f"  ✓ Model appears resistant to membership inference")

        return attack_accuracy

    def attribute_inference_test(self, model, feature_idx, known_features, labels):
        """
        Test if sensitive attribute can be inferred from other features.

        Args:
            model: Model trained without feature_idx
            feature_idx: Index of sensitive feature to infer
            known_features: Features available to attacker
            labels: Model predictions

        Returns:
            Inference accuracy
        """
        print("\n" + "="*70)
        print("ATTRIBUTE INFERENCE TEST")
        print("="*70)

        # This is a simplified test
        # In practice, attacker trains a separate model to infer the attribute

        print(f"\nTesting if feature {feature_idx} can be inferred...")
        print(f"(Full implementation would train an attack model)")

        # Placeholder
        inference_accuracy = 0.55  # Just above random

        if inference_accuracy > 0.7:
            print(f"  ⚠️  Sensitive attribute may be inferable!")
        else:
            print(f"  ✓ Attribute appears protected")

        return inference_accuracy


# Example 19: Privacy Budget Tracking
class PrivacyBudgetTracker:
    """
    Track privacy budget (epsilon) consumption over time.

    Critical for differential privacy - ensure ε doesn't exceed threshold.
    """

    def __init__(self, total_epsilon=1.0):
        """
        Args:
            total_epsilon: Total privacy budget
        """
        self.total_epsilon = total_epsilon
        self.spent_epsilon = 0
        self.queries = []

    def spend(self, epsilon, query_description):
        """
        Spend privacy budget on a query.

        Args:
            epsilon: Privacy cost of query
            query_description: Description of query

        Returns:
            True if budget available, False otherwise
        """
        if self.spent_epsilon + epsilon > self.total_epsilon:
            print(f"\n❌ Privacy budget exceeded!")
            print(f"   Remaining: {self.total_epsilon - self.spent_epsilon:.3f}")
            print(f"   Requested: {epsilon:.3f}")
            return False

        self.spent_epsilon += epsilon
        self.queries.append({
            'epsilon': epsilon,
            'description': query_description,
            'timestamp': datetime.now()
        })

        remaining = self.total_epsilon - self.spent_epsilon

        print(f"\n✓ Privacy budget spent: ε={epsilon:.3f}")
        print(f"  Query: {query_description}")
        print(f"  Remaining budget: {remaining:.3f} / {self.total_epsilon:.3f}")

        if remaining < 0.1:
            print(f"  ⚠️  Privacy budget nearly exhausted!")

        return True

    def get_report(self):
        """Generate privacy budget report."""
        print("\n" + "="*70)
        print("PRIVACY BUDGET REPORT")
        print("="*70)

        print(f"\nTotal budget: ε = {self.total_epsilon}")
        print(f"Spent: ε = {self.spent_epsilon:.3f}")
        print(f"Remaining: ε = {self.total_epsilon - self.spent_epsilon:.3f}")

        print(f"\nQuery history ({len(self.queries)} queries):")
        for i, query in enumerate(self.queries, 1):
            print(f"  {i}. [{query['timestamp'].strftime('%H:%M:%S')}] "
                  f"ε={query['epsilon']:.3f} - {query['description']}")


# Example usage
budget_tracker = PrivacyBudgetTracker(total_epsilon=1.0)

budget_tracker.spend(0.1, "Count users by region")
budget_tracker.spend(0.2, "Average purchase amount")
budget_tracker.spend(0.3, "Top 10 products histogram")
budget_tracker.spend(0.5, "User segmentation analysis")  # Should fail

budget_tracker.get_report()
```

---

## Practice Exercises

### Exercise 1: Build a Differentially Private Analytics API

```python
"""
Exercise: Create an API that answers analytics queries with differential privacy.

Task:
1. Implement endpoints for common aggregations (count, sum, mean, histogram)
2. Add differential privacy to each endpoint
3. Track privacy budget across queries
4. Return error when budget exhausted
"""

def exercise_dp_analytics_api():
    # TODO: Implement
    # Hint: Use Flask/FastAPI + Opacus/custom DP mechanisms
    pass
```

### Exercise 2: Implement Federated Learning with Secure Aggregation

```python
"""
Exercise: Implement federated learning with cryptographic secure aggregation.

Task:
1. Create federated learning setup with 5+ clients
2. Implement secure aggregation using secret sharing or homomorphic encryption
3. Ensure server cannot see individual client updates
4. Measure accuracy vs centralized training
"""

def exercise_secure_federated_learning():
    # TODO: Implement
    pass
```

### Exercise 3: PII Scrubbing Pipeline

```python
"""
Exercise: Build a production PII scrubbing pipeline.

Task:
1. Support both text and structured data
2. Detect PII using multiple methods (regex, NER, column names)
3. Support multiple anonymization strategies
4. Generate audit logs of what was scrubbed
5. Provide option to pseudonymize (reversible) vs anonymize (irreversible)
"""

def exercise_pii_scrubbing_pipeline():
    # TODO: Implement
    pass
```

---

## Key Takeaways

1. **Privacy Regulations**:
   - **GDPR**: Right to erasure, data portability, privacy by design
   - **CCPA**: Right to know, delete, opt-out of sale
   - **HIPAA**: Safeguards for health information
   - Privacy Impact Assessment (PIA) required for high-risk processing

2. **Differential Privacy**:
   - Formal privacy guarantee: (ε, δ)-DP
   - Lower ε = stronger privacy but more noise
   - Laplace mechanism for numeric queries
   - Opacus for differentially private deep learning
   - Track privacy budget - it's exhaustible!

3. **Federated Learning**:
   - Train on decentralized data without centralization
   - FedAvg: Weighted averaging of client updates
   - Secure aggregation prevents server from seeing individual updates
   - Combine with differential privacy for stronger guarantees

4. **PII Protection**:
   - Use Presidio for text PII detection
   - Pattern matching + NER for structured data
   - Anonymization techniques: masking, hashing, generalization, removal
   - Distinguish PII (identifiable) from sensitive (private but not identifying)

5. **Data Minimization**:
   - Collect only data necessary for stated purpose
   - Purpose limitation: Use data only for stated purpose
   - Retention policies: Delete data when no longer needed
   - Document and enforce via technical controls

6. **Privacy Auditing**:
   - Membership inference: Can you tell if data was in training set?
   - Attribute inference: Can you infer sensitive attributes?
   - Model inversion: Can you reconstruct training data?
   - Regular audits are essential for compliance

7. **Production Best Practices**:
   - Privacy by design (not afterthought)
   - Multi-layer defense (DE + FL + anonymization)
   - Continuous monitoring and auditing
   - Document everything for compliance
   - Privacy-utility tradeoff is fundamental

---

## Further Reading

### Papers
1. **Differential Privacy**: "The Algorithmic Foundations of Differential Privacy" (Dwork & Roth, 2014)
2. **Federated Learning**: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
3. **Privacy Attacks**: "Membership Inference Attacks Against Machine Learning Models" (Shokri et al., 2017)
4. **Opacus**: "Opacus: User-Friendly Differential Privacy Library in PyTorch" (Yousefpour et al., 2021)
5. **GDPR**: "The General Data Protection Regulation (GDPR)" - Official EU regulation

### Tools & Libraries
- **Opacus**: Differential privacy for PyTorch
- **PySyft**: Federated learning and encrypted computation
- **CrypTen**: Privacy-preserving ML with secure computation
- **Presidio**: PII detection and anonymization
- **Google DP**: Differential privacy library

### Resources
- GDPR official text and guidelines
- NIST Privacy Framework
- Microsoft Privacy Guidelines
- Google's Differential Privacy course

### Related Modules
- **Module 10**: Production ML (monitoring, compliance)
- **Module 17 Lesson 1**: Bias and Fairness
- **Module 17 Lesson 2**: AI Safety
- **Module 17 Lesson 4**: Model Cards and Documentation

---

**Next Lesson**: Model Cards, Documentation, and Responsible Deployment (transparency, accountability, carbon tracking)
