# Lesson 4: Model Cards, Documentation and Responsible Deployment

## Overview

This lesson covers the critical final step in responsible AI development: proper documentation, governance, and deployment practices. We'll learn how to create model cards, assess environmental impact, implement responsible deployment frameworks, and establish monitoring and incident response systems.

**Learning Objectives:**
- Create comprehensive model cards and datasheets
- Assess and reduce environmental impact of AI systems
- Implement responsible AI frameworks from industry leaders
- Build deployment checklists and governance processes
- Set up monitoring and incident response systems
- Conduct stakeholder analysis and impact assessments

**Prerequisites:**
- Module 10 (Deep Learning Fundamentals)
- Module 12 (Advanced Training)
- Module 15 (Advanced LLMs)
- Lessons 1-3 from this module

---

## 1. Model Cards and Documentation

### 1.1 What Are Model Cards?

Model cards are structured documents that provide transparency about ML models, including their intended use, performance characteristics, and limitations.

**Key Components:**
1. Model details (architecture, training data, developers)
2. Intended use and out-of-scope applications
3. Performance metrics across different groups
4. Ethical considerations and limitations
5. Training and evaluation data details
6. Quantitative analysis and fairness metrics

### 1.2 Creating Model Cards

```python
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
from datetime import datetime

@dataclass
class ModelDetails:
    """Core model information."""
    name: str
    version: str
    description: str
    architecture: str
    model_type: str  # 'classification', 'generation', 'regression', etc.
    developers: List[str]
    paper_citation: Optional[str] = None
    license: Optional[str] = None
    contact: Optional[str] = None

@dataclass
class IntendedUse:
    """Intended and out-of-scope uses."""
    primary_uses: List[str]
    primary_users: List[str]
    out_of_scope: List[str]

@dataclass
class TrainingData:
    """Information about training data."""
    dataset_name: str
    dataset_size: int
    data_sources: List[str]
    preprocessing: List[str]
    demographic_info: Optional[Dict[str, any]] = None

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_name: str
    metric_value: float
    slices: Optional[Dict[str, float]] = None  # Performance on different groups
    confidence_interval: Optional[tuple] = None

@dataclass
class EthicalConsiderations:
    """Ethical implications and limitations."""
    risks: List[str]
    mitigation_strategies: List[str]
    known_biases: List[str]
    limitations: List[str]
    use_cases_to_avoid: List[str]

class ModelCard:
    """Comprehensive model card generator."""

    def __init__(self):
        self.model_details: Optional[ModelDetails] = None
        self.intended_use: Optional[IntendedUse] = None
        self.training_data: Optional[TrainingData] = None
        self.performance_metrics: List[PerformanceMetric] = []
        self.ethical_considerations: Optional[EthicalConsiderations] = None
        self.created_at = datetime.now().isoformat()

    def set_model_details(self, details: ModelDetails):
        """Set model details."""
        self.model_details = details

    def set_intended_use(self, use: IntendedUse):
        """Set intended use information."""
        self.intended_use = use

    def set_training_data(self, data: TrainingData):
        """Set training data information."""
        self.training_data = data

    def add_performance_metric(self, metric: PerformanceMetric):
        """Add a performance metric."""
        self.performance_metrics.append(metric)

    def set_ethical_considerations(self, considerations: EthicalConsiderations):
        """Set ethical considerations."""
        self.ethical_considerations = considerations

    def generate_markdown(self) -> str:
        """Generate model card in Markdown format."""
        md = f"# Model Card: {self.model_details.name}\n\n"
        md += f"**Version:** {self.model_details.version}  \n"
        md += f"**Last Updated:** {self.created_at}  \n\n"

        # Model Details
        md += "## Model Details\n\n"
        md += f"**Description:** {self.model_details.description}\n\n"
        md += f"**Architecture:** {self.model_details.architecture}\n\n"
        md += f"**Type:** {self.model_details.model_type}\n\n"
        md += f"**Developers:** {', '.join(self.model_details.developers)}\n\n"

        if self.model_details.license:
            md += f"**License:** {self.model_details.license}\n\n"

        # Intended Use
        md += "## Intended Use\n\n"
        md += "**Primary Uses:**\n"
        for use in self.intended_use.primary_uses:
            md += f"- {use}\n"
        md += "\n**Primary Users:**\n"
        for user in self.intended_use.primary_users:
            md += f"- {user}\n"
        md += "\n**Out of Scope:**\n"
        for scope in self.intended_use.out_of_scope:
            md += f"- {scope}\n"
        md += "\n"

        # Training Data
        md += "## Training Data\n\n"
        md += f"**Dataset:** {self.training_data.dataset_name}\n\n"
        md += f"**Size:** {self.training_data.dataset_size:,} examples\n\n"
        md += "**Sources:**\n"
        for source in self.training_data.data_sources:
            md += f"- {source}\n"
        md += "\n"

        # Performance
        md += "## Performance\n\n"
        for metric in self.performance_metrics:
            md += f"**{metric.metric_name}:** {metric.metric_value:.4f}\n\n"
            if metric.slices:
                md += "Performance by group:\n"
                for group, value in metric.slices.items():
                    md += f"- {group}: {value:.4f}\n"
                md += "\n"

        # Ethical Considerations
        md += "## Ethical Considerations\n\n"
        md += "**Known Risks:**\n"
        for risk in self.ethical_considerations.risks:
            md += f"- {risk}\n"
        md += "\n**Limitations:**\n"
        for limitation in self.ethical_considerations.limitations:
            md += f"- {limitation}\n"
        md += "\n**Known Biases:**\n"
        for bias in self.ethical_considerations.known_biases:
            md += f"- {bias}\n"
        md += "\n"

        return md

    def save_json(self, filepath: str):
        """Save model card as JSON."""
        card_dict = {
            'model_details': asdict(self.model_details),
            'intended_use': asdict(self.intended_use),
            'training_data': asdict(self.training_data),
            'performance_metrics': [asdict(m) for m in self.performance_metrics],
            'ethical_considerations': asdict(self.ethical_considerations),
            'created_at': self.created_at
        }

        with open(filepath, 'w') as f:
            json.dump(card_dict, f, indent=2)

    def save_markdown(self, filepath: str):
        """Save model card as Markdown."""
        with open(filepath, 'w') as f:
            f.write(self.generate_markdown())

# Example: Creating a model card for a sentiment classifier
card = ModelCard()

# Set model details
card.set_model_details(ModelDetails(
    name="SentimentBERT",
    version="1.2.0",
    description="BERT-based sentiment classifier for product reviews",
    architecture="BERT-base with classification head",
    model_type="classification",
    developers=["AI Safety Team", "NLP Research Group"],
    license="Apache 2.0",
    contact="mlops@company.com"
))

# Set intended use
card.set_intended_use(IntendedUse(
    primary_uses=[
        "Analyzing customer feedback sentiment",
        "Prioritizing support tickets based on sentiment",
        "Monitoring brand perception in reviews"
    ],
    primary_users=[
        "Customer support teams",
        "Product managers",
        "Marketing analysts"
    ],
    out_of_scope=[
        "Medical or mental health assessment",
        "Legal document analysis",
        "High-stakes decision making without human review",
        "Detecting sarcasm or complex emotional states"
    ]
))

# Set training data
card.set_training_data(TrainingData(
    dataset_name="ProductReviews-v2",
    dataset_size=500000,
    data_sources=[
        "E-commerce platform reviews (2020-2023)",
        "Publicly available review datasets",
        "Synthetic data augmentation (10%)"
    ],
    preprocessing=[
        "Removed PII and identifying information",
        "Balanced classes via oversampling",
        "Text normalization and cleaning"
    ],
    demographic_info={
        "languages": {"English": 0.95, "Other": 0.05},
        "product_categories": {"Electronics": 0.4, "Clothing": 0.3, "Home": 0.3}
    }
))

# Add performance metrics
card.add_performance_metric(PerformanceMetric(
    metric_name="Accuracy",
    metric_value=0.912,
    slices={
        "Overall": 0.912,
        "Electronics": 0.925,
        "Clothing": 0.905,
        "Home": 0.908
    }
))

card.add_performance_metric(PerformanceMetric(
    metric_name="F1-Score",
    metric_value=0.908,
    slices={
        "Positive": 0.920,
        "Neutral": 0.885,
        "Negative": 0.910
    }
))

# Set ethical considerations
card.set_ethical_considerations(EthicalConsiderations(
    risks=[
        "May reinforce existing biases in product reviews",
        "Performance degrades on non-English text",
        "May misclassify sarcastic or nuanced sentiment"
    ],
    mitigation_strategies=[
        "Regular bias audits across product categories",
        "Human review for borderline cases",
        "Confidence thresholds for automated actions",
        "Continuous monitoring of performance across groups"
    ],
    known_biases=[
        "Better performance on mainstream product categories",
        "May reflect reviewer demographics in training data",
        "Lower accuracy on technical jargon"
    ],
    limitations=[
        "Trained only on product reviews",
        "Primarily English language",
        "Cannot detect complex emotions like disappointment vs anger",
        "May struggle with context-dependent sentiment"
    ],
    use_cases_to_avoid=[
        "Mental health assessment",
        "Legal proceedings",
        "Hiring decisions",
        "Any high-stakes automated decisions without oversight"
    ]
))

# Save the model card
card.save_markdown("model_card_sentimentbert.md")
card.save_json("model_card_sentimentbert.json")

print("Model card created successfully!")
print("\nPreview:")
print(card.generate_markdown()[:500] + "...")
```

### 1.3 Datasheets for Datasets

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class DatasetMotivation:
    """Why the dataset was created."""
    purpose: str
    creators: List[str]
    funding: Optional[str] = None

@dataclass
class DatasetComposition:
    """What's in the dataset."""
    instance_count: int
    instance_type: str
    splits: Dict[str, int]
    missing_data: bool
    missing_data_description: Optional[str] = None
    confidential_data: bool

@dataclass
class DatasetCollection:
    """How the data was collected."""
    collection_process: str
    collection_timeframe: str
    ethical_review: bool
    consent_obtained: bool

@dataclass
class DatasetPreprocessing:
    """What preprocessing was done."""
    preprocessing_steps: List[str]
    was_raw_data_saved: bool
    software_used: List[str]

class Datasheet:
    """Generate comprehensive dataset documentation."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.motivation: Optional[DatasetMotivation] = None
        self.composition: Optional[DatasetComposition] = None
        self.collection: Optional[DatasetCollection] = None
        self.preprocessing: Optional[DatasetPreprocessing] = None
        self.uses_recommended: List[str] = []
        self.uses_not_recommended: List[str] = []
        self.distribution_info: Dict = {}

    def set_motivation(self, motivation: DatasetMotivation):
        self.motivation = motivation

    def set_composition(self, composition: DatasetComposition):
        self.composition = composition

    def set_collection(self, collection: DatasetCollection):
        self.collection = collection

    def set_preprocessing(self, preprocessing: DatasetPreprocessing):
        self.preprocessing = preprocessing

    def generate_markdown(self) -> str:
        """Generate datasheet in Markdown."""
        md = f"# Datasheet: {self.dataset_name}\n\n"

        md += "## Motivation\n\n"
        md += f"**Purpose:** {self.motivation.purpose}\n\n"
        md += f"**Creators:** {', '.join(self.motivation.creators)}\n\n"

        md += "## Composition\n\n"
        md += f"**Total Instances:** {self.composition.instance_count:,}\n\n"
        md += f"**Instance Type:** {self.composition.instance_type}\n\n"
        md += "**Splits:**\n"
        for split, count in self.composition.splits.items():
            md += f"- {split}: {count:,}\n"
        md += "\n"

        md += "## Collection Process\n\n"
        md += f"{self.collection.collection_process}\n\n"
        md += f"**Timeframe:** {self.collection.collection_timeframe}\n\n"
        md += f"**Ethical Review:** {'Yes' if self.collection.ethical_review else 'No'}\n\n"
        md += f"**Consent Obtained:** {'Yes' if self.collection.consent_obtained else 'No'}\n\n"

        md += "## Preprocessing\n\n"
        for step in self.preprocessing.preprocessing_steps:
            md += f"- {step}\n"
        md += "\n"

        md += "## Recommended Uses\n\n"
        for use in self.uses_recommended:
            md += f"- {use}\n"
        md += "\n"

        md += "## Uses to Avoid\n\n"
        for use in self.uses_not_recommended:
            md += f"- {use}\n"
        md += "\n"

        return md

# Example
datasheet = Datasheet("CustomerReviews2023")

datasheet.set_motivation(DatasetMotivation(
    purpose="Create a diverse dataset for training sentiment analysis models",
    creators=["Data Science Team", "Ethics Board"],
    funding="Internal R&D budget"
))

datasheet.set_composition(DatasetComposition(
    instance_count=500000,
    instance_type="Product review text with sentiment labels",
    splits={"train": 400000, "validation": 50000, "test": 50000},
    missing_data=True,
    missing_data_description="5% of reviews missing product category",
    confidential_data=False
))

datasheet.set_collection(DatasetCollection(
    collection_process="Scraped from public e-commerce platforms with robots.txt compliance",
    collection_timeframe="January 2020 - December 2023",
    ethical_review=True,
    consent_obtained=True
))

datasheet.set_preprocessing(DatasetPreprocessing(
    preprocessing_steps=[
        "Removed all PII (emails, phone numbers, addresses)",
        "Language detection and filtering (English only)",
        "Duplicate removal based on text similarity",
        "Balanced sampling across product categories"
    ],
    was_raw_data_saved=True,
    software_used=["Python 3.9", "spaCy", "pandas"]
))

datasheet.uses_recommended = [
    "Training sentiment classifiers",
    "Benchmarking NLP models",
    "Research on review analysis"
]

datasheet.uses_not_recommended = [
    "Identifying individual reviewers",
    "Medical or legal applications",
    "High-stakes automated decision making"
]

print(datasheet.generate_markdown())
```

---

## 2. Environmental Impact and Carbon Footprint

### 2.1 Measuring Carbon Emissions

```python
import time
from typing import Optional, Dict
from dataclasses import dataclass
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
import torch
import numpy as np

@dataclass
class CarbonReport:
    """Carbon emissions report."""
    emissions_kg: float
    duration_hours: float
    energy_consumed_kwh: float
    carbon_intensity: float
    region: str

    def to_readable_string(self) -> str:
        """Convert to human-readable format."""
        return f"""
Carbon Emissions Report:
- Total Emissions: {self.emissions_kg:.4f} kg CO2
- Duration: {self.duration_hours:.2f} hours
- Energy Consumed: {self.energy_consumed_kwh:.4f} kWh
- Carbon Intensity: {self.carbon_intensity:.2f} gCO2/kWh
- Region: {self.region}
- Equivalent to: {self.emissions_kg * 2.20462:.2f} miles driven by average car
"""

class CarbonTracker:
    """Track carbon emissions during ML training."""

    def __init__(self, project_name: str, region: str = "usa"):
        self.project_name = project_name
        self.region = region
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir="./carbon_reports"
        )
        self.start_time = None

    def __enter__(self):
        """Start tracking."""
        self.tracker.start()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and generate report."""
        emissions = self.tracker.stop()
        duration = (time.time() - self.start_time) / 3600  # hours

        print(f"\n{'='*60}")
        print(f"Carbon Footprint Report: {self.project_name}")
        print(f"{'='*60}")
        print(f"Total Emissions: {emissions:.6f} kg CO2")
        print(f"Duration: {duration:.2f} hours")
        print(f"Equivalent to: {emissions * 2.20462:.4f} miles driven")
        print(f"{'='*60}\n")

# Example: Track carbon during model training
def train_with_carbon_tracking():
    """Example training with carbon tracking."""
    with CarbonTracker(project_name="sentiment_classifier_v2"):
        # Simulate model training
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )

        optimizer = torch.optim.Adam(model.parameters())

        # Training loop
        for epoch in range(5):
            for batch in range(100):
                # Simulated batch
                inputs = torch.randn(32, 768)
                labels = torch.randint(0, 3, (32,))

                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1} completed")

# Run training
# train_with_carbon_tracking()
```

### 2.2 Carbon-Efficient Training Strategies

```python
from typing import Tuple
import torch
import torch.nn as nn

class CarbonEfficientTrainer:
    """Training strategies to reduce carbon footprint."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

    def use_mixed_precision(self) -> torch.cuda.amp.GradScaler:
        """Enable mixed precision training (FP16)."""
        # Mixed precision reduces memory and energy
        scaler = torch.cuda.amp.GradScaler()
        print("✓ Mixed precision enabled (FP16)")
        return scaler

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

    def optimize_batch_size(self, dataset_size: int) -> int:
        """Find optimal batch size for efficiency."""
        # Larger batches = fewer iterations = less energy
        # But must fit in memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Heuristic: use 80% of GPU memory
            optimal_batch = min(256, int(gpu_memory / (4 * 1024**3)))  # Rough estimate
        else:
            optimal_batch = 32

        print(f"✓ Optimal batch size: {optimal_batch}")
        return optimal_batch

    def early_stopping_energy_aware(self, patience: int = 3) -> dict:
        """Early stopping to avoid wasted compute."""
        return {
            'patience': patience,
            'min_delta': 0.001,
            'mode': 'min',
            'restore_best_weights': True
        }

    def carbon_aware_scheduling(self, region: str) -> dict:
        """Schedule training during low-carbon times."""
        # In practice, check grid carbon intensity API
        low_carbon_hours = {
            'california': [10, 11, 12, 13, 14],  # Solar peak
            'texas': [11, 12, 13, 14, 15],
            'germany': [12, 13, 14, 15],
        }

        return {
            'preferred_hours': low_carbon_hours.get(region, []),
            'avoid_hours': [18, 19, 20, 21]  # Peak demand
        }

# Example usage
model = nn.Transformer(d_model=512, nhead=8)
trainer = CarbonEfficientTrainer(model)

scaler = trainer.use_mixed_precision()
trainer.enable_gradient_checkpointing()
batch_size = trainer.optimize_batch_size(100000)
early_stop_config = trainer.early_stopping_energy_aware()
schedule = trainer.carbon_aware_scheduling('california')

print(f"\nCarbon-efficient training configured:")
print(f"- Mixed precision: Enabled")
print(f"- Gradient checkpointing: Enabled")
print(f"- Batch size: {batch_size}")
print(f"- Early stopping patience: {early_stop_config['patience']}")
print(f"- Preferred training hours: {schedule['preferred_hours']}")
```

### 2.3 Reporting Carbon Metrics

```python
from typing import Dict, List
import json
from datetime import datetime

class ModelEnvironmentalReport:
    """Generate environmental impact report for models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.training_emissions: List[Dict] = []
        self.inference_emissions: Dict = {}

    def add_training_run(self, emissions_kg: float, duration_hours: float,
                        hardware: str, region: str):
        """Record training run emissions."""
        self.training_emissions.append({
            'timestamp': datetime.now().isoformat(),
            'emissions_kg': emissions_kg,
            'duration_hours': duration_hours,
            'hardware': hardware,
            'region': region
        })

    def set_inference_metrics(self, emissions_per_1k_requests: float,
                             avg_latency_ms: float):
        """Set inference emissions."""
        self.inference_emissions = {
            'emissions_per_1k_requests_kg': emissions_per_1k_requests,
            'avg_latency_ms': avg_latency_ms
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive environmental report."""
        total_training_emissions = sum(r['emissions_kg'] for r in self.training_emissions)
        total_training_hours = sum(r['duration_hours'] for r in self.training_emissions)

        # Comparisons for context
        miles_driven = total_training_emissions * 2.20462
        trees_needed = total_training_emissions / 21  # kg CO2 absorbed per tree per year

        report = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'training': {
                'total_emissions_kg': round(total_training_emissions, 4),
                'total_duration_hours': round(total_training_hours, 2),
                'number_of_runs': len(self.training_emissions),
                'runs': self.training_emissions
            },
            'inference': self.inference_emissions,
            'environmental_context': {
                'equivalent_miles_driven': round(miles_driven, 2),
                'trees_to_offset_one_year': round(trees_needed, 1),
                'smartphones_charged': round(total_training_emissions / 0.008, 0)
            }
        }

        return report

    def save_report(self, filepath: str):
        """Save report to file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nEnvironmental Impact Report: {self.model_name}")
        print(f"{'='*60}")
        print(f"Total Training Emissions: {report['training']['total_emissions_kg']:.4f} kg CO2")
        print(f"Equivalent to: {report['environmental_context']['equivalent_miles_driven']:.1f} miles driven")
        print(f"Trees needed to offset (1 year): {report['environmental_context']['trees_to_offset_one_year']:.1f}")
        print(f"{'='*60}\n")

# Example
env_report = ModelEnvironmentalReport("BERT-Sentiment-v2")

# Add training runs
env_report.add_training_run(
    emissions_kg=2.5,
    duration_hours=12.5,
    hardware="4x NVIDIA A100",
    region="us-west-2"
)

env_report.add_training_run(
    emissions_kg=0.8,
    duration_hours=4.2,
    hardware="1x NVIDIA A100",
    region="us-west-2"
)

# Set inference metrics
env_report.set_inference_metrics(
    emissions_per_1k_requests=0.001,
    avg_latency_ms=45
)

# Generate and save
env_report.save_report("environmental_report_bert_sentiment.json")
```

---

## 3. Responsible AI Frameworks

### 3.1 Microsoft Responsible AI Framework

```python
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

class RAIPrinciple(Enum):
    """Microsoft's Responsible AI Principles."""
    FAIRNESS = "fairness"
    RELIABILITY = "reliability_safety"
    PRIVACY = "privacy_security"
    INCLUSIVENESS = "inclusiveness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"

@dataclass
class RAIAssessment:
    """Assessment for a single principle."""
    principle: RAIPrinciple
    score: int  # 1-5
    evidence: List[str]
    gaps: List[str]
    action_items: List[str]

class ResponsibleAIFramework:
    """Microsoft-style Responsible AI assessment."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.assessments: Dict[RAIPrinciple, RAIAssessment] = {}

    def assess_principle(self, assessment: RAIAssessment):
        """Add an assessment for a principle."""
        self.assessments[assessment.principle] = assessment

    def calculate_overall_score(self) -> float:
        """Calculate overall RAI score."""
        if not self.assessments:
            return 0.0
        return sum(a.score for a in self.assessments.values()) / len(self.assessments)

    def generate_report(self) -> str:
        """Generate RAI assessment report."""
        report = f"# Responsible AI Assessment: {self.project_name}\n\n"
        report += f"**Overall Score:** {self.calculate_overall_score():.1f}/5.0\n\n"

        for principle in RAIPrinciple:
            if principle in self.assessments:
                assessment = self.assessments[principle]
                report += f"## {principle.value.replace('_', ' ').title()}\n\n"
                report += f"**Score:** {assessment.score}/5\n\n"

                report += "**Evidence:**\n"
                for evidence in assessment.evidence:
                    report += f"- ✓ {evidence}\n"
                report += "\n"

                if assessment.gaps:
                    report += "**Gaps:**\n"
                    for gap in assessment.gaps:
                        report += f"- ⚠ {gap}\n"
                    report += "\n"

                if assessment.action_items:
                    report += "**Action Items:**\n"
                    for action in assessment.action_items:
                        report += f"- → {action}\n"
                    report += "\n"

        return report

# Example assessment
rai = ResponsibleAIFramework("Sentiment Classifier")

# Fairness assessment
rai.assess_principle(RAIAssessment(
    principle=RAIPrinciple.FAIRNESS,
    score=4,
    evidence=[
        "Tested across multiple demographic groups",
        "Performance disparity < 5% across product categories",
        "Used fairness metrics (demographic parity, equalized odds)"
    ],
    gaps=[
        "Limited testing on non-English text",
        "No testing on users with different writing styles"
    ],
    action_items=[
        "Expand testing to include multi-lingual reviews",
        "Conduct bias audit with external experts"
    ]
))

# Privacy assessment
rai.assess_principle(RAIAssessment(
    principle=RAIPrinciple.PRIVACY,
    score=5,
    evidence=[
        "All PII removed during preprocessing",
        "Data anonymization applied",
        "GDPR compliance verified",
        "Regular privacy audits conducted"
    ],
    gaps=[],
    action_items=["Continue quarterly privacy audits"]
))

# Transparency assessment
rai.assess_principle(RAIAssessment(
    principle=RAIPrinciple.TRANSPARENCY,
    score=3,
    evidence=[
        "Model card published",
        "Basic documentation available"
    ],
    gaps=[
        "No user-facing explanations for predictions",
        "Limited interpretability features"
    ],
    action_items=[
        "Add SHAP explanations for predictions",
        "Create user-friendly documentation",
        "Build explanation dashboard"
    ]
))

print(rai.generate_report())
```

### 3.2 Google's PAIR Checklist

```python
from typing import List, Dict, Optional
from enum import Enum

class PAIRCategory(Enum):
    """Google PAIR checklist categories."""
    USER_NEEDS = "Understanding User Needs"
    DATA = "Data Collection & Evaluation"
    MENTAL_MODELS = "Mental Models"
    EXPLAINABILITY = "Explainability & Trust"
    FEEDBACK = "Feedback & Control"
    ERRORS = "Errors & Graceful Failure"

class PAIRChecklist:
    """Google PAIR (People + AI Research) checklist."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.items: Dict[PAIRCategory, List[Dict]] = {cat: [] for cat in PAIRCategory}

    def add_item(self, category: PAIRCategory, question: str,
                 status: str, notes: str = ""):
        """Add checklist item."""
        self.items[category].append({
            'question': question,
            'status': status,  # 'complete', 'in_progress', 'not_started'
            'notes': notes
        })

    def generate_checklist(self) -> str:
        """Generate checklist report."""
        report = f"# PAIR Checklist: {self.project_name}\n\n"

        for category in PAIRCategory:
            report += f"## {category.value}\n\n"

            items = self.items[category]
            if items:
                for item in items:
                    status_emoji = {
                        'complete': '✓',
                        'in_progress': '⏳',
                        'not_started': '○'
                    }.get(item['status'], '?')

                    report += f"{status_emoji} **{item['question']}**\n"
                    if item['notes']:
                        report += f"   - {item['notes']}\n"
                    report += "\n"
            else:
                report += "*No items added*\n\n"

        return report

# Example
pair = PAIRChecklist("Content Moderation System")

# User Needs
pair.add_item(
    PAIRCategory.USER_NEEDS,
    "Have you identified who will use the system?",
    "complete",
    "Primary users: content moderators, secondary: platform administrators"
)

pair.add_item(
    PAIRCategory.USER_NEEDS,
    "Do you understand the user's goals?",
    "complete",
    "Goal: Quickly identify and remove harmful content while minimizing false positives"
)

# Data
pair.add_item(
    PAIRCategory.DATA,
    "Is your training data representative of the real-world data?",
    "in_progress",
    "Current data from 2020-2023, need more recent examples"
)

pair.add_item(
    PAIRCategory.DATA,
    "Have you tested for biases in your data?",
    "complete",
    "Bias audit completed, found and mitigated language bias"
)

# Explainability
pair.add_item(
    PAIRCategory.EXPLAINABILITY,
    "Can users understand why the system made a decision?",
    "in_progress",
    "Added keyword highlighting, working on more detailed explanations"
)

pair.add_item(
    PAIRCategory.EXPLAINABILITY,
    "Are confidence scores shown to users?",
    "complete",
    "Confidence displayed for all predictions"
)

# Errors
pair.add_item(
    PAIRCategory.ERRORS,
    "How does the system handle errors gracefully?",
    "complete",
    "Fallback to human review for low-confidence predictions"
)

print(pair.generate_checklist())
```

---

## 4. Deployment Checklists and Governance

### 4.1 Pre-Deployment Checklist

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChecklistItem:
    """Individual checklist item."""
    category: str
    item: str
    completed: bool
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    notes: Optional[str] = None

class DeploymentChecklist:
    """Comprehensive pre-deployment checklist."""

    def __init__(self, project_name: str, version: str):
        self.project_name = project_name
        self.version = version
        self.items: List[ChecklistItem] = []
        self._initialize_checklist()

    def _initialize_checklist(self):
        """Initialize standard checklist items."""

        # Model Quality
        self.items.extend([
            ChecklistItem("Model Quality", "Model achieves target performance metrics", False),
            ChecklistItem("Model Quality", "Performance validated on held-out test set", False),
            ChecklistItem("Model Quality", "Cross-validation completed", False),
            ChecklistItem("Model Quality", "Performance across different subgroups verified", False),
        ])

        # Fairness & Bias
        self.items.extend([
            ChecklistItem("Fairness", "Bias audit completed", False),
            ChecklistItem("Fairness", "Fairness metrics calculated and acceptable", False),
            ChecklistItem("Fairness", "Tested on diverse demographic groups", False),
            ChecklistItem("Fairness", "Mitigation strategies implemented if needed", False),
        ])

        # Safety & Robustness
        self.items.extend([
            ChecklistItem("Safety", "Adversarial testing completed", False),
            ChecklistItem("Safety", "Edge case testing performed", False),
            ChecklistItem("Safety", "Failure modes documented", False),
            ChecklistItem("Safety", "Graceful degradation implemented", False),
        ])

        # Privacy & Security
        self.items.extend([
            ChecklistItem("Privacy", "PII detection and removal verified", False),
            ChecklistItem("Privacy", "Data access controls implemented", False),
            ChecklistItem("Privacy", "GDPR/CCPA compliance verified", False),
            ChecklistItem("Privacy", "Security audit completed", False),
        ])

        # Documentation
        self.items.extend([
            ChecklistItem("Documentation", "Model card created and reviewed", False),
            ChecklistItem("Documentation", "API documentation complete", False),
            ChecklistItem("Documentation", "User guide created", False),
            ChecklistItem("Documentation", "Known limitations documented", False),
        ])

        # Monitoring & Operations
        self.items.extend([
            ChecklistItem("Operations", "Monitoring dashboards configured", False),
            ChecklistItem("Operations", "Alerting thresholds set", False),
            ChecklistItem("Operations", "Logging implemented", False),
            ChecklistItem("Operations", "Incident response plan created", False),
        ])

        # Compliance & Legal
        self.items.extend([
            ChecklistItem("Compliance", "Legal review completed", False),
            ChecklistItem("Compliance", "Terms of use updated", False),
            ChecklistItem("Compliance", "Regulatory requirements verified", False),
        ])

    def complete_item(self, item_text: str, verified_by: str, notes: str = ""):
        """Mark an item as complete."""
        for item in self.items:
            if item.item == item_text:
                item.completed = True
                item.verified_by = verified_by
                item.verified_at = datetime.now().isoformat()
                item.notes = notes
                break

    def get_completion_percentage(self) -> float:
        """Get overall completion percentage."""
        if not self.items:
            return 0.0
        completed = sum(1 for item in self.items if item.completed)
        return (completed / len(self.items)) * 100

    def get_incomplete_items(self) -> List[ChecklistItem]:
        """Get all incomplete items."""
        return [item for item in self.items if not item.completed]

    def can_deploy(self) -> bool:
        """Check if model is ready for deployment."""
        return all(item.completed for item in self.items)

    def generate_report(self) -> str:
        """Generate checklist report."""
        report = f"# Deployment Checklist: {self.project_name} v{self.version}\n\n"
        report += f"**Completion:** {self.get_completion_percentage():.1f}%\n\n"
        report += f"**Ready to Deploy:** {'✓ YES' if self.can_deploy() else '✗ NO'}\n\n"

        # Group by category
        categories = {}
        for item in self.items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)

        for category, items in categories.items():
            report += f"## {category}\n\n"
            for item in items:
                status = "✓" if item.completed else "○"
                report += f"{status} {item.item}\n"
                if item.completed and item.verified_by:
                    report += f"   - Verified by: {item.verified_by}\n"
                    if item.notes:
                        report += f"   - Notes: {item.notes}\n"
                report += "\n"

        # Incomplete items
        incomplete = self.get_incomplete_items()
        if incomplete:
            report += "## Remaining Items\n\n"
            for item in incomplete:
                report += f"- [ ] {item.category}: {item.item}\n"

        return report

# Example usage
checklist = DeploymentChecklist("SentimentClassifier", "2.0")

# Complete some items
checklist.complete_item(
    "Model achieves target performance metrics",
    verified_by="data-scientist@company.com",
    notes="F1-score: 0.912, exceeds target of 0.90"
)

checklist.complete_item(
    "Performance validated on held-out test set",
    verified_by="data-scientist@company.com",
    notes="Test accuracy: 0.908"
)

checklist.complete_item(
    "Bias audit completed",
    verified_by="ethics-team@company.com",
    notes="Performance disparity < 5% across all groups"
)

checklist.complete_item(
    "Model card created and reviewed",
    verified_by="ml-engineer@company.com"
)

print(checklist.generate_report())
print(f"\nCan deploy: {checklist.can_deploy()}")
print(f"Completion: {checklist.get_completion_percentage():.1f}%")
```

### 4.2 AI Governance Framework

```python
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

class RiskLevel(Enum):
    """AI system risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class StakeholderRole(Enum):
    """Stakeholder roles in AI governance."""
    DEVELOPER = "developer"
    DATA_SCIENTIST = "data_scientist"
    ETHICS_BOARD = "ethics_board"
    LEGAL = "legal"
    PRODUCT = "product"
    SECURITY = "security"
    END_USER = "end_user"

@dataclass
class GovernanceReview:
    """Governance review record."""
    reviewer_role: StakeholderRole
    review_date: str
    approved: bool
    comments: str
    conditions: Optional[List[str]] = None

class AIGovernanceFramework:
    """AI governance and approval workflow."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.risk_level: Optional[RiskLevel] = None
        self.impact_areas: List[str] = []
        self.reviews: List[GovernanceReview] = []

    def assess_risk(self, use_case: str, user_impact: str,
                   data_sensitivity: str, automation_level: str) -> RiskLevel:
        """Assess risk level based on use case characteristics."""
        risk_score = 0

        # High-stakes use cases
        high_stakes = ['healthcare', 'legal', 'financial', 'hiring', 'criminal justice']
        if any(area in use_case.lower() for area in high_stakes):
            risk_score += 3

        # User impact
        impact_scores = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        risk_score += impact_scores.get(user_impact.lower(), 1)

        # Data sensitivity
        if 'pii' in data_sensitivity.lower() or 'sensitive' in data_sensitivity.lower():
            risk_score += 2

        # Automation level
        if 'fully automated' in automation_level.lower():
            risk_score += 2
        elif 'human in loop' in automation_level.lower():
            risk_score += 0

        # Determine risk level
        if risk_score >= 7:
            level = RiskLevel.CRITICAL
        elif risk_score >= 5:
            level = RiskLevel.HIGH
        elif risk_score >= 3:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        self.risk_level = level
        return level

    def get_required_approvals(self) -> List[StakeholderRole]:
        """Get required approvals based on risk level."""
        approvals = {
            RiskLevel.LOW: [
                StakeholderRole.DEVELOPER,
                StakeholderRole.DATA_SCIENTIST
            ],
            RiskLevel.MEDIUM: [
                StakeholderRole.DEVELOPER,
                StakeholderRole.DATA_SCIENTIST,
                StakeholderRole.PRODUCT,
                StakeholderRole.SECURITY
            ],
            RiskLevel.HIGH: [
                StakeholderRole.DEVELOPER,
                StakeholderRole.DATA_SCIENTIST,
                StakeholderRole.PRODUCT,
                StakeholderRole.SECURITY,
                StakeholderRole.ETHICS_BOARD,
                StakeholderRole.LEGAL
            ],
            RiskLevel.CRITICAL: [
                StakeholderRole.DEVELOPER,
                StakeholderRole.DATA_SCIENTIST,
                StakeholderRole.PRODUCT,
                StakeholderRole.SECURITY,
                StakeholderRole.ETHICS_BOARD,
                StakeholderRole.LEGAL,
                StakeholderRole.END_USER  # User testing required
            ]
        }

        return approvals.get(self.risk_level, [])

    def add_review(self, review: GovernanceReview):
        """Add a governance review."""
        self.reviews.append(review)

    def is_approved(self) -> bool:
        """Check if project has all required approvals."""
        required = self.get_required_approvals()
        approved_roles = {r.reviewer_role for r in self.reviews if r.approved}
        return all(role in approved_roles for role in required)

    def generate_governance_report(self) -> str:
        """Generate governance report."""
        report = f"# AI Governance Report: {self.project_name}\n\n"
        report += f"**Risk Level:** {self.risk_level.value.upper()}\n\n"

        report += "## Required Approvals\n\n"
        required = self.get_required_approvals()
        for role in required:
            approved = any(r.reviewer_role == role and r.approved for r in self.reviews)
            status = "✓" if approved else "○"
            report += f"{status} {role.value.replace('_', ' ').title()}\n"
        report += "\n"

        report += "## Review History\n\n"
        for review in self.reviews:
            status = "✓ APPROVED" if review.approved else "✗ REJECTED"
            report += f"**{review.reviewer_role.value.replace('_', ' ').title()}** - {status}\n"
            report += f"- Date: {review.review_date}\n"
            report += f"- Comments: {review.comments}\n"
            if review.conditions:
                report += "- Conditions:\n"
                for condition in review.conditions:
                    report += f"  - {condition}\n"
            report += "\n"

        report += f"**Overall Status:** {'✓ APPROVED FOR DEPLOYMENT' if self.is_approved() else '✗ PENDING APPROVALS'}\n"

        return report

# Example
governance = AIGovernanceFramework("Resume Screening AI")

# Assess risk
risk = governance.assess_risk(
    use_case="hiring - resume screening",
    user_impact="high",
    data_sensitivity="contains PII",
    automation_level="human in loop for final decisions"
)

print(f"Risk Level: {risk.value.upper()}")
print(f"\nRequired Approvals: {[r.value for r in governance.get_required_approvals()]}")

# Add reviews
governance.add_review(GovernanceReview(
    reviewer_role=StakeholderRole.DATA_SCIENTIST,
    review_date="2024-01-15",
    approved=True,
    comments="Model performance meets requirements, bias audit completed"
))

governance.add_review(GovernanceReview(
    reviewer_role=StakeholderRole.ETHICS_BOARD,
    review_date="2024-01-18",
    approved=True,
    comments="Approved with conditions",
    conditions=[
        "Must maintain human review for all hiring decisions",
        "Quarterly bias audits required",
        "Candidates must be informed of AI usage"
    ]
))

governance.add_review(GovernanceReview(
    reviewer_role=StakeholderRole.LEGAL,
    review_date="2024-01-20",
    approved=True,
    comments="Complies with EEOC guidelines"
))

print("\n" + governance.generate_governance_report())
```

---

## 5. Stakeholder Analysis and Impact Assessment

### 5.1 Stakeholder Mapping

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class StakeholderType(Enum):
    """Types of stakeholders."""
    PRIMARY_USER = "primary_user"
    SECONDARY_USER = "secondary_user"
    AFFECTED_PARTY = "affected_party"
    REGULATOR = "regulator"
    DEVELOPER = "developer"
    BUSINESS = "business"

class ImpactLevel(Enum):
    """Level of impact on stakeholder."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

@dataclass
class Stakeholder:
    """Individual stakeholder."""
    name: str
    type: StakeholderType
    description: str
    impact_level: ImpactLevel
    concerns: List[str]
    needs: List[str]
    engagement_strategy: str

class StakeholderAnalysis:
    """Analyze and map stakeholders for AI system."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.stakeholders: List[Stakeholder] = []

    def add_stakeholder(self, stakeholder: Stakeholder):
        """Add a stakeholder."""
        self.stakeholders.append(stakeholder)

    def get_high_impact_stakeholders(self) -> List[Stakeholder]:
        """Get stakeholders with high (positive or negative) impact."""
        return [s for s in self.stakeholders
                if s.impact_level in [ImpactLevel.VERY_POSITIVE, ImpactLevel.VERY_NEGATIVE]]

    def get_stakeholders_by_type(self, stakeholder_type: StakeholderType) -> List[Stakeholder]:
        """Get stakeholders of a specific type."""
        return [s for s in self.stakeholders if s.type == stakeholder_type]

    def generate_stakeholder_map(self) -> str:
        """Generate stakeholder analysis report."""
        report = f"# Stakeholder Analysis: {self.project_name}\n\n"

        # Group by type
        types = {}
        for stakeholder in self.stakeholders:
            if stakeholder.type not in types:
                types[stakeholder.type] = []
            types[stakeholder.type].append(stakeholder)

        for stakeholder_type, stakeholders in types.items():
            report += f"## {stakeholder_type.value.replace('_', ' ').title()}s\n\n"

            for stakeholder in stakeholders:
                impact_emoji = {
                    ImpactLevel.VERY_POSITIVE: "✓✓",
                    ImpactLevel.POSITIVE: "✓",
                    ImpactLevel.NEUTRAL: "○",
                    ImpactLevel.NEGATIVE: "✗",
                    ImpactLevel.VERY_NEGATIVE: "✗✗"
                }[stakeholder.impact_level]

                report += f"### {stakeholder.name} {impact_emoji}\n\n"
                report += f"**Description:** {stakeholder.description}\n\n"
                report += f"**Impact:** {stakeholder.impact_level.value.replace('_', ' ').title()}\n\n"

                report += "**Concerns:**\n"
                for concern in stakeholder.concerns:
                    report += f"- {concern}\n"
                report += "\n"

                report += "**Needs:**\n"
                for need in stakeholder.needs:
                    report += f"- {need}\n"
                report += "\n"

                report += f"**Engagement Strategy:** {stakeholder.engagement_strategy}\n\n"
                report += "---\n\n"

        return report

# Example: Content moderation system
analysis = StakeholderAnalysis("Content Moderation AI")

# Primary users - moderators
analysis.add_stakeholder(Stakeholder(
    name="Content Moderators",
    type=StakeholderType.PRIMARY_USER,
    description="Human moderators who use AI to assist in reviewing content",
    impact_level=ImpactLevel.VERY_POSITIVE,
    concerns=[
        "Job security - fear of being replaced",
        "Accuracy of AI recommendations",
        "Workload and mental health"
    ],
    needs=[
        "Clear explanations for AI decisions",
        "Easy override mechanism",
        "Reduced exposure to harmful content"
    ],
    engagement_strategy="Regular feedback sessions, training on AI usage, involve in evaluation"
))

# Affected parties - content creators
analysis.add_stakeholder(Stakeholder(
    name="Content Creators",
    type=StakeholderType.AFFECTED_PARTY,
    description="Users whose content may be flagged or removed",
    impact_level=ImpactLevel.NEGATIVE,
    concerns=[
        "False positives - legitimate content removed",
        "Lack of transparency in decisions",
        "No clear appeal process",
        "Bias in moderation"
    ],
    needs=[
        "Clear guidelines on acceptable content",
        "Timely notifications about removals",
        "Fair appeal process",
        "Consistent enforcement"
    ],
    engagement_strategy="Clear communication, appeal dashboard, regular policy updates"
))

# Affected parties - platform users
analysis.add_stakeholder(Stakeholder(
    name="Platform Users",
    type=StakeholderType.AFFECTED_PARTY,
    description="General users consuming content on platform",
    impact_level=ImpactLevel.POSITIVE,
    concerns=[
        "Over-moderation limiting free expression",
        "Harmful content slipping through"
    ],
    needs=[
        "Safe browsing experience",
        "Protection from harassment and harmful content",
        "Ability to report content"
    ],
    engagement_strategy="Transparent safety reports, user surveys, reporting tools"
))

# Business stakeholders
analysis.add_stakeholder(Stakeholder(
    name="Platform Leadership",
    type=StakeholderType.BUSINESS,
    description="Executive team concerned with platform health and growth",
    impact_level=ImpactLevel.VERY_POSITIVE,
    concerns=[
        "Cost of moderation",
        "Legal and regulatory compliance",
        "Brand reputation",
        "User growth and retention"
    ],
    needs=[
        "Scalable moderation solution",
        "Reduced legal risk",
        "Demonstrable safety improvements",
        "ROI metrics"
    ],
    engagement_strategy="Regular performance reports, risk assessments, cost-benefit analysis"
))

# Regulators
analysis.add_stakeholder(Stakeholder(
    name="Regulatory Bodies",
    type=StakeholderType.REGULATOR,
    description="Government agencies enforcing content safety laws",
    impact_level=ImpactLevel.NEUTRAL,
    concerns=[
        "Compliance with laws (COPPA, GDPR, etc.)",
        "Protection of minors",
        "Hate speech and illegal content"
    ],
    needs=[
        "Demonstrable compliance",
        "Audit trails",
        "Incident reporting",
        "Transparency reports"
    ],
    engagement_strategy="Regular compliance audits, transparency reports, liaison relationships"
))

print(analysis.generate_stakeholder_map())
```

### 5.2 Impact Assessment

```python
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

class ImpactDimension(Enum):
    """Dimensions of societal impact."""
    INDIVIDUAL_RIGHTS = "individual_rights"
    FAIRNESS = "fairness_equity"
    ENVIRONMENTAL = "environmental"
    ECONOMIC = "economic"
    SOCIAL = "social_cohesion"
    SAFETY = "safety_security"

@dataclass
class Impact:
    """Individual impact assessment."""
    dimension: ImpactDimension
    description: str
    severity: int  # 1-5
    likelihood: int  # 1-5
    mitigation: str

class ImpactAssessment:
    """Comprehensive impact assessment for AI system."""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.impacts: List[Impact] = []

    def add_impact(self, impact: Impact):
        """Add an impact."""
        self.impacts.append(impact)

    def get_risk_score(self, impact: Impact) -> int:
        """Calculate risk score (severity × likelihood)."""
        return impact.severity * impact.likelihood

    def get_high_risk_impacts(self) -> List[Impact]:
        """Get high-risk impacts (score >= 15)."""
        return [i for i in self.impacts if self.get_risk_score(i) >= 15]

    def generate_assessment(self) -> str:
        """Generate impact assessment report."""
        report = f"# Impact Assessment: {self.project_name}\n\n"

        # Sort by risk score
        sorted_impacts = sorted(self.impacts, key=self.get_risk_score, reverse=True)

        for impact in sorted_impacts:
            risk_score = self.get_risk_score(impact)
            risk_level = "🔴 CRITICAL" if risk_score >= 20 else "🟠 HIGH" if risk_score >= 15 else "🟡 MEDIUM" if risk_score >= 8 else "🟢 LOW"

            report += f"## {impact.dimension.value.replace('_', ' ').title()} - {risk_level}\n\n"
            report += f"**Risk Score:** {risk_score}/25 (Severity: {impact.severity}, Likelihood: {impact.likelihood})\n\n"
            report += f"**Description:** {impact.description}\n\n"
            report += f"**Mitigation:** {impact.mitigation}\n\n"
            report += "---\n\n"

        # Summary
        high_risk = len(self.get_high_risk_impacts())
        report += f"## Summary\n\n"
        report += f"- Total impacts assessed: {len(self.impacts)}\n"
        report += f"- High-risk impacts: {high_risk}\n"
        report += f"- Average risk score: {sum(self.get_risk_score(i) for i in self.impacts) / len(self.impacts):.1f}\n"

        return report

# Example: Facial recognition system
assessment = ImpactAssessment("Facial Recognition Access Control")

assessment.add_impact(Impact(
    dimension=ImpactDimension.INDIVIDUAL_RIGHTS,
    description="Collection and storage of biometric data raises privacy concerns. Users may not have meaningful choice in consent.",
    severity=5,
    likelihood=5,
    mitigation="Implement explicit opt-in consent, data minimization, encryption, and right to deletion"
))

assessment.add_impact(Impact(
    dimension=ImpactDimension.FAIRNESS,
    description="Facial recognition has documented accuracy disparities across demographic groups, potentially denying access to certain populations",
    severity=5,
    likelihood=4,
    mitigation="Regular bias testing, accuracy requirements across all groups, manual override process, diverse training data"
))

assessment.add_impact(Impact(
    dimension=ImpactDimension.SAFETY,
    description="False rejections could prevent authorized access in emergency situations",
    severity=4,
    likelihood=3,
    mitigation="Backup access methods (key card, PIN), regular accuracy monitoring, fail-safe design"
))

assessment.add_impact(Impact(
    dimension=ImpactDimension.SOCIAL,
    description="Surveillance capabilities could create chilling effect on behavior in monitored spaces",
    severity=3,
    likelihood=4,
    mitigation="Clear signage, limited retention periods, access controls on footage, transparency about usage"
))

assessment.add_impact(Impact(
    dimension=ImpactDimension.ENVIRONMENTAL,
    description="Continuous processing of video streams has energy cost",
    severity=2,
    likelihood=5,
    mitigation="Energy-efficient hardware, optimize model size, use motion detection to reduce processing"
))

print(assessment.generate_assessment())
```

---

## 6. Monitoring and Incident Response

### 6.1 Incident Response Plan

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentType(Enum):
    """Types of AI incidents."""
    BIAS = "bias_discrimination"
    PRIVACY = "privacy_breach"
    SAFETY = "safety_issue"
    PERFORMANCE = "performance_degradation"
    SECURITY = "security_breach"
    MISUSE = "system_misuse"

@dataclass
class Incident:
    """AI incident record."""
    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    description: str
    detected_at: str
    detected_by: str
    affected_users: Optional[int] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    resolved_at: Optional[str] = None

class IncidentResponsePlan:
    """Incident response system for AI systems."""

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.incidents: List[Incident] = []
        self.response_procedures: Dict[IncidentType, List[str]] = {}
        self._initialize_procedures()

    def _initialize_procedures(self):
        """Initialize standard response procedures."""

        self.response_procedures[IncidentType.BIAS] = [
            "1. Immediately flag affected predictions for human review",
            "2. Notify affected users if possible",
            "3. Analyze data for root cause",
            "4. Temporarily increase confidence threshold or route to human review",
            "5. Conduct emergency bias audit",
            "6. Implement mitigation (retraining, post-processing adjustment)",
            "7. Validate fix on test data",
            "8. Gradual rollout with monitoring"
        ]

        self.response_procedures[IncidentType.PRIVACY] = [
            "1. Immediately contain the breach",
            "2. Assess scope of data exposure",
            "3. Notify affected users within 72 hours (GDPR)",
            "4. Notify regulatory bodies if required",
            "5. Investigate root cause",
            "6. Implement technical fixes",
            "7. Review and update privacy controls",
            "8. Document incident and response"
        ]

        self.response_procedures[IncidentType.PERFORMANCE] = [
            "1. Verify the degradation with metrics",
            "2. Check for data drift or distribution shift",
            "3. If critical: rollback to previous version",
            "4. Analyze recent changes (data, model, infrastructure)",
            "5. Identify root cause",
            "6. Implement fix and test",
            "7. Deploy fix with monitoring",
            "8. Post-mortem and prevention measures"
        ]

    def report_incident(self, incident: Incident) -> str:
        """Report a new incident."""
        self.incidents.append(incident)

        response = f"\n{'='*60}\n"
        response += f"🚨 INCIDENT ALERT - {incident.severity.value.upper()}\n"
        response += f"{'='*60}\n"
        response += f"Incident ID: {incident.incident_id}\n"
        response += f"Type: {incident.incident_type.value}\n"
        response += f"Time: {incident.detected_at}\n"
        response += f"Description: {incident.description}\n"

        if incident.affected_users:
            response += f"Affected Users: {incident.affected_users}\n"

        response += f"\n{'='*60}\n"
        response += "RESPONSE PROCEDURES:\n"
        response += f"{'='*60}\n"

        procedures = self.response_procedures.get(incident.incident_type, ["No standard procedure defined"])
        for procedure in procedures:
            response += f"{procedure}\n"

        response += f"{'='*60}\n\n"

        return response

    def get_incident_stats(self) -> Dict:
        """Get incident statistics."""
        total = len(self.incidents)
        by_type = {}
        by_severity = {}

        for incident in self.incidents:
            # Count by type
            incident_type = incident.incident_type.value
            by_type[incident_type] = by_type.get(incident_type, 0) + 1

            # Count by severity
            severity = incident.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            'total_incidents': total,
            'by_type': by_type,
            'by_severity': by_severity,
            'resolved': sum(1 for i in self.incidents if i.resolved_at is not None)
        }

# Example usage
irp = IncidentResponsePlan("Content Moderation AI")

# Report bias incident
incident1 = Incident(
    incident_id="INC-2024-001",
    incident_type=IncidentType.BIAS,
    severity=IncidentSeverity.HIGH,
    description="Model showing 15% higher false positive rate for content from non-native English speakers",
    detected_at=datetime.now().isoformat(),
    detected_by="monitoring-system",
    affected_users=1200
)

print(irp.report_incident(incident1))

# Report performance degradation
incident2 = Incident(
    incident_id="INC-2024-002",
    incident_type=IncidentType.PERFORMANCE,
    severity=IncidentSeverity.MEDIUM,
    description="Model accuracy dropped from 0.92 to 0.85 over past week",
    detected_at=datetime.now().isoformat(),
    detected_by="automated-monitoring"
)

print(irp.report_incident(incident2))

# Get stats
stats = irp.get_incident_stats()
print(f"Total incidents: {stats['total_incidents']}")
print(f"By type: {stats['by_type']}")
print(f"By severity: {stats['by_severity']}")
```

---

## 7. Complete Production Example

```python
"""
Complete Responsible Deployment System
Combines all components: Model card, checklist, governance, monitoring
"""

from typing import Dict, List, Optional
import json

class ResponsibleDeploymentSystem:
    """End-to-end responsible AI deployment system."""

    def __init__(self, project_name: str, version: str):
        self.project_name = project_name
        self.version = version

        # Initialize components
        self.model_card = ModelCard()
        self.deployment_checklist = DeploymentChecklist(project_name, version)
        self.governance = AIGovernanceFramework(project_name)
        self.stakeholder_analysis = StakeholderAnalysis(project_name)
        self.impact_assessment = ImpactAssessment(project_name)
        self.incident_response = IncidentResponsePlan(project_name)
        self.env_report = ModelEnvironmentalReport(project_name)

    def configure_model_card(self, details: ModelDetails, use: IntendedUse,
                            data: TrainingData, ethics: EthicalConsiderations):
        """Configure the model card."""
        self.model_card.set_model_details(details)
        self.model_card.set_intended_use(use)
        self.model_card.set_training_data(data)
        self.model_card.set_ethical_considerations(ethics)

    def add_performance_metrics(self, metrics: List[PerformanceMetric]):
        """Add performance metrics to model card."""
        for metric in metrics:
            self.model_card.add_performance_metric(metric)

    def assess_risk_and_governance(self, use_case: str, user_impact: str,
                                   data_sensitivity: str, automation_level: str):
        """Assess risk and determine governance requirements."""
        risk = self.governance.assess_risk(
            use_case, user_impact, data_sensitivity, automation_level
        )
        print(f"\n✓ Risk assessed: {risk.value.upper()}")
        print(f"✓ Required approvals: {len(self.governance.get_required_approvals())}")
        return risk

    def run_pre_deployment_check(self) -> bool:
        """Run pre-deployment verification."""
        print("\n" + "="*60)
        print("PRE-DEPLOYMENT CHECK")
        print("="*60)

        # Check 1: Model card
        has_model_card = self.model_card.model_details is not None
        print(f"{'✓' if has_model_card else '✗'} Model card: {'Complete' if has_model_card else 'Missing'}")

        # Check 2: Deployment checklist
        checklist_complete = self.deployment_checklist.can_deploy()
        completion = self.deployment_checklist.get_completion_percentage()
        print(f"{'✓' if checklist_complete else '✗'} Deployment checklist: {completion:.0f}% complete")

        # Check 3: Governance approvals
        governance_approved = self.governance.is_approved()
        print(f"{'✓' if governance_approved else '✗'} Governance: {'Approved' if governance_approved else 'Pending'}")

        # Check 4: Impact assessment
        has_impact_assessment = len(self.impact_assessment.impacts) > 0
        high_risks = len(self.impact_assessment.get_high_risk_impacts())
        print(f"{'✓' if has_impact_assessment else '✗'} Impact assessment: {len(self.impact_assessment.impacts)} impacts, {high_risks} high-risk")

        # Check 5: Stakeholder analysis
        has_stakeholders = len(self.stakeholder_analysis.stakeholders) > 0
        print(f"{'✓' if has_stakeholders else '✗'} Stakeholder analysis: {len(self.stakeholder_analysis.stakeholders)} stakeholders")

        print("="*60)

        all_checks = all([
            has_model_card,
            checklist_complete,
            governance_approved,
            has_impact_assessment,
            has_stakeholders
        ])

        if all_checks:
            print("✓✓✓ ALL CHECKS PASSED - READY FOR DEPLOYMENT ✓✓✓")
        else:
            print("✗✗✗ DEPLOYMENT BLOCKED - COMPLETE REQUIREMENTS ✗✗✗")

        print("="*60 + "\n")
        return all_checks

    def generate_deployment_package(self, output_dir: str = "./deployment_docs"):
        """Generate all deployment documentation."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save model card
        self.model_card.save_markdown(f"{output_dir}/model_card.md")
        self.model_card.save_json(f"{output_dir}/model_card.json")

        # Save deployment checklist
        with open(f"{output_dir}/deployment_checklist.md", 'w') as f:
            f.write(self.deployment_checklist.generate_report())

        # Save governance report
        with open(f"{output_dir}/governance_report.md", 'w') as f:
            f.write(self.governance.generate_governance_report())

        # Save stakeholder analysis
        with open(f"{output_dir}/stakeholder_analysis.md", 'w') as f:
            f.write(self.stakeholder_analysis.generate_stakeholder_map())

        # Save impact assessment
        with open(f"{output_dir}/impact_assessment.md", 'w') as f:
            f.write(self.impact_assessment.generate_assessment())

        # Save environmental report
        self.env_report.save_report(f"{output_dir}/environmental_report.json")

        print(f"\n✓ Deployment package generated in: {output_dir}")
        print(f"  - model_card.md / .json")
        print(f"  - deployment_checklist.md")
        print(f"  - governance_report.md")
        print(f"  - stakeholder_analysis.md")
        print(f"  - impact_assessment.md")
        print(f"  - environmental_report.json\n")

# Example: Complete deployment workflow
print("="*60)
print("RESPONSIBLE AI DEPLOYMENT WORKFLOW")
print("="*60)

# Initialize system
system = ResponsibleDeploymentSystem(
    project_name="MedicalImageClassifier",
    version="1.0.0"
)

# Step 1: Configure model card
print("\n[Step 1] Configuring model card...")
system.configure_model_card(
    details=ModelDetails(
        name="ChestXRayClassifier",
        version="1.0.0",
        description="Deep learning model for detecting pneumonia in chest X-rays",
        architecture="ResNet-50",
        model_type="classification",
        developers=["Medical AI Team"],
        license="Proprietary - Medical Use Only"
    ),
    use=IntendedUse(
        primary_uses=["Assist radiologists in pneumonia detection"],
        primary_users=["Radiologists", "Emergency room physicians"],
        out_of_scope=[
            "Standalone diagnosis without physician review",
            "Pediatric patients (not validated)",
            "Non-chest X-ray images"
        ]
    ),
    data=TrainingData(
        dataset_name="ChestX-ray14 + Internal Hospital Data",
        dataset_size=150000,
        data_sources=["NIH ChestX-ray14", "Partner hospital EHR"],
        preprocessing=["DICOM to PNG conversion", "Normalization", "Augmentation"]
    ),
    ethics=EthicalConsiderations(
        risks=["Misdiagnosis if used without physician review", "Lower performance on underrepresented demographics"],
        mitigation_strategies=["Mandatory physician review", "Regular bias audits", "Continuous monitoring"],
        known_biases=["Trained primarily on adult patients", "Limited data from certain demographics"],
        limitations=["Cannot detect all lung pathologies", "Requires high-quality X-ray images"]
    )
)
print("✓ Model card configured")

# Step 2: Assess risk and governance requirements
print("\n[Step 2] Assessing risk...")
system.assess_risk_and_governance(
    use_case="Healthcare - Medical imaging diagnosis",
    user_impact="critical",
    data_sensitivity="Protected Health Information (PHI)",
    automation_level="Human in loop - physician must review"
)

# Step 3: Run pre-deployment check
print("\n[Step 3] Running pre-deployment check...")
ready = system.run_pre_deployment_check()

# Step 4: Generate documentation
if not ready:
    print("[Step 4] ⚠ Skipping documentation generation - deployment requirements not met")
else:
    print("\n[Step 4] Generating deployment package...")
    # system.generate_deployment_package()
    print("✓ Complete!")
```

---

## Practice Exercises

1. **Model Card Creation**: Create a comprehensive model card for a loan approval model, including all required sections with careful attention to fairness metrics across demographic groups.

2. **Carbon Tracking**: Implement carbon tracking for a large language model fine-tuning job. Calculate the environmental impact and suggest 3 optimizations to reduce the footprint.

3. **Deployment Checklist**: Build a custom deployment checklist for a facial recognition system used in airport security. What additional items beyond the standard checklist should be included?

4. **Stakeholder Analysis**: Conduct a stakeholder analysis for an AI-powered resume screening system. Identify at least 6 different stakeholders and their concerns.

5. **Impact Assessment**: Perform a comprehensive impact assessment for a predictive policing system. Identify impacts across all dimensions (individual rights, fairness, social, etc.).

6. **Incident Response**: Design an incident response plan for a content recommendation system that might amplify misinformation. Define severity levels and response procedures.

7. **Governance Framework**: Implement a governance approval workflow for a credit scoring model. What approvals are needed at each risk level?

8. **Environmental Reporting**: Generate an environmental report for training a GPT-3 sized model (175B parameters). Research typical carbon costs and create realistic estimates.

---

## Key Takeaways

1. **Documentation is Critical**: Model cards and datasheets provide essential transparency and enable informed decision-making by stakeholders.

2. **Environmental Impact Matters**: Training large models has real environmental costs. Measure, report, and optimize carbon emissions.

3. **Governance Prevents Problems**: Risk-based governance frameworks ensure appropriate oversight before deployment, catching issues early.

4. **Stakeholders Have Diverse Needs**: Different stakeholders (users, affected parties, regulators, business) have different concerns that must all be addressed.

5. **Deployment Requires Rigor**: Comprehensive checklists ensure nothing is missed before launching AI systems into production.

6. **Impact Assessment is Proactive**: Assessing potential harms before deployment allows for mitigation strategies rather than reactive responses.

7. **Incident Response Must Be Ready**: Have clear procedures in place for when things go wrong - and they will.

8. **Responsible AI is a System**: These components work together - documentation enables governance, impact assessment informs stakeholder engagement, etc.

9. **Continuous Process**: Responsible AI doesn't end at deployment. Ongoing monitoring, auditing, and stakeholder engagement are essential.

10. **Transparency Builds Trust**: Open documentation about capabilities, limitations, and risks builds trust with users and society.

---

## Further Reading

### Standards and Frameworks
- [Model Cards for Model Reporting (Google, 2019)](https://arxiv.org/abs/1810.03993)
- [Datasheets for Datasets (Microsoft, 2018)](https://arxiv.org/abs/1803.09010)
- [Microsoft Responsible AI Standard](https://www.microsoft.com/en-us/ai/responsible-ai)
- [Google PAIR Guidebook](https://pair.withgoogle.com/)
- [EU AI Act](https://artificialintelligenceact.eu/)

### Environmental Impact
- [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02243)
- [Carbon Emissions and Large Neural Network Training](https://arxiv.org/abs/1906.02243)
- [ML CO2 Impact Calculator](https://mlco2.github.io/impact/)

### Governance and Risk
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [ISO/IEC 42001 AI Management System](https://www.iso.org/standard/81230.html)
- [IEEE 7000 Series on AI Ethics](https://standards.ieee.org/industry-connections/ec/autonomous-systems.html)

### Tools and Libraries
- **codecarbon**: Python package for tracking carbon emissions
- **ml-co2-impact**: CO2 impact calculator
- **ABOUT ML**: Model documentation tool
- **Model Card Toolkit**: Google's model card generation tool

### Case Studies
- [Algorithmic Accountability Case Studies](https://ainowinstitute.org/)
- [Partnership on AI Case Study Library](https://partnershiponai.org/)
- [AI Incident Database](https://incidentdatabase.ai/)

### Regulations and Compliance
- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- HIPAA (Health Insurance Portability and Accountability Act)
- FCRA (Fair Credit Reporting Act)
- EU AI Act (proposed)

---

## Conclusion

This lesson covered the critical final steps in responsible AI development: proper documentation, environmental impact assessment, governance frameworks, stakeholder engagement, and incident response. These practices are not optional extras but essential components of professional AI engineering.

Creating comprehensive model cards and datasheets provides transparency. Measuring and reducing carbon emissions addresses environmental responsibility. Governance frameworks ensure appropriate oversight. Stakeholder analysis surfaces diverse concerns and needs. Impact assessments identify potential harms proactively. Deployment checklists prevent critical oversights. And incident response plans ensure quick, effective action when problems arise.

Together, these practices form a comprehensive responsible AI system that builds trust with users, meets regulatory requirements, addresses societal concerns, and ultimately leads to better, more sustainable AI systems.

**Next Steps:**
- Review Modules 13-16 for complete AI/ML knowledge
- Practice implementing these frameworks in your own projects
- Stay current with evolving regulations and best practices
- Contribute to the responsible AI community

**Congratulations on completing Module 17 and the entire Advanced AI Curriculum!** 🎉
