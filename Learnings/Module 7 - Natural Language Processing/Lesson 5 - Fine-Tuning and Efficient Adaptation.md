# Lesson 5: Fine-Tuning & Efficient Adaptation ⚡

**Module 7: Natural Language Processing | Lesson 5 of 7**

Learn to adapt large language models efficiently with LoRA, QLoRA, and PEFT!

---

## When to Fine-Tune vs Prompt? 🤔

```
Use Prompting when:
✅ Need quick solution
✅ Few examples (<100)
✅ Task is general
✅ No GPU access

Use Fine-Tuning when:
✅ Have labeled data (>1000)
✅ Need consistent format
✅ Domain-specific language
✅ Latency critical (smaller model)
✅ Privacy concerns (own model)
```

---

## 1. Full Fine-Tuning (Baseline) 📚

### Fine-Tune BERT for Classification

```python
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch

# Load model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset
dataset = load_dataset('imdb')

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].select(range(1000)),
    eval_dataset=tokenized_datasets['test'].select(range(200)),
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(f"Accuracy: {metrics['eval_accuracy']:.4f}")

# Save
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
```

### Memory Usage Analysis

```python
def get_model_size(model):
    \"\"\"Calculate model parameters and memory\"\"\"
    param_size = sum(p.numel() for p in model.parameters())
    param_size_mb = param_size * 4 / (1024 ** 2)  # 4 bytes per float32
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {param_size:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {param_size_mb:.2f} MB")
    
    return param_size, trainable

get_model_size(model)

# BERT-base: ~110M parameters, ~440MB
# Full fine-tuning updates ALL parameters!
```

---

## 2. LoRA: Low-Rank Adaptation 🎯

### How LoRA Works

**Key Idea:** Instead of updating full weight matrix W, add small low-rank matrices:

```
W' = W + BA

Where:
- W: Original weights (frozen)
- B: n × r matrix  
- A: r × m matrix
- r: rank (typically 4-64)

Parameters: r(n + m) << nm
```

### LoRA with PEFT Library

```python
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                    # Rank
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"],  # Which layers to adapt
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 109,483,778 || trainable%: 0.27%

# Train as before
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].select(range(1000)),
    eval_dataset=tokenized_datasets['test'].select(range(200)),
)

trainer.train()

# Save only LoRA weights (tiny!)
model.save_pretrained('./lora_weights')  # Only a few MB!
```

### LoRA for GPT-2 Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Prepare dataset (example: Python code)
texts = [
    "def factorial(n):\\n    if n == 0: return 1\\n    return n * factorial(n-1)",
    # ... more code examples
]

# Tokenize
def tokenize(examples):
    return tokenizer(examples, truncation=True, max_length=128)

# Train (simplified)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked
)

training_args = TrainingArguments(
    output_dir='./gpt2_lora',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
)

# ... training code ...

# Generate
model.eval()
prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0]))
```

---

## 3. QLoRA: Quantized LoRA 🚀

### Fine-Tune 7B Model on Single GPU!

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normalized float 4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,      # Nested quantization
)

# Load Llama-2-7B in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",            # Auto GPU assignment
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
# Llama-2-7B in 4-bit: ~5GB vs 28GB (float32)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qlora_llama",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=10,
    optim="paged_adamw_32bit",  # Efficient optimizer
)

# Train on instruction dataset
# ... (see full example in practice section)
```

### QLoRA vs LoRA vs Full Fine-Tuning

```python
# Comparison table
comparison = {
    'Method': ['Full FT', 'LoRA', 'QLoRA'],
    'Model Size': ['28GB', '28GB', '5GB'],
    'Trainable Params': ['7B', '~19M', '~19M'],
    'GPU Memory': ['80GB', '40GB', '16GB'],
    'Training Speed': ['1x', '1.2x', '0.7x'],
    'Quality': ['100%', '~99%', '~98%']
}

import pandas as pd
df = pd.DataFrame(comparison)
print(df)
```

---

## 4. Other PEFT Methods 🛠️

### Prefix Tuning

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # Number of prefix tokens
)

model = get_peft_model(model, prefix_config)
# Trains only prefix embeddings, keeps model frozen
```

### Prompt Tuning (Soft Prompts)

```python
from peft import PromptTuningConfig

prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify this text:",
)

model = get_peft_model(model, prompt_config)
```

### Adapter Layers

```python
from peft import AdapterConfig

adapter_config = AdapterConfig(
    r=16,  # Bottleneck dimension
)

model = get_peft_model(model, adapter_config)
```

---

## 5. Multi-Task Learning with Adapters 🎭

### Train Multiple Task-Specific Adapters

```python
# Train adapter for Task 1 (sentiment)
lora_config_sentiment = LoraConfig(r=8, target_modules=["query", "value"])
model_sentiment = get_peft_model(model, lora_config_sentiment)
# ... train on sentiment data ...
model_sentiment.save_pretrained('./adapters/sentiment')

# Train adapter for Task 2 (NER)
lora_config_ner = LoraConfig(r=8, target_modules=["query", "value"])
model_ner = get_peft_model(model, lora_config_ner)
# ... train on NER data ...
model_ner.save_pretrained('./adapters/ner')

# Use at inference: load different adapters
from peft import PeftModel

base_model = BertModel.from_pretrained('bert-base-uncased')

# Load sentiment adapter
model_sentiment = PeftModel.from_pretrained(base_model, './adapters/sentiment')

# Or load NER adapter
model_ner = PeftModel.from_pretrained(base_model, './adapters/ner')
```

---

## 6. Domain Adaptation 🏥

### Fine-Tune for Medical Domain

```python
# Medical corpus
medical_texts = [
    "Patient presents with acute myocardial infarction...",
    "Diagnosis: Type 2 diabetes mellitus with complications...",
    # ... thousands more
]

# Continued pre-training (MLM)
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Apply LoRA
lora_config = LoraConfig(r=16, target_modules=["query", "value"])
model = get_peft_model(model, lora_config)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./medical_bert', num_train_epochs=3),
    train_dataset=tokenized_medical_data,
    data_collator=data_collator,
)

trainer.train()

# Now this model understands medical terminology!
```

---

## Quick Reference 📖

### PEFT Methods Comparison

```
Method          Params    Memory    Speed    Use Case
─────────────────────────────────────────────────────────
Full FT         100%      Highest   Fast     Unlimited resources
LoRA            0.1-1%    Medium    Fast     Most common
QLoRA           0.1-1%    Lowest    Medium   Large models, limited GPU
Prefix          0.01%     Low       Fast     Few parameters needed
Prompt          0.001%    Lowest    Fast     Very limited resources
Adapters        1-5%      Medium    Medium   Multi-task scenarios
```

### Recommended Hyperparameters

```python
# LoRA for BERT-sized models (110M-340M params)
LoraConfig(r=8, lora_alpha=32)

# LoRA for larger models (1B-7B params)
LoraConfig(r=16, lora_alpha=32)

# QLoRA for very large models (7B-70B params)
LoraConfig(r=64, lora_alpha=16)
```

---

## Practice Exercises 🏋️

### Exercise 1: Compare Methods
Fine-tune BERT on same dataset using:
1. Full fine-tuning
2. LoRA
3. Prefix tuning
Compare accuracy, memory, and speed.

### Exercise 2: Domain Adaptation
Adapt a model to a specific domain (medical, legal, technical).

### Exercise 3: Multi-Task
Train multiple task-specific adapters and combine them.

---

## Key Takeaways 💡

1. **LoRA reduces trainable parameters** by 99% with minimal quality loss
2. **QLoRA enables fine-tuning 70B models** on consumer GPUs
3. **Adapters are small** (few MB vs GB) and composable
4. **Choose method based on constraints** (GPU, data, time)
5. **LoRA is production-ready** and widely adopted
6. **Can train multiple adapters** for different tasks
7. **Merge adapters with base model** for deployment

---

**Next:** [Lesson 6 - RAG & Vector Search →](Lesson%206%20-%20RAG%20and%20Vector%20Search.md)
