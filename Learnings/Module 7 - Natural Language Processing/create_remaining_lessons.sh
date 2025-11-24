#!/bin/bash

# Lesson 4: Prompt Engineering
cat > "Lesson 4 - Prompt Engineering and Few-Shot Learning.md" << 'L4EOF'
# Lesson 4: Prompt Engineering & Few-Shot Learning 🎯

**Module 7: Natural Language Processing | Lesson 4 of 7**

Master the art of prompting - the #1 skill for using LLMs in 2024!

---

## The Prompting Paradigm Shift 🚀

**Traditional NLP:**
```
Collect data → Label data → Train model → Deploy
(Months of work, expensive)
```

**Modern LLM Prompting:**
```
Write prompt → Test → Iterate
(Minutes to hours, cheap)
```

---

## 1. Zero-Shot Prompting 🎯

### Basic Classification

```python
import openai
# For this lesson, you'll need an API key
# openai.api_key = "your-key-here"

def zero_shot_classify(text, categories):
    prompt = f"""
Classify the following text into one of these categories: {', '.join(categories)}

Text: {text}

Category:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

# Test
text = "The product quality is excellent! Fast shipping too."
categories = ["positive", "negative", "neutral"]
result = zero_shot_classify(text, categories)
print(f"Sentiment: {result}")
```

### Structured Output

```python
def extract_info_zero_shot(text):
    prompt = f"""
Extract the following information from the text and return as JSON:
- company_name
- person_name
- amount
- date

Text: {text}

JSON:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

text = "Apple CEO Tim Cook announced a $1B investment on January 15, 2024"
result = extract_info_zero_shot(text)
print(result)
```

---

## 2. Few-Shot Prompting 📚

### In-Context Learning

```python
def few_shot_sentiment(text, examples):
    # Build prompt with examples
    prompt = "Classify sentiment as positive, negative, or neutral.\\n\\n"
    
    for ex_text, ex_label in examples:
        prompt += f"Text: {ex_text}\\nSentiment: {ex_label}\\n\\n"
    
    prompt += f"Text: {text}\\nSentiment:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

# Examples
examples = [
    ("I love this product!", "positive"),
    ("Terrible experience, very disappointed", "negative"),
    ("It's okay, nothing special", "neutral")
]

# Test
test_text = "The service exceeded my expectations"
result = few_shot_sentiment(test_text, examples)
print(f"Sentiment: {result}")
```

### Example Selection Strategy

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def select_best_examples(query, example_pool, k=3):
    \"\"\"Select most similar examples to query\"\"\"
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode
    query_emb = model.encode([query])
    example_texts = [ex[0] for ex in example_pool]
    example_embs = model.encode(example_texts)
    
    # Find most similar
    similarities = cosine_similarity(query_emb, example_embs)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    return [example_pool[i] for i in top_indices]

# Large pool of examples
example_pool = [
    ("This product is amazing! Best purchase ever.", "positive"),
    ("Absolute waste of money. Do not buy.", "negative"),
    ("It's fine, does what it says.", "neutral"),
    # ... many more ...
]

# Select best examples for query
query = "Incredible quality and fast delivery!"
selected_examples = select_best_examples(query, example_pool, k=3)

result = few_shot_sentiment(query, selected_examples)
```

---

## 3. Chain-of-Thought (CoT) Prompting 🧠

### Basic CoT

```python
def solve_with_cot(problem):
    prompt = f"""
Solve this problem step by step.

Problem: {problem}

Let's think step by step:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # GPT-4 better for reasoning
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# Math problem
problem = "A store has 15 apples. They sell 6 in the morning and 4 in the afternoon. Then they receive 12 more apples. How many apples do they have now?"

solution = solve_with_cot(problem)
print(solution)

# Expected output:
# Step 1: Start with 15 apples
# Step 2: Sell 6 in morning: 15 - 6 = 9 apples
# Step 3: Sell 4 in afternoon: 9 - 4 = 5 apples  
# Step 4: Receive 12 more: 5 + 12 = 17 apples
# Answer: 17 apples
```

### Few-Shot CoT

```python
def few_shot_cot(problem):
    prompt = \"\"\"
Solve these problems step by step.

Problem: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?
Solution: 
Step 1: Roger starts with 5 balls
Step 2: He buys 2 cans, each with 3 balls: 2 × 3 = 6 balls  
Step 3: Total: 5 + 6 = 11 balls
Answer: 11

Problem: {problem}
Solution:\"\"\"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

problem = "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"
print(few_shot_cot(problem))
```

---

## 4. Self-Consistency 🔄

### Multiple Reasoning Paths

```python
def self_consistency_solve(problem, n_samples=5):
    \"\"\"Generate multiple solutions and pick most common answer\"\"\"
    
    answers = []
    
    for _ in range(n_samples):
        prompt = f"Solve this problem step by step:\\n\\n{problem}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temp for diversity
            max_tokens=200
        )
        
        solution = response.choices[0].message.content
        
        # Extract final answer (simple regex)
        import re
        match = re.search(r'Answer: (\\d+)', solution)
        if match:
            answers.append(int(match.group(1)))
    
    # Return most common answer
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    
    return most_common, answers

problem = "Sarah has 7 apples. She gives 3 to her friend. Then she picks 5 more from her tree. How many does she have?"

final_answer, all_answers = self_consistency_solve(problem, n_samples=5)
print(f"All answers: {all_answers}")
print(f"Final answer (most common): {final_answer}")
```

---

## 5. Advanced Prompting Techniques 🚀

### Tree-of-Thought (ToT)

```python
def tree_of_thought_solve(problem, depth=3):
    \"\"\"Explore multiple reasoning branches\"\"\"
    
    def explore_branch(problem, path, depth):
        if depth == 0:
            return path
        
        # Generate possible next steps
        prompt = f\"\"\"
Given this problem: {problem}
Current reasoning path: {' -> '.join(path)}

What are 3 possible next reasoning steps? List them as:
1. ...
2. ...
3. ...\"\"\"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # Parse steps and evaluate each
        # (Simplified - full ToT more complex)
        
        return path
    
    return explore_branch(problem, [], depth)
```

### ReAct (Reasoning + Acting)

```python
def react_solve(question, tools):
    \"\"\"Combine reasoning with tool use\"\"\"
    
    prompt = f\"\"\"
Answer this question using the available tools.

Question: {question}

Available tools:
{chr(10).join(f"- {tool}" for tool in tools)}

Think out loud about what to do, then use tools as needed.

Format:
Thought: ...
Action: tool_name(args)
Observation: ...
... (repeat)
Final Answer: ...\"\"\"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

tools = ["Calculator", "Wikipedia", "Calendar"]
question = "What is the age difference between Barack Obama and Donald Trump?"

result = react_solve(question, tools)
print(result)
```

---

## 6. Prompt Templates with LangChain 📦

### Basic Templates

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Simple template
template = \"\"\"
Question: {question}

Answer: Let's think step by step.\"\"\"

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(question="What is 25 * 4?")
print(result)
```

### Few-Shot Template

```python
from langchain import FewShotPromptTemplate

examples = [
    {
        "question": "Who was the first president of the USA?",
        "answer": "George Washington was the first president of the USA."
    },
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    }
]

example_template = \"\"\"
Question: {question}
Answer: {answer}\"\"\"

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Answer the following questions:",
    suffix="Question: {input}\\nAnswer:",
    input_variables=["input"]
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt)
result = chain.run(input="What is the capital of Spain?")
print(result)
```

---

## 7. Prompt Optimization 🎨

### Iterative Refinement

```python
# Version 1: Basic
prompt_v1 = "Classify sentiment: {text}"

# Version 2: More specific
prompt_v2 = "Classify the sentiment of this text as positive, negative, or neutral: {text}"

# Version 3: With examples
prompt_v3 = \"\"\"
Classify sentiment as positive, negative, or neutral.

Examples:
- "I love it!" → positive
- "Terrible" → negative
- "It's okay" → neutral

Text: {text}
Sentiment:\"\"\"

# Version 4: With reasoning
prompt_v4 = \"\"\"
Analyze the sentiment of this text and explain your reasoning.

Text: {text}

Analysis:
1. Key phrases:
2. Overall tone:
3. Sentiment: [positive/negative/neutral]\"\"\"

# Test all versions
def test_prompts(text, prompts):
    for i, prompt in enumerate(prompts, 1):
        print(f"\\nVersion {i}:")
        # Run prompt...
```

### A/B Testing Prompts

```python
import random

def ab_test_prompts(texts, prompt_a, prompt_b):
    \"\"\"Compare two prompts\"\"\"
    
    results_a = []
    results_b = []
    
    for text in texts:
        # Randomly assign
        if random.random() < 0.5:
            result = run_prompt(prompt_a, text)
            results_a.append((text, result))
        else:
            result = run_prompt(prompt_b, text)
            results_b.append((text, result))
    
    # Compare metrics
    accuracy_a = evaluate(results_a)
    accuracy_b = evaluate(results_b)
    
    print(f"Prompt A accuracy: {accuracy_a:.2%}")
    print(f"Prompt B accuracy: {accuracy_b:.2%}")
    
    return "A" if accuracy_a > accuracy_b else "B"
```

---

## Quick Reference 📖

### Prompting Best Practices

```
1. Be specific and clear
2. Provide examples (few-shot)
3. Use step-by-step reasoning (CoT)
4. Specify output format
5. Set temperature appropriately:
   - 0 for factual/deterministic
   - 0.7-1.0 for creative
6. Iterate and test
7. Use system messages for context
```

### When to Use Each Technique

```
Zero-shot        → Simple tasks, clear instructions
Few-shot         → Need specific format/style
Chain-of-Thought → Reasoning/math problems
Self-Consistency → Need high accuracy
Tree-of-Thought  → Complex planning
ReAct            → Need tool use
```

---

## Practice Exercises 🏋️

### Exercise 1: Build Classifier
Create a zero-shot classifier for customer support tickets (urgent/normal/low priority).

### Exercise 2: CoT Math Solver
Build a chain-of-thought math problem solver with verification.

### Exercise 3: Optimize Prompt
Take a basic prompt and improve it through 5 iterations.

---

## Key Takeaways 💡

1. **Prompting is the new programming** for LLMs
2. **Few-shot > Zero-shot** for complex/specific tasks
3. **Chain-of-Thought** dramatically improves reasoning
4. **Self-consistency** averages multiple reasoning paths
5. **Example selection matters** - use semantic similarity
6. **Iterate and test** prompts like code
7. **Temperature controls randomness** (0=deterministic, 1=creative)

---

**Next:** [Lesson 5 - Fine-Tuning & Efficient Adaptation →](Lesson%205%20-%20Fine-Tuning%20and%20Efficient%20Adaptation.md)
L4EOF

echo "✓ Lesson 4 created"

# Lesson 5: Fine-Tuning
cat > "Lesson 5 - Fine-Tuning and Efficient Adaptation.md" << 'L5EOF'
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
L5EOF

echo "✓ Lesson 5 created"

# Lessons 6 and 7 will be created next
echo "Creating Lessons 6 and 7..."

