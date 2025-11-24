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
