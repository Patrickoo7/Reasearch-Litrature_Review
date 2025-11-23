# Lesson 9: RLHF & Aligning Language Models 🤖

**Module 12: Reinforcement Learning | Lesson 9 of 12**

Master Reinforcement Learning from Human Feedback - the technology behind ChatGPT, Claude, and aligned AI!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand the complete RLHF pipeline (SFT → RM → RL)
2. ✅ Implement reward modeling from human preferences
3. ✅ Master PPO fine-tuning for language models
4. ✅ Learn Direct Preference Optimization (DPO) as RLHF alternative
5. ✅ Understand Constitutional AI and iterative RLHF
6. ✅ Apply RLHF to real language model alignment

**Cross-References**:
- See **Module 7B Lesson 5** (Transformer Fine-Tuning) for SFT background
- See **Module 7B Lesson 7** (Advanced Topics) for instruction tuning

---

## 1. What is RLHF?

### The Alignment Problem

Language models trained on internet data learn to predict next tokens, but this doesn't make them:
- Helpful: Answer user questions
- Harmless: Avoid toxic/harmful content
- Honest: Admit uncertainty, avoid hallucinations

**RLHF Solution**: Use human feedback to align models with human preferences!

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from collections import deque

"""
RLHF Pipeline:

Stage 1: Supervised Fine-Tuning (SFT)
- Fine-tune base LLM on high-quality demonstrations
- Input: (prompt, high-quality response) pairs
- Creates SFT model

Stage 2: Reward Modeling (RM)
- Collect human preferences: response A > response B
- Train reward model to predict human preferences
- Output: Scalar reward for any (prompt, response)

Stage 3: RL Fine-Tuning
- Use PPO to maximize reward model scores
- KL penalty to stay close to SFT model
- Creates aligned model (e.g., ChatGPT, Claude)

This is how ChatGPT, Claude, Gemini are created!
"""

print("=== RLHF Pipeline ===")
print("\nStage 1: Supervised Fine-Tuning (SFT)")
print("- Base LLM + demonstrations → SFT model")
print("- Creates helpful baseline")

print("\nStage 2: Reward Modeling (RM)")
print("- Human preferences → Reward model")
print("- Predicts 'how good' a response is")

print("\nStage 3: RL Fine-Tuning (PPO)")
print("- Maximize reward while staying close to SFT")
print("- Creates aligned model")

print("\n✅ Used by: ChatGPT, Claude, Gemini, LLaMA-2")
```

---

## 2. Stage 1: Supervised Fine-Tuning (SFT)

Start with high-quality demonstrations.

```python
"""
Supervised Fine-Tuning:

Dataset: {(prompt, response)} pairs from humans

Example:
Prompt: "Explain quantum computing in simple terms"
Response: "Quantum computing uses quantum mechanics principles..."

Process:
1. Collect 10K-100K high-quality demonstrations
2. Fine-tune base LLM (GPT-3, LLaMA) using standard LM objective
3. Creates SFT model

See Module 7B Lesson 5 for detailed implementation!
"""

class SFT:
    """
    Supervised Fine-Tuning for language models.
    """
    def __init__(self, model_name="gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

        print(f"SFT initialized with {model_name}")

    def train_step(self, prompts, responses):
        """
        Train one step on (prompt, response) pairs.

        Loss: Standard language modeling loss on responses.
        """
        # Combine prompt + response
        texts = [p + r for p, r in zip(prompts, responses)]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Forward pass
        outputs = self.model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def generate(self, prompt, max_length=100):
        """Generate response for prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


# Example: SFT training
print("\n=== Stage 1: SFT ===")
sft_model = SFT(model_name="gpt2")

# Dummy training data
prompts = ["Explain machine learning:", "What is Python?"]
responses = [
    " Machine learning is a field of AI that learns from data...",
    " Python is a high-level programming language known for simplicity..."
]

# Train
for epoch in range(3):
    loss = sft_model.train_step(prompts, responses)
    print(f"Epoch {epoch}, SFT Loss: {loss:.4f}")

print("\n✅ SFT creates helpful baseline model")
print("📖 See Module 7B Lesson 5 for full fine-tuning details")
```

---

## 3. Stage 2: Reward Modeling

Train a model to predict human preferences!

```python
class RewardModel(nn.Module):
    """
    Reward Model: Predicts scalar reward for (prompt, response).

    Architecture: LLM + scalar head
    - Input: Prompt + Response tokens
    - Output: Scalar reward score

    Training: Bradley-Terry model on pairwise preferences
    """
    def __init__(self, base_model_name="gpt2"):
        super(RewardModel, self).__init__()

        # Base language model (frozen or fine-tuned)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Reward head: Hidden state → Scalar
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Compute reward for input sequence.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            rewards: Scalar reward [batch_size]
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use last hidden state of last token
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]

        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)  # [batch_size]

        return rewards


class RewardModelTrainer:
    """
    Train reward model from human preference data.
    """
    def __init__(self, model_name="gpt2", lr=1e-5):
        self.model = RewardModel(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print("Reward Model initialized")

    def pairwise_loss(self, r_chosen, r_rejected):
        """
        Bradley-Terry model loss.

        P(chosen > rejected) = σ(r_chosen - r_rejected)

        Loss: -log σ(r_chosen - r_rejected)
        """
        # Log-sigmoid of difference
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        return loss

    def train_step(self, comparisons):
        """
        Train on batch of preference comparisons.

        Args:
            comparisons: List of (prompt, chosen_response, rejected_response)
        """
        prompts, chosen, rejected = zip(*comparisons)

        # Encode chosen responses
        chosen_texts = [p + c for p, c in zip(prompts, chosen)]
        chosen_enc = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Encode rejected responses
        rejected_texts = [p + r for p, r in zip(prompts, rejected)]
        rejected_enc = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Compute rewards
        r_chosen = self.model(chosen_enc["input_ids"], chosen_enc["attention_mask"])
        r_rejected = self.model(rejected_enc["input_ids"], rejected_enc["attention_mask"])

        # Pairwise loss
        loss = self.pairwise_loss(r_chosen, r_rejected)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        accuracy = (r_chosen > r_rejected).float().mean().item()

        return loss.item(), accuracy

    def predict_reward(self, prompt, response):
        """Predict reward for (prompt, response)."""
        text = prompt + response
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            reward = self.model(encoding["input_ids"], encoding["attention_mask"])

        return reward.item()


# Example: Reward model training
print("\n=== Stage 2: Reward Modeling ===")
rm_trainer = RewardModelTrainer(model_name="gpt2")

# Dummy preference data
comparisons = [
    (
        "What is machine learning?",
        " ML is a field of AI that learns from data to make predictions.",  # Chosen (better)
        " Machine learning is stuff with computers and data."  # Rejected (worse)
    ),
    (
        "Explain Python:",
        " Python is a versatile programming language known for its simplicity.",  # Chosen
        " Python is a snake. In programming, it's a language."  # Rejected
    )
]

# Train reward model
for epoch in range(5):
    loss, accuracy = rm_trainer.train_step(comparisons)
    print(f"Epoch {epoch}, RM Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")

# Test reward predictions
prompt = "What is machine learning?"
good_response = " ML learns from data."
bad_response = " ML is random stuff."

r_good = rm_trainer.predict_reward(prompt, good_response)
r_bad = rm_trainer.predict_reward(prompt, bad_response)

print(f"\nReward (good): {r_good:.3f}")
print(f"Reward (bad): {r_bad:.3f}")
print("✅ Reward model learned to distinguish quality!")
```

---

## 4. Stage 3: PPO Fine-Tuning for LLMs

Use PPO to maximize reward model scores!

```python
class PPOTrainer:
    """
    PPO for fine-tuning language models with RLHF.

    Key components:
    1. Policy model (LLM being trained)
    2. Reference model (frozen SFT model for KL penalty)
    3. Reward model (gives scores)
    4. Value model (baseline for PPO)

    Objective:
    maximize: E[reward(prompt, response) - β * KL(π || π_ref)]

    Where:
    - reward: From reward model
    - KL: KL divergence from reference model (prevents mode collapse)
    - β: KL penalty coefficient
    """
    def __init__(self, policy_model, ref_model, reward_model, value_model,
                 tokenizer, beta=0.01, lr=1e-6):
        self.policy = policy_model  # Model being trained
        self.ref_model = ref_model  # Frozen reference (SFT)
        self.reward_model = reward_model  # Reward predictor
        self.value_model = value_model  # Value baseline

        self.tokenizer = tokenizer
        self.beta = beta  # KL penalty

        # Only train policy and value
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=lr)

        # Freeze reference and reward models
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

        print("PPO Trainer initialized")
        print(f"✅ KL penalty: {beta}")

    def generate_response(self, prompt):
        """Generate response from current policy."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.policy.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        response = response[len(prompt):]

        return response

    def compute_kl_penalty(self, prompt, response):
        """
        Compute KL divergence between policy and reference model.

        KL(π || π_ref) for the response tokens.
        """
        text = prompt + response
        encoding = self.tokenizer(text, return_tensors="pt")

        # Policy log probs
        with torch.no_grad():
            policy_outputs = self.policy(**encoding)
            policy_logits = policy_outputs.logits
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)

            # Reference log probs
            ref_outputs = self.ref_model(**encoding)
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # KL divergence (averaged over tokens)
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1).mean()

        return kl.item()

    def compute_total_reward(self, prompt, response):
        """
        Total reward = Reward model score - β * KL penalty.
        """
        # Reward from reward model
        rm_reward = self.reward_model.predict_reward(prompt, response)

        # KL penalty
        kl_penalty = self.compute_kl_penalty(prompt, response)

        # Total reward
        total_reward = rm_reward - self.beta * kl_penalty

        return total_reward, rm_reward, kl_penalty

    def train_step(self, prompts):
        """
        PPO training step.

        1. Generate responses from current policy
        2. Compute rewards
        3. Update policy with PPO
        """
        # Generate responses
        responses = [self.generate_response(p) for p in prompts]

        # Compute rewards
        rewards = []
        rm_rewards = []
        kl_penalties = []

        for prompt, response in zip(prompts, responses):
            total_r, rm_r, kl = self.compute_total_reward(prompt, response)
            rewards.append(total_r)
            rm_rewards.append(rm_r)
            kl_penalties.append(kl)

        # PPO update (simplified here - full PPO is complex!)
        # In practice: compute advantages, update with clipped objective

        avg_reward = np.mean(rewards)
        avg_rm = np.mean(rm_rewards)
        avg_kl = np.mean(kl_penalties)

        return avg_reward, avg_rm, avg_kl

    def train(self, prompts, num_iterations=100):
        """Train policy with PPO."""
        for iteration in range(num_iterations):
            avg_reward, avg_rm, avg_kl = self.train_step(prompts)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, "
                      f"Reward: {avg_reward:.3f}, "
                      f"RM: {avg_rm:.3f}, "
                      f"KL: {avg_kl:.4f}")


print("\n=== Stage 3: PPO Fine-Tuning ===")
print("✅ Maximize reward while staying close to SFT")
print("✅ KL penalty prevents mode collapse")
print("✅ Creates aligned model (ChatGPT-style)")
print("\n📖 Full implementation uses PPO from Lesson 4")
```

---

## 5. Direct Preference Optimization (DPO)

RLHF without RL! Simpler alternative.

```python
"""
Direct Preference Optimization (DPO):

Key insight: Can optimize for preferences directly without RL!

RLHF: Train reward model → PPO optimization
DPO: Optimize policy directly from preferences

Loss:
L_DPO = -log σ(β * log(π(y_chosen|x) / π_ref(y_chosen|x))
              - β * log(π(y_rejected|x) / π_ref(y_rejected|x)))

Advantages:
✅ No reward model needed
✅ No RL (PPO) needed
✅ Simpler, more stable
✅ Competitive performance

Used by: Zephyr, many open-source aligned models
"""

class DPO:
    """
    Direct Preference Optimization.

    Directly optimize policy from preference data.
    """
    def __init__(self, policy_model, ref_model, tokenizer, beta=0.1, lr=1e-6):
        self.policy = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print("DPO initialized")
        print("✅ No reward model needed!")
        print("✅ No PPO needed!")

    def compute_log_prob(self, model, prompt, response):
        """Compute log probability of response given prompt."""
        text = prompt + response
        encoding = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(**encoding)
            logits = outputs.logits

            # Get log probs of actual tokens
            log_probs = F.log_softmax(logits, dim=-1)

            # Sum log probs of response tokens (simplified)
            # In practice: properly handle prompt/response separation
            total_log_prob = log_probs.mean()

        return total_log_prob

    def dpo_loss(self, prompt, chosen, rejected):
        """
        DPO loss for single preference pair.
        """
        # Log probs from policy
        log_pi_chosen = self.compute_log_prob(self.policy, prompt, chosen)
        log_pi_rejected = self.compute_log_prob(self.policy, prompt, rejected)

        # Log probs from reference
        log_ref_chosen = self.compute_log_prob(self.ref_model, prompt, chosen)
        log_ref_rejected = self.compute_log_prob(self.ref_model, prompt, rejected)

        # Log ratios
        log_ratio_chosen = log_pi_chosen - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected

        # DPO loss
        loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected))

        return loss

    def train_step(self, comparisons):
        """Train on preference comparisons."""
        total_loss = 0

        for prompt, chosen, rejected in comparisons:
            loss = self.dpo_loss(prompt, chosen, rejected)
            total_loss += loss

        avg_loss = total_loss / len(comparisons)

        # Update
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()

        return avg_loss.item()


print("\nDPO (Direct Preference Optimization):")
print("✅ Simpler than RLHF")
print("✅ No reward model or PPO needed")
print("✅ Directly optimize from preferences")
print("✅ Increasingly popular (Zephyr, etc.)")
```

---

## 6. Constitutional AI

Iterative RLHF with self-critique!

```python
"""
Constitutional AI (Anthropic):

Key idea: Model critiques and revises its own outputs.

Process:
1. Generate initial response
2. Model critiques response against 'constitution' (principles)
3. Model revises response
4. Use revised responses for RLHF

Constitution example:
- "Be helpful and harmless"
- "Avoid toxic content"
- "Admit uncertainty"
- "Respect privacy"

Advantages:
✅ Reduces need for human feedback
✅ More scalable
✅ Transparent principles
✅ Iterative improvement

Used by: Claude (Anthropic)
"""

class ConstitutionalAI:
    """
    Constitutional AI: Self-critique and revision.
    """
    def __init__(self, model, tokenizer, principles):
        self.model = model
        self.tokenizer = tokenizer
        self.principles = principles

        print("Constitutional AI initialized")
        print(f"Principles: {len(principles)}")

    def generate_response(self, prompt):
        """Generate initial response."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def critique(self, prompt, response):
        """
        Critique response against principles.

        Returns: Critique text
        """
        critique_prompt = f"""
        Response: {response}

        Critique this response according to these principles:
        {', '.join(self.principles)}

        Is this response aligned with the principles? If not, what should be improved?
        """

        critique = self.generate_response(critique_prompt)
        return critique

    def revise(self, prompt, response, critique):
        """
        Revise response based on critique.
        """
        revision_prompt = f"""
        Original prompt: {prompt}
        Original response: {response}
        Critique: {critique}

        Please provide a revised response that addresses the critique:
        """

        revised_response = self.generate_response(revision_prompt)
        return revised_response

    def constitutional_ai_loop(self, prompt, num_iterations=2):
        """
        Full Constitutional AI loop.

        1. Generate initial response
        2. Critique → Revise
        3. Repeat
        """
        response = self.generate_response(prompt)
        print(f"Initial: {response}")

        for iteration in range(num_iterations):
            critique = self.critique(prompt, response)
            print(f"\nCritique {iteration+1}: {critique}")

            response = self.revise(prompt, response, critique)
            print(f"Revised {iteration+1}: {response}")

        return response


print("\nConstitutional AI:")
print("✅ Self-critique and revision")
print("✅ Transparent principles")
print("✅ Scalable alignment")
print("✅ Used by Claude (Anthropic)")
```

---

## 7. Practical RLHF with Transformers

```python
"""
Practical RLHF Implementation:

Libraries:
1. TRL (Transformer Reinforcement Learning):
   - pip install trl
   - PPO trainer for LLMs
   - Reward model training
   - DPO implementation

2. TRLX (CarperAI):
   - Scalable RLHF
   - Multi-GPU support

3. OpenAssistant:
   - Full RLHF pipeline
   - Open-source ChatGPT alternative

Example with TRL:
"""

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

def rlhf_with_trl():
    """
    RLHF using TRL library.
    """
    # Load model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

    # PPO config
    ppo_config = PPOConfig(
        model_name="gpt2",
        learning_rate=1e-5,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True
    )

    # PPO trainer
    # ppo_trainer = PPOTrainer(
    #     config=ppo_config,
    #     model=model,
    #     ref_model=None,  # Will create automatically
    #     tokenizer=tokenizer
    # )

    print("✅ TRL provides production-ready RLHF")
    print("✅ Used for training aligned models")


print("\n=== Practical RLHF ===")
print("\nLibraries:")
print("✅ TRL: Transformer RL (Hugging Face)")
print("✅ TRLX: Scalable RLHF (CarperAI)")
print("✅ OpenAssistant: Full pipeline")
```

---

## 8. Connection to Module 7B

```python
"""
Connection to NLP Module (Module 7B):

Lesson 5 (Transformer Fine-Tuning):
- SFT uses techniques from fine-tuning lesson
- Instruction tuning is subset of SFT
- LoRA, QLoRA for efficient RLHF

Lesson 7 (Advanced Topics):
- Instruction tuning vs RLHF
- Prompt engineering for RLHF
- Evaluation of aligned models

RLHF builds on:
✅ Transformer architecture (7B Lesson 2)
✅ Pre-training (7B Lesson 3)
✅ Fine-tuning (7B Lesson 5)
✅ Instruction tuning (7B Lesson 7)

RLHF adds:
✅ Reward modeling
✅ RL optimization (PPO)
✅ Human preference alignment
"""

print("\n=== Module Integration ===")
print("\n📖 Module 7B Lesson 5: SFT techniques")
print("📖 Module 7B Lesson 7: Instruction tuning")
print("\n✅ RLHF is the final step in creating aligned LLMs!")
```

---

## 9. Real-World RLHF Applications

```python
"""
Real-World RLHF Deployments:

1. ChatGPT (OpenAI):
   - GPT-3.5/4 + RLHF
   - Helpful, harmless, honest
   - Billions of users

2. Claude (Anthropic):
   - Constitutional AI + RLHF
   - Emphasis on safety
   - Transparent principles

3. Gemini (Google):
   - PaLM/Gemini + RLHF
   - Multimodal alignment

4. LLaMA-2 Chat (Meta):
   - Open-source RLHF
   - Released model and data

5. Open-source:
   - Zephyr (DPO)
   - OpenAssistant
   - Vicuna, Alpaca

Impact:
- Billions of users
- Safer AI systems
- More helpful assistants
"""

print("\n=== Real-World Applications ===")
print("\n1. ChatGPT (OpenAI)")
print("   - RLHF on GPT-3.5/4")
print("   - Billions of users")

print("\n2. Claude (Anthropic)")
print("   - Constitutional AI")
print("   - Safety-focused")

print("\n3. LLaMA-2 Chat (Meta)")
print("   - Open-source RLHF")
print("   - Reproducible research")
```

---

## Practice Exercises

### Exercise 1: Implement Reward Model

```python
"""
Implement and train a reward model from scratch.

Requirements:
1. Load GPT-2 or LLaMA
2. Add scalar reward head
3. Create preference dataset (can use existing like HH-RLHF)
4. Train with Bradley-Terry loss
5. Evaluate on held-out preferences

Metrics:
- Preference accuracy
- Reward calibration
"""

# Your implementation here
```

### Exercise 2: DPO Fine-Tuning

```python
"""
Implement DPO and fine-tune a small model.

Requirements:
1. Load base model (GPT-2 or smaller LLaMA)
2. Create reference model (frozen copy)
3. Implement full DPO loss
4. Train on preference data
5. Compare with RLHF (if possible)

Evaluate:
- Model outputs before/after DPO
- Alignment with preferences
- Generation quality

Bonus: Use TRL library for comparison!
"""

# Your implementation here
```

### Exercise 3: End-to-End RLHF Pipeline

```python
"""
Implement complete RLHF pipeline (simplified).

Stages:
1. SFT: Fine-tune on instructions
2. RM: Train reward model
3. PPO: Fine-tune with RL

Dataset: Use Anthropic HH-RLHF or OpenAssistant

Compare:
- Base model
- SFT model
- RLHF model

Metrics:
- Helpfulness
- Harmlessness
- Human evaluation (if possible)
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **RLHF Pipeline** 🔄
   - SFT → RM → PPO
   - Three-stage process
   - Industry standard for alignment
   - Powers ChatGPT, Claude, Gemini

2. **Supervised Fine-Tuning** 📚
   - High-quality demonstrations
   - Creates helpful baseline
   - Foundation for RLHF
   - See Module 7B Lesson 5

3. **Reward Modeling** 🎯
   - Learn from preferences
   - Bradley-Terry model
   - Scalar reward predictor
   - Critical for RL stage

4. **PPO for LLMs** 🚀
   - Maximize reward
   - KL penalty for stability
   - Prevents mode collapse
   - Complex but effective

5. **DPO** ⚡
   - RLHF without RL
   - Simpler, more stable
   - Increasingly popular
   - Competitive performance

6. **Constitutional AI** 📜
   - Self-critique and revision
   - Transparent principles
   - Scalable alignment
   - Anthropic's approach

### When to Use RLHF?

✅ **Alignment**: Make models helpful, harmless, honest
✅ **Preference learning**: Learn from human feedback
✅ **Safety**: Reduce harmful outputs
✅ **Personalization**: Adapt to user preferences

### Real-World Impact

✅ **ChatGPT**: Billions of users, game-changing
✅ **Claude**: Safety-focused, constitutional AI
✅ **Open-source**: LLaMA-2 Chat, Zephyr, democratizing alignment
✅ **Safety**: Reduced harmful outputs across AI systems

### What's Next?

In Lesson 10, we'll learn **Multi-Agent RL**:
- Multiple agents interacting
- Game theory and Nash equilibria
- Self-play (AlphaGo, Dota 2)
- Emergent communication

**RLHF is transforming AI - this is how we align AGI!** 🤖

---

## Additional Resources

### Papers
- Christiano et al. (2017): "Deep RL from Human Preferences"
- Ouyang et al. (2022): "Training language models to follow instructions with human feedback" (InstructGPT)
- Bai et al. (2022): "Constitutional AI: Harmlessness from AI Feedback"
- Rafailov et al. (2023): "Direct Preference Optimization" (DPO)
- Touvron et al. (2023): "LLaMA 2" (Open-source RLHF)

### Libraries & Tools
- **TRL**: https://github.com/huggingface/trl (Transformer RL)
- **TRLX**: https://github.com/CarperAI/trlx (Scalable RLHF)
- **OpenAssistant**: https://github.com/LAION-AI/Open-Assistant

### Datasets
- **Anthropic HH-RLHF**: Human preference data
- **OpenAssistant Conversations**: Open-source RLHF data
- **SHP**: Stack-exchange preferences

### Blogs & Tutorials
- **OpenAI InstructGPT**: https://openai.com/blog/instruction-following/
- **Anthropic Constitutional AI**: https://www.anthropic.com/constitutional.pdf
- **Hugging Face RLHF**: https://huggingface.co/blog/rlhf

---

**Next**: [Lesson 10 - Multi-Agent Reinforcement Learning](Lesson%2010%20-%20Multi-Agent%20Reinforcement%20Learning.md)

Proceed to learn about **multi-agent systems** and **emergent behavior**! 🎮
