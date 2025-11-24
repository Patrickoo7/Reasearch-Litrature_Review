# Lesson 1: NLP Fundamentals & Modern Tokenization 🔤

**Module 7: Natural Language Processing | Lesson 1 of 7**

Master the foundation of modern NLP - from text preprocessing to state-of-the-art tokenization methods!

---

## Why NLP Matters in 2024 🌟

**Natural Language Processing is everywhere:**
- ChatGPT, Claude, Gemini (LLM assistants)
- Google Search, recommendation systems
- Customer support automation
- Code generation (GitHub Copilot)
- Translation, summarization, sentiment analysis

**Key challenge:** Computers don't understand text - they need numbers!

---

## 1. Text Preprocessing Basics 🧹

### The Preprocessing Pipeline

```
Raw Text → Cleaning → Tokenization → Normalization → Ready for ML
```

### Basic Cleaning

```python
import re
import string

def clean_text(text):
    """Basic text cleaning"""
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation (optional)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

# Example
raw_text = "Check out https://example.com! Email me@email.com. <b>Great</b> product!!!"
cleaned = clean_text(raw_text)
print(cleaned)  # "check out email great product"
```

### Tokenization with NLTK

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Natural Language Processing is amazing! It powers ChatGPT."

# Word tokenization
words = word_tokenize(text)
print(words)  # ['Natural', 'Language', 'Processing', 'is', 'amazing', '!', 'It', 'powers', 'ChatGPT', '.']

# Sentence tokenization
sentences = sent_tokenize(text)
print(sentences)  # ['Natural Language Processing is amazing!', 'It powers ChatGPT.']
```

### Stopword Removal

```python
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

words = word_tokenize("This is an example sentence demonstrating stopword removal")
filtered = [w for w in words if w.lower() not in stop_words]

print("Original:", words)
print("Filtered:", filtered)  # ['example', 'sentence', 'demonstrating', 'stopword', 'removal']
```

### Stemming vs Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "better", "cats", "feet"]

print("Word\t\tStemmed\t\tLemmatized")
print("-" * 50)
for word in words:
    stemmed = stemmer.stem(word)
    lemmatized = lemmatizer.lemmatize(word, pos='v')  # v = verb
    print(f"{word}\t\t{stemmed}\t\t{lemmatized}")

# Output:
# running      run         run
# runs         run         run
# ran          ran         run
# better       better      well
# cats         cat         cat
# feet         feet        foot
```

**Key Difference:**
- **Stemming:** Chops off word endings (crude, fast) → "running" → "run"
- **Lemmatization:** Uses dictionary + grammar (accurate, slower) → "better" → "well"

### Advanced Preprocessing with spaCy

```python
import spacy

# Load English model
nlp = spacy.load('en_core_web_sm')

text = "Apple Inc. is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# Tokenization + lemmatization + POS tagging
for token in doc:
    print(f"{token.text:12} {token.lemma_:12} {token.pos_:6} {token.is_stop}")

# Remove stopwords and punctuation
cleaned_tokens = [token.lemma_ for token in doc
                  if not token.is_stop and not token.is_punct]
print("\nCleaned:", cleaned_tokens)  # ['Apple', 'Inc.', 'look', 'buy', 'U.K.', 'startup', '$', '1', 'billion']
```

---

## 2. Evolution of Tokenization 📈

### Problem with Word-Level Tokenization

```python
# Word-level issues
vocab = ["run", "running", "ran", "runner"]

# Problems:
# 1. Vocabulary explosion (millions of words)
# 2. Out-of-vocabulary (OOV) words
# 3. Rare words handled poorly
# 4. Can't handle new words

# Example: "ChatGPT" would be <UNK> (unknown)
```

### Why Subword Tokenization?

**Advantages:**
- ✅ Smaller vocabulary (30K-50K vs millions)
- ✅ No OOV problem
- ✅ Handles rare/new words
- ✅ Works across languages

**Trade-off:** Sequences become longer (more tokens)

---

## 3. Byte-Pair Encoding (BPE) 🔥

**Used by:** GPT-2, GPT-3, GPT-4, RoBERTa, BART

### How BPE Works

```
1. Start with character-level vocabulary
2. Find most frequent pair of tokens
3. Merge them into new token
4. Repeat until desired vocabulary size
```

### Example: BPE Training

```python
# Simple BPE example
corpus = ["low", "lower", "newest", "widest"]

# Step 1: Character-level + word boundary marker
# ["l o w </w>", "l o w e r </w>", "n e w e s t </w>", "w i d e s t </w>"]

# Step 2: Count pairs
# Most frequent: ('e', 's') appears in "newest", "widest"

# Step 3: Merge
# Vocabulary: ['l', 'o', 'w', 'e', 'r', 'n', 'i', 'd', 's', 't', '</w>', 'es']

# Continue merging...
```

### Train BPE from Scratch

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The five boxing wizards jump quickly",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump"
] * 100  # Repeat for better training

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=1000
)

tokenizer.train_from_iterator(corpus, trainer)

# Test tokenization
output = tokenizer.encode("The quick brown fox")
print("Tokens:", output.tokens)
print("IDs:", output.ids)

# Decode
decoded = tokenizer.decode(output.ids)
print("Decoded:", decoded)

# Save
tokenizer.save("bpe_tokenizer.json")
```

### Visualize BPE Merges

```python
# See merge operations
merges = tokenizer.model.get_vocab()
print(f"Vocabulary size: {len(merges)}")
print("\nFirst 20 tokens:")
for token, idx in sorted(merges.items(), key=lambda x: x[1])[:20]:
    print(f"{idx:4d}: {token}")
```

### GPT-2 Tokenizer (BPE)

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "ChatGPT is an amazing language model!"

# Encode
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# ['Chat', 'G', 'PT', 'Ġis', 'Ġan', 'Ġamazing', 'Ġlanguage', 'Ġmodel', '!']
# Note: Ġ = space character

ids = tokenizer.encode(text)
print("IDs:", ids)

# Decode
decoded = tokenizer.decode(ids)
print("Decoded:", decoded)

# Check vocabulary size
print(f"Vocabulary size: {tokenizer.vocab_size}")  # 50,257
```

---

## 4. WordPiece (BERT Tokenization) 📚

**Used by:** BERT, DistilBERT, ELECTRA

### How WordPiece Differs from BPE

- **BPE:** Merges most *frequent* pairs
- **WordPiece:** Merges pairs that *maximize likelihood* on training corpus

### BERT Tokenizer

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "ChatGPT is revolutionizing natural language processing"

# Tokenize
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# ['chat', '##gp', '##t', 'is', 'revolution', '##izing', 'natural', 'language', 'processing']
# Note: ## = continuation of previous word

ids = tokenizer.encode(text)
print("IDs:", ids)

# Special tokens
print("\nSpecial tokens:")
print(f"[CLS]: {tokenizer.cls_token_id}")  # 101
print(f"[SEP]: {tokenizer.sep_token_id}")  # 102
print(f"[PAD]: {tokenizer.pad_token_id}")  # 0
print(f"[MASK]: {tokenizer.mask_token_id}")  # 103

# Full encoding with special tokens
encoding = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=20,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print("\nInput IDs:", encoding['input_ids'])
print("Attention Mask:", encoding['attention_mask'])
```

### Handling OOV with WordPiece

```python
# WordPiece breaks down unknown words
rare_word = "supercalifragilisticexpialidocious"
tokens = tokenizer.tokenize(rare_word)
print(f"Rare word '{rare_word}' tokenized as:")
print(tokens)
# ['super', '##cal', '##if', '##rag', '##il', '##istic', '##ex', '##pia', '##lid', '##oc', '##ious']
```

---

## 5. SentencePiece (Language-Agnostic) 🌍

**Used by:** T5, XLNet, ALBERT, multilingual models

### Why SentencePiece?

- **Language-agnostic:** Works for any language (no spaces needed)
- **Handles raw text:** No pre-tokenization required
- **Reversible:** Can perfectly reconstruct original text

### Train SentencePiece

```python
import sentencepiece as spm

# Prepare corpus file
corpus_file = 'corpus.txt'
with open(corpus_file, 'w', encoding='utf-8') as f:
    f.write("The quick brown fox jumps over the lazy dog\n" * 1000)
    f.write("Natural language processing is amazing\n" * 1000)

# Train model
spm.SentencePieceTrainer.train(
    input=corpus_file,
    model_prefix='spm_model',
    vocab_size=1000,
    model_type='bpe',  # or 'unigram'
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# Load trained model
sp = spm.SentencePieceProcessor()
sp.load('spm_model.model')

# Encode
text = "Natural language processing"
encoded = sp.encode_as_pieces(text)
print("Pieces:", encoded)

ids = sp.encode_as_ids(text)
print("IDs:", ids)

# Decode
decoded = sp.decode_pieces(encoded)
print("Decoded:", decoded)
```

### T5 Tokenizer (SentencePiece)

```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-small')

text = "translate English to French: Hello, how are you?"

# Tokenize
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# ['▁translate', '▁English', '▁to', '▁French', ':', '▁Hello', ',', '▁how', '▁are', '▁you', '?']
# Note: ▁ = space character

ids = tokenizer.encode(text)
print("IDs:", ids)

# Works great for non-English
chinese_text = "自然语言处理很有趣"
tokens_zh = tokenizer.tokenize(chinese_text)
print("\nChinese tokens:", tokens_zh)
```

---

## 6. Comparing Tokenization Methods 🔍

### Side-by-Side Comparison

```python
from transformers import (
    GPT2Tokenizer,      # BPE
    BertTokenizer,      # WordPiece
    T5Tokenizer,        # SentencePiece
    RobertaTokenizer    # Byte-level BPE
)

text = "ChatGPT is revolutionizing AI!"

tokenizers = {
    'GPT-2 (BPE)': GPT2Tokenizer.from_pretrained('gpt2'),
    'BERT (WordPiece)': BertTokenizer.from_pretrained('bert-base-uncased'),
    'T5 (SentencePiece)': T5Tokenizer.from_pretrained('t5-small'),
    'RoBERTa (Byte-BPE)': RobertaTokenizer.from_pretrained('roberta-base')
}

print(f"Text: '{text}'\n")
print(f"{'Tokenizer':<25} {'Tokens':<50} {'Count'}")
print("=" * 80)

for name, tok in tokenizers.items():
    tokens = tok.tokenize(text)
    print(f"{name:<25} {str(tokens):<50} {len(tokens)}")

# Vocabulary sizes
print("\nVocabulary Sizes:")
for name, tok in tokenizers.items():
    print(f"{name:<25} {tok.vocab_size:,}")
```

### Tokenization Efficiency

```python
import time

def benchmark_tokenizer(tokenizer, texts):
    """Measure tokenization speed"""
    start = time.time()
    for text in texts:
        tokenizer.encode(text)
    elapsed = time.time() - start
    return elapsed

# Generate test corpus
test_texts = ["This is a test sentence for benchmarking"] * 10000

print("Tokenization Speed (10,000 sentences):")
print("-" * 50)

for name, tok in tokenizers.items():
    elapsed = benchmark_tokenizer(tok, test_texts)
    print(f"{name:<25} {elapsed:.2f}s ({10000/elapsed:.0f} sent/sec)")
```

---

## 7. Advanced Tokenization Techniques 🚀

### Handling Special Cases

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Email addresses
email = "contact@example.com"
print(f"Email: {tokenizer.tokenize(email)}")

# URLs
url = "https://www.example.com/page"
print(f"URL: {tokenizer.tokenize(url)}")

# Hashtags
hashtag = "#NaturalLanguageProcessing"
print(f"Hashtag: {tokenizer.tokenize(hashtag)}")

# Numbers
number = "1,234,567.89"
print(f"Number: {tokenizer.tokenize(number)}")

# Code
code = "def hello_world():"
print(f"Code: {tokenizer.tokenize(code)}")
```

### Custom Vocabulary Extension

```python
# Add custom tokens
new_tokens = ['[USER]', '[AGENT]', '[PRODUCT]', 'COVID-19', 'ChatGPT']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f"Original vocab size: {tokenizer.vocab_size}")

# Add new tokens
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} tokens")
print(f"New vocab size: {len(tokenizer)}")

# Test new tokens
text = "[USER] What is ChatGPT? [AGENT] ChatGPT is a language model."
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Important: Resize model embeddings when using custom tokens
# model.resize_token_embeddings(len(tokenizer))
```

### Handling Multiple Languages

```python
from transformers import XLMRobertaTokenizer

# XLM-RoBERTa: Multilingual model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

texts = {
    'English': "Natural language processing is amazing",
    'Spanish': "El procesamiento del lenguaje natural es increíble",
    'Chinese': "自然语言处理非常棒",
    'Arabic': "معالجة اللغة الطبيعية مذهلة",
    'Hindi': "प्राकृतिक भाषा प्रसंस्करण अद्भुत है"
}

for lang, text in texts.items():
    tokens = tokenizer.tokenize(text)
    print(f"{lang:10} ({len(tokens):2} tokens): {tokens[:5]}...")
```

---

## 8. Practical Considerations 💡

### Choosing Vocabulary Size

```python
# Trade-offs
vocab_sizes = {
    '10K': {
        'pros': ['Fast inference', 'Small memory'],
        'cons': ['Long sequences', 'Loss of meaning']
    },
    '30K-50K': {
        'pros': ['Good balance', 'Standard choice'],
        'cons': ['Moderate memory']
    },
    '100K+': {
        'pros': ['Short sequences', 'Preserve meaning'],
        'cons': ['Slow inference', 'Large memory']
    }
}

# BERT: 30K, GPT-2: 50K, T5: 32K
```

### Sequence Length Considerations

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Long text
long_text = "This is a very long sentence " * 100

# Truncation
encoded = tokenizer.encode_plus(
    long_text,
    max_length=512,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)

print(f"Input IDs shape: {encoded['input_ids'].shape}")
print(f"Attention mask shape: {encoded['attention_mask'].shape}")

# Count actual tokens (excluding padding)
actual_tokens = encoded['attention_mask'].sum().item()
print(f"Actual tokens (non-padding): {actual_tokens}")
```

### Batch Processing

```python
# Efficient batch encoding
texts = [
    "First sentence",
    "Second sentence that is longer",
    "Third"
]

# Without padding - different lengths
for text in texts:
    tokens = tokenizer.encode(text)
    print(f"{len(tokens)} tokens: {text}")

# With padding - same length
batch_encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=20,
    return_tensors='pt'
)

print("\nBatch encoding:")
print(f"Input IDs shape: {batch_encoding['input_ids'].shape}")
print(f"Attention mask:\n{batch_encoding['attention_mask']}")
```

---

## Quick Reference 📖

### Common Tokenizer Methods

```python
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Encode
ids = tokenizer.encode("Hello world")
tokens = tokenizer.tokenize("Hello world")
encoding = tokenizer.encode_plus("Hello", "world")  # Pair of sentences

# Decode
text = tokenizer.decode(ids)
text_skip_special = tokenizer.decode(ids, skip_special_tokens=True)

# Batch encode
batch = tokenizer(["Text 1", "Text 2"], padding=True, return_tensors='pt')

# Vocabulary
vocab_size = tokenizer.vocab_size
token_to_id = tokenizer.convert_tokens_to_ids(["hello", "world"])
id_to_token = tokenizer.convert_ids_to_tokens([101, 102])

# Special tokens
tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token
```

### Tokenization Decision Tree

```
Which tokenizer to use?

1. Pre-trained model? → Use its tokenizer
   - BERT → BertTokenizer
   - GPT-2 → GPT2Tokenizer
   - T5 → T5Tokenizer

2. Training from scratch?
   - English only → BPE or WordPiece
   - Multilingual → SentencePiece
   - Code → Byte-level BPE

3. Custom domain?
   - Start with pre-trained, add custom tokens
   - Or train from scratch on domain corpus
```

---

## Practice Exercises 🏋️

### Exercise 1: Build Custom Tokenizer

Train a BPE tokenizer on your favorite book or dataset.

```python
# Your code here
# 1. Load corpus
# 2. Train BPE with vocab_size=5000
# 3. Compare with GPT-2 tokenizer on same text
```

### Exercise 2: Tokenization Analysis

Compare how different tokenizers handle:
- Domain-specific terms (medical, legal, technical)
- Code snippets
- Social media text (hashtags, @mentions, emojis)
- Non-English text

### Exercise 3: Efficiency Study

Measure memory and speed trade-offs:
- Vocab size: 1K, 10K, 50K, 100K
- Sequence length: 128, 512, 1024, 2048

---

## Key Takeaways 💡

1. **Modern NLP uses subword tokenization** (BPE, WordPiece, SentencePiece)
2. **BPE:** Merge most frequent pairs (GPT family)
3. **WordPiece:** Merge based on likelihood (BERT family)
4. **SentencePiece:** Language-agnostic, handles raw text (T5, multilingual)
5. **Trade-off:** Vocabulary size ↔ Sequence length ↔ Memory
6. **Always use the tokenizer that matches your model**
7. **Custom domains may need vocabulary extension or retraining**

---

**Next:** [Lesson 2 - Embeddings & Transformer Architecture →](Lesson%202%20-%20Embeddings%20and%20Transformer%20Architecture.md)
