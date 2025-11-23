# Lesson 3: Syntax, Semantics & Language Understanding 📖

**Module 7: Natural Language Processing | Lesson 3 of 7**

Deep dive into linguistic structures - what transformers learn implicitly!

---

## Why Syntax & Semantics Matter 🎯

**Transformers learn these patterns, but understanding them helps:**
- Debug model failures
- Design better prompts
- Build structured information extractors
- Understand what layers learn

---

## 1. Part-of-Speech (POS) Tagging 🏷️

### What is POS Tagging?

Labeling each word with its grammatical role:
- **Noun (NN):** person, place, thing
- **Verb (VB):** action
- **Adjective (JJ):** describes noun
- **Adverb (RB):** describes verb

### POS Tagging with spaCy

```python
import spacy

nlp = spacy.load('en_core_web_sm')

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

print("Word\t\tPOS\tTag\tDependency")
print("-" * 50)
for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.tag_:6} {token.dep_}")

# Output:
# The          DET    DT     det
# quick        ADJ    JJ     amod
# brown        ADJ    JJ     amod
# fox          NOUN   NN     nsubj
# jumps        VERB   VBZ    ROOT
# ...
```

### Build Custom POS Tagger

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import treebank
nltk.download('treebank')

# Load training data
tagged_sentences = treebank.tagged_sents()

def word_features(sentence, i):
    word = sentence[i][0]
    return {
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sentence) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if i == 0 else sentence[i-1][0],
        'next_word': '' if i == len(sentence)-1 else sentence[i+1][0],
    }

# Prepare training data
X_train = []
y_train = []

for sent in tagged_sentences[:3000]:
    for i in range(len(sent)):
        X_train.append(word_features(sent, i))
        y_train.append(sent[i][1])

# Vectorize
vectorizer = DictVectorizer()
X_vec = vectorizer.fit_transform(X_train)

# Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, y_train)

# Test
test_sentence = [("The", ""), ("cat", ""), ("sits", "")]
X_test = [word_features(test_sentence, i) for i in range(len(test_sentence))]
X_test_vec = vectorizer.transform(X_test)
predictions = clf.predict(X_test_vec)

for word, tag in zip([w[0] for w in test_sentence], predictions):
    print(f"{word}: {tag}")
```

---

## 2. Dependency Parsing 🌳

### Understand Dependencies

**Dependency:** Relationship between words
- **nsubj:** Nominal subject (who does the action)
- **dobj:** Direct object (receives the action)
- **amod:** Adjectival modifier
- **det:** Determiner

### Visualize Dependency Tree

```python
from spacy import displacy

text = "The autonomous vehicle detected pedestrians crossing the busy street"
doc = nlp(text)

# Visualize in notebook
# displacy.render(doc, style='dep', jupyter=True)

# Save as SVG
svg = displacy.render(doc, style='dep')
with open('dependency_tree.svg', 'w') as f:
    f.write(svg)

# Extract dependencies manually
print("\nDependency Relations:")
for token in doc:
    print(f"{token.text:15} --{token.dep_:10}--> {token.head.text}")
```

### Extract Subject-Verb-Object

```python
def extract_svo(doc):
    """Extract Subject-Verb-Object triples"""
    svo_triples = []
    
    for token in doc:
        if token.pos_ == 'VERB':
            subject = None
            obj = None
            
            for child in token.children:
                if child.dep_ == 'nsubj':
                    subject = child
                elif child.dep_ in ['dobj', 'attr']:
                    obj = child
            
            if subject and obj:
                svo_triples.append((subject.text, token.text, obj.text))
    
    return svo_triples

sentences = [
    "The cat chased the mouse",
    "Scientists discovered a new planet",
    "The algorithm processes large datasets"
]

for sent in sentences:
    doc = nlp(sent)
    triples = extract_svo(doc)
    print(f"{sent}:")
    for s, v, o in triples:
        print(f"  ({s}, {v}, {o})")
```

---

## 3. Named Entity Recognition (NER) 🎯

### Extract Named Entities

```python
text = """
Apple Inc. is planning to open a new office in London next year.
Tim Cook announced this during a meeting with British Prime Minister
Rishi Sunak on January 15, 2024. The investment is estimated at $500 million.
"""

doc = nlp(text)

print("Entity\t\t\tType")
print("-" * 40)
for ent in doc.ents:
    print(f"{ent.text:25} {ent.label_}")

# Entity types:
# ORG: Organization
# GPE: Geopolitical entity (country, city)
# PERSON: Person name
# DATE: Date
# MONEY: Monetary value
```

### Train Custom NER Model

```python
import random
from spacy.training import Example

# Training data (text, annotations)
TRAIN_DATA = [
    ("Apple is looking at buying UK startup for $1 billion", {
        "entities": [(0, 5, "ORG"), (27, 29, "GPE"), (42, 52, "MONEY")]
    }),
    ("Google acquired DeepMind for $500 million", {
        "entities": [(0, 6, "ORG"), (16, 24, "ORG"), (29, 41, "MONEY")]
    }),
]

# Create blank model
nlp_custom = spacy.blank("en")
ner = nlp_custom.add_pipe("ner")

# Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Train
optimizer = nlp_custom.begin_training()
for i in range(30):
    random.shuffle(TRAIN_DATA)
    losses = {}
    
    for text, annots in TRAIN_DATA:
        doc = nlp_custom.make_doc(text)
        example = Example.from_dict(doc, annots)
        nlp_custom.update([example], drop=0.5, losses=losses)
    
    print(f"Epoch {i+1}, Loss: {losses['ner']:.2f}")

# Test
test_text = "Microsoft bought LinkedIn for $26 billion"
doc = nlp_custom(test_text)
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

---

## 4. Semantic Role Labeling (SRL) 🎭

### What is SRL?

**Identifies "who did what to whom, when, where, why"**

```
Sentence: "John gave Mary a book in the library yesterday"

Roles:
- Agent (who): John
- Action: gave
- Recipient (to whom): Mary  
- Theme (what): a book
- Location (where): in the library
- Time (when): yesterday
```

### SRL with AllenNLP

```python
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz"
)

sentence = "The scientist discovered a new species in the Amazon rainforest"
result = predictor.predict(sentence=sentence)

# Print roles
for i, verb in enumerate(result['verbs']):
    print(f"\nVerb: {verb['verb']}")
    print(f"Description: {verb['description']}")
    
    # Extract arguments
    for tag in verb['tags']:
        if tag != 'O':
            print(f"  {tag}")
```

---

## 5. Coreference Resolution 🔗

### Resolve Pronouns to Entities

```python
import neuralcoref

# Add to spaCy pipeline
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

text = """
John went to the store. He bought some milk. 
Mary met him there. She was surprised to see him.
"""

doc = nlp(text)

# Find coreferences
if doc._.has_coref:
    print("Coreference Chains:")
    for cluster in doc._.coref_clusters:
        print(f"  {cluster.main} -> {cluster.mentions}")
    
    # Resolve
    print("\nResolved text:")
    print(doc._.coref_resolved)
```

---

## 6. Information Extraction Pipeline 🏭

### Build Complete IE System

```python
def extract_information(text):
    """Complete information extraction"""
    doc = nlp(text)
    
    # Extract entities
    entities = {
        'organizations': [],
        'people': [],
        'locations': [],
        'dates': [],
        'money': []
    }
    
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            entities['organizations'].append(ent.text)
        elif ent.label_ == 'PERSON':
            entities['people'].append(ent.text)
        elif ent.label_ == 'GPE':
            entities['locations'].append(ent.text)
        elif ent.label_ == 'DATE':
            entities['dates'].append(ent.text)
        elif ent.label_ == 'MONEY':
            entities['money'].append(ent.text)
    
    # Extract relations (SVO triples)
    relations = extract_svo(doc)
    
    return {
        'entities': entities,
        'relations': relations,
        'sentiment': analyze_sentiment(text)
    }

def analyze_sentiment(text):
    # Simple sentiment (upgrade with transformers)
    from textblob import TextBlob
    return TextBlob(text).sentiment.polarity

# Test
news = """
Apple CEO Tim Cook announced on January 15, 2024 that the company 
will invest $500 million in renewable energy projects across California.
The initiative aims to reduce carbon emissions by 50% by 2030.
"""

info = extract_information(news)

print("Extracted Information:")
print(json.dumps(info, indent=2))
```

---

## 7. What BERT Learns About Syntax 🔍

### Probe BERT for Linguistic Knowledge

```python
from transformers import BertForMaskedLM, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Test syntactic agreement
sentences = [
    "The keys to the cabinet [MASK] on the table",  # are
    "The key to the cabinet [MASK] on the table",   # is
]

for sent in sentences:
    inputs = tokenizer(sent, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions for [MASK]
    mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    predictions = outputs.logits[0, mask_idx]
    
    top_tokens = torch.topk(predictions, 5, dim=1)
    
    print(f"\nSentence: {sent}")
    print("Top predictions:")
    for token_id in top_tokens.indices[0]:
        token = tokenizer.decode([token_id])
        print(f"  {token}")
```

---

## Quick Reference 📖

### Common Dependency Relations

```
nsubj     - Nominal subject
dobj      - Direct object  
iobj      - Indirect object
amod      - Adjectival modifier
advmod    - Adverbial modifier
det       - Determiner
prep      - Prepositional modifier
pobj      - Object of preposition
compound  - Compound
```

### Entity Types (spaCy)

```
PERSON    - People, characters
ORG       - Companies, agencies
GPE       - Countries, cities
LOC       - Non-GPE locations
DATE      - Absolute or relative dates
TIME      - Times
MONEY     - Monetary values
PERCENT   - Percentages
PRODUCT   - Objects, vehicles
EVENT     - Named events
```

---

## Practice Exercises 🏋️

### Exercise 1: Build Knowledge Graph
Extract (subject, predicate, object) triples from news articles and build a knowledge graph.

### Exercise 2: Question Answering with Syntax
Use dependency parsing to answer "who", "what", "when" questions.

### Exercise 3: Custom Entity Extractor
Train NER for domain-specific entities (medical terms, legal entities, etc.).

---

## Key Takeaways 💡

1. **POS tagging** labels grammatical roles
2. **Dependency parsing** reveals word relationships
3. **NER** extracts named entities (people, places, organizations)
4. **SRL** identifies semantic roles (who did what to whom)
5. **Coreference** links pronouns to entities
6. **Transformers implicitly learn these patterns** in their layers
7. **Understanding syntax helps debug and design better systems**

---

**Next:** [Lesson 4 - Prompt Engineering & Few-Shot Learning →](Lesson%204%20-%20Prompt%20Engineering%20and%20Few-Shot%20Learning.md)
