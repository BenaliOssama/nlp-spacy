# spaCy NLP Reference Guide

## Core Concept: From Text to Numbers

**The Problem:** Computers can't understand text as meaning. They need numbers to do math and analysis.

**The Solution:** Convert words and sentences into numerical vectors (lists of numbers). Words with similar meanings get similar vectors.

**How spaCy does it:** Pre-trained models that already learned these mappings from billions of English sentences.

---

## Models Used

### en_core_web_sm (Small)
- **Word vector size:** 96 dimensions
- **Use case:** Fast, lightweight, sufficient for basic NLP tasks
- **Load:** `nlp = spacy.load('en_core_web_sm')`

### en_core_web_md (Medium)
- **Word vector size:** 300 dimensions
- **Use case:** More nuanced semantic relationships, better accuracy
- **Load:** `nlp = spacy.load('en_core_web_md')`

---

## spaCy Pipeline Components

When you load a model, you get a complete pipeline with these components already trained:

1. **Tokenizer** — Splits text into individual words (tokens), handling punctuation intelligently
2. **Tagger** — Labels parts of speech (NOUN, VERB, ADJ, etc.) and POS tags (NN, VB, JJ, etc.)
3. **Parser** — Analyzes sentence structure and dependencies
4. **NER (Named Entity Recognition)** — Identifies and classifies named entities (PERSON, ORG, GPE, DATE, etc.)
5. **Lemmatizer** — Reduces words to their base form (running → run, better → good)

---

## Key Concepts & Operations

### Embeddings (Word Vectors)

An embedding is a list of numbers that represents a word's meaning.

```python
nlp = spacy.load('en_core_web_md')
doc = nlp("car")
vector = doc[0].vector  # 300-dimensional vector

print(vector.shape)  # (300,)
```

**Why 300 numbers?** It's a design choice balancing detail (more dimensions = more nuance) vs. speed (fewer dimensions = faster computation).

### Sentence Embeddings

A sentence's embedding is the **average of all its word embeddings**.

```python
doc = nlp("I want to buy shoes")
sentence_embedding = doc.vector  # Average of all word vectors
```

### Similarity (Cosine Distance)

Measure how semantically similar two words/sentences are (0 = opposite, 1 = identical).

```python
from sklearn.metrics.pairwise import cosine_similarity

doc1 = nlp("I want to buy shoes")
doc2 = nlp("I would love to purchase running shoes")

similarity = cosine_similarity([doc1.vector], [doc2.vector])[0][0]
# Result: ~0.707 (high similarity - both about buying shoes)
```

### Tokenization

Breaking text into individual words, with intelligent punctuation handling.

```python
doc = nlp("Tokenize this sentence.")
for token in doc:
    print(token.text)

# Output:
# Tokenize
# this
# sentence
# .
```

### Named Entity Recognition (NER)

Identifying and classifying important entities in text.

```python
doc = nlp("Apple Inc. is headquartered in Cupertino, California.")

for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Output:
# Apple Inc. (ORG)
# Cupertino (GPE)
# California (GPE)
```

**Common entity labels:**
- `PERSON` — Names of people
- `ORG` — Organizations, companies
- `GPE` — Geopolitical entities (countries, cities)
- `DATE` — Dates and times
- `CARDINAL` — Numbers
- `NORP` — Nationalities, political/religious groups
- `LOC` — Non-GPE locations
- `MONEY` — Monetary values
- `FAC` — Facilities (buildings, airports, etc.)

### Part-of-Speech (POS) Tagging

Labeling the grammatical role of each word.

```python
doc = nlp("Heat water in a large vessel")

for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")

# Output:
# Heat: VERB (VB)
# water: NOUN (NN)
# in: ADP (IN)
# a: DET (DT)
# large: ADJ (JJ)
# vessel: NOUN (NN)
```

**POS tags (simplified):**
- `NOUN` / `NN`, `NNS` — Nouns
- `VERB` / `VB`, `VBD`, `VBG` — Verbs
- `ADJ` / `JJ` — Adjectives
- `ADP` / `IN` — Prepositions
- `DET` / `DT` — Determiners
- `PROPN` / `NNP` — Proper nouns (names)

---

## Exercises Summary

### Ex1: Embedding Basics
- Load a pre-trained model
- Extract word embeddings
- Verify shape and values

### Ex2: Tokenization
- Break text into tokens
- See how spaCy handles punctuation

### Ex3: Multi-word Embeddings & Similarity
- Get embeddings for multiple words
- Compute cosine similarity between all pairs
- Visualize with heatmap

### Ex4: Sentence Similarity
- Compute embeddings for full sentences (average of word vectors)
- Measure similarity between sentences
- Use case: Intent detection for e-commerce

### Ex5: Named Entity Recognition
- Extract entities from text
- Handle disambiguation (same word, different meanings)
- Label entities by type

### Ex6: Part-of-Speech Tagging
- Label grammatical roles of words
- Filter sentences by POS tags
- Extract specific word types

---

## Common Workflow

```python
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Load model
nlp = spacy.load('en_core_web_md')

# 2. Process text
text = "Your text here"
doc = nlp(text)

# 3. Access information
for token in doc:
    print(token.text, token.pos_, token.vector)

for ent in doc.ents:
    print(ent.text, ent.label_)

for sent in doc.sents:
    print(sent.text)

# 4. Compute similarity
doc1 = nlp("text 1")
doc2 = nlp("text 2")
sim = cosine_similarity([doc1.vector], [doc2.vector])[0][0]
```

---

## Setup

Create a fresh environment (reproducible on any computer):

```bash
chmod +x setup_nlp_spacy.sh
./setup_nlp_spacy.sh
source ex00/bin/activate
```

This installs:
- spacy==3.4.4
- numpy==1.24.3
- pandas, jupyter, scikit-learn, matplotlib
- en_core_web_sm-3.4.1 (96-dim vectors)
- en_core_web_md-3.4.1 (300-dim vectors)

---

## Key Takeaways

1. **Embeddings are learned, not hand-coded** — They emerge from training on billions of examples
2. **Vector size is a design choice** — Larger = more detail, smaller = faster
3. **Similarity is mathematical** — You can compute how "close" two words are numerically
4. **Pre-trained models do everything** — Tokenization, NER, POS tagging all happen automatically
5. **You ask for what you need** — The model computes everything; you just access what you want
