import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load medium model
nlp = spacy.load('en_core_web_md')

# Three sentences
sentence_1 = "I want to buy shoes"
sentence_2 = "I would love to purchase running shoes"
sentence_3 = "I am in my room"

# Process each sentence
doc1 = nlp(sentence_1)
doc2 = nlp(sentence_2)
doc3 = nlp(sentence_3)

# Get sentence embeddings (average of word vectors)
# spaCy automatically does this with doc.vector
embedding_1 = doc1.vector
embedding_2 = doc2.vector
embedding_3 = doc3.vector

# Compute pairwise cosine similarities
sim_1_2 = cosine_similarity([embedding_1], [embedding_2])[0][0]
sim_1_3 = cosine_similarity([embedding_1], [embedding_3])[0][0]
sim_2_3 = cosine_similarity([embedding_2], [embedding_3])[0][0]

# Print results
print(f"sentence_1 <=> sentence 2 : {sim_1_2}")
print(f"sentence_1 <=> sentence 3: {sim_1_3}")
print(f"sentence_2 <=> sentence 3: {sim_2_3}")
