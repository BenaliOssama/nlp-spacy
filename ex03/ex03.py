import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load medium model (300 dimensions)
nlp = spacy.load('en_core_web_md')

# Text with multiple words
text = "laptop computer coffee tea water liquid dog cat kitty"

# Process
doc = nlp(text)

# Part 1: Get embeddings and verify shape
print("=== Part 1: Embeddings ===")
for token in doc:
    print(f"{token.text}: shape {token.vector.shape}")

# Sum first 20 values of "laptop"
laptop_vector = doc[0].vector
sum_first_20 = laptop_vector[:20].sum()
print(f"\nSum of first 20 values of 'laptop': {sum_first_20}")

# Part 2: Compute pairwise cosine similarity
print("\n=== Part 2: Similarity Matrix ===")

# Get all vectors as a matrix (9 words × 300 dimensions)
vectors = np.array([token.vector for token in doc])

# Compute cosine similarity between all pairs
similarity_matrix = cosine_similarity(vectors)

# Plot heatmap
plt.figure(figsize=(8, 8))
plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')

# Label axes with word names
words = [token.text for token in doc]
plt.xticks(range(len(words)), words, rotation=45, ha='right')
plt.yticks(range(len(words)), words)

plt.title('Word Similarity Heatmap (Cosine Distance)')
plt.tight_layout()
plt.savefig('similarity_heatmap.png', dpi=100)
plt.show()

print("Heatmap saved as 'similarity_heatmap.png'")
