import numpy as np
np.random.seed(42)
# Generate the first 1024 elements as a random normalized vector
vec1_main = np.random.randn(1024)
vec1_main = vec1_main / np.linalg.norm(vec1_main)

# Generate a 20-dimensional binary vector for the last part
vec1_tail = np.zeros(20)
vec1_tail[0] =  0.1# Set the first 10 elements to

# Full vector: concatenate
vector1 = np.concatenate([vec1_main, vec1_tail])

# For comparison, we create another vector with the same structure
vec2_main = np.random.randn(1024)
vec2_main = vec2_main / np.linalg.norm(vec2_main)
vec2_tail = np.random.randint(0, 2, 20)
vector2 = np.concatenate([vec2_main, vec1_tail])

# Compute cosine similarity
cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

print("Cosine similarity:", cosine_similarity)