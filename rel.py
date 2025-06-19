# main.py
from sentence_transformers import SentenceTransformer, util

# Load model hanya sekali
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def semantic_similarity(instruction, response):
    embeddings = embedder.encode([instruction, response], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

def is_semantically_relevant(instruction, response, threshold=0.2):
    similarity = semantic_similarity(instruction, response)
    return similarity >= threshold, similarity