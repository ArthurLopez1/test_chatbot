import faiss
import numpy as np

def create_faiss_index(embeddings):
    # Create a FAISS index from embeddings
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

