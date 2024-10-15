import faiss
import numpy as np

def build_faiss_index(embeddings):
    """Create a FAISS index for fast similarity search on embeddings"""
    embeddings_dim = len(embeddings[0])

    # initialize FAIss index for L2 (Euclidean) distance
    faiss_index = faiss.IndexFlatL2(embeddings_dim)
    print(f"[INFO] Building FAISS index with {len(embeddings)} embeddings")

    faiss_index.add(np.array(embeddings).astype('float32'))
    print(f"[INFO] FAISS index built seccesfully!")

    return faiss_index

def query_faiss_index(faiss_index, query_embedding, docs, top_k=3):
    """
    Query the FAISS index to find the most similar documents
    """
    print(f"[INFO] Querying FAISS index with the input embedding...")

    # Perform similarity search for the top_k results
    distances, indices = faiss_index.search(np.array([]))


