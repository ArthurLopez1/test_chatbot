import faiss
import numpy as np

def build_faiss_index(embeddings):
    """Create a FAISS index for fast similarity search on embeddings"""
    embeddings_dim = len(embeddings[0])

    # initialize FAIss index for L2 (Euclidean) distance
    faiss_index = faiss.IndexFlatL2(embeddings_dim)
    print(f"[RETRIEVER INFO] Building FAISS index with {len(embeddings)} embeddings")

    faiss_index.add(np.array(embeddings).astype('float32'))
    print(f"[RETRIEVER INFO] FAISS index built seccesfully!")

    return faiss_index

def query_faiss_index(faiss_index, query_embedding, docs, top_k=3):
    """
    Query the FAISS index to find the most similar documents
    """
    print(f"[RETRIEVER INFO] Querying FAISS index with the input embedding...")

    # Perform similarity search for the top_k results
    distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), top_k)

    # Checking what was returned from FAISS
    print(f"[FAISS INFO] Returned distances: {distances.flatten()}")
    print(f"[FAISS INFO] Returned indices: {indices.flatten()}")

    # Handling cases where no valid indices are returned
    if len(indices) ==0:
        print(f"[ERROR] No documents retrieved from FAISS index.")
        return None
    
    retrieved_docs = [docs[i] for i in indices[0] if i < len(docs)]

    # Checking if no valid documents were retrieved
    if not retrieved_docs:
        print("[ERROR] No documents found corresponding to the retrieved indices.")
        return None
    
    return retrieved_docs



    # print(f"[RETRIEVER INFO] Retrieved top {top_k} documents with the following distances: {distances.flatten()}") 

    # Return the documents corresponding to the top indices


