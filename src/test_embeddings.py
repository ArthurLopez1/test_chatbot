from embeddings import generate_embeddings
from retriever import build_faiss_index, query_faiss_index
from pathlib import Path

# small sample for testing
 
file_path = Path(__file__).parent.parent / "data/sample_docs.txt"
print(f"[INFO] Loading documents from: {file_path}")

with open(file_path, 'r', encoding='utf-8') as file:
    documents = file.readlines()

print(f"[TEST INFO] Loaded {len(documents)} documents from the file.")

docs, embeddings = generate_embeddings(documents)

# Building FAISS index with generated embeddings
faiss_index = build_faiss_index(embeddings)

# test a sample query (later to be replaced with dynamic input for the chatbot)
query = "Is python a popular programming language?"

_, query_embedding = generate_embeddings([query])

# Query the FAISS index with the sample query's embedding
retrieved_docs = query_faiss_index(faiss_index, query_embedding[0], docs, top_k=3)

if retrieved_docs:
    print("\n[TEST INFO]\n\nRetrived documents for the query: {query}")
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i + 1}: {doc[:50]}...")
else:
    print("[INFO] No documents were retrieved")
