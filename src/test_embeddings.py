from embeddings import generate_embeddings
from pathlib import Path

# small sample for testing
 
documents = Path(__file__).parent / "data/sample_docs.txt"

docs, embeddings = generate_embeddings(documents)

print("[INFO] Embeddings generated for test documents:")
for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
    print(f"Document {i + 1}: {doc[:50]}... -> Embedding (first 5 values): {embedding[:5]}")