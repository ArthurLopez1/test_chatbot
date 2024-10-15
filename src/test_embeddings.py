from embeddings import generate_embeddings
from pathlib import Path

# small sample for testing
 
file_path = Path(__file__).parent.parent / "data/sample_docs.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    documents = file.readlines()

print(f"[INFO] Loaded {len(documents)} documents from the file.")

docs, embeddings = generate_embeddings(documents)

print("[INFO] Embeddings generated for test documents:")
for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
    print(f"Document {i + 1}: {doc[:50]}... -> Embedding (first 5 values): {embedding[:5]}")