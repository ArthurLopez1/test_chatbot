from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def generate_embeddings(texts):
    # Generate embeddings from a list of list of texts.
    # Split the text into chunks if the documents are large
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    docs = text_splitter.split_text(texts)

    print(f"[INFO] Number of chuncks created: {len(docs)}")

    # Using a local HuggingFace model for embeddings
    embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"[INFO] Generating embeddings for the documents...")

    embeddings = embedding_model.embed_documents(docs)
    print(f"[INFO] Generated {len(embeddings)} embeddings, each of size {len(embeddings[0])}.\n")
    
    return docs, embeddings