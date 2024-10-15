import streamlit as st
from embeddings import generate_embeddings
from retriever import build_faiss_index, query_faiss_index
#from ollama import ollama
from pathlib import Path

# Loading data 
file_path = Path(__file__).parent.parent / "data/sample_docs.txt"

# Reading data
with open(file_path, 'r', encoding='utf-8') as file:
    documents = file.readlines()

# Generating embeddings for documents
docs, embeddings = generate_embeddings(documents)

# Build FAISS index with generated embeddings
faiss_index = build_faiss_index(embeddings)

# Initialize Ollama for LLaMA-based responses
#llama = Ollama()

# Streamlit UI
st.title("Chatbot with Retrival-Augmented Generation")

user_input = st.text_input("Ask me a question:", "")

# User input for the query
if user_input:
    # Generate embedding for the query
    _, query_embedding = generate_embeddings([user_input])

    # Retrive top 3 documents
    retrieved_documents = query_faiss_index(faiss_index, query_embedding[0], docs, top_k=3)

    if retrieved_documents:
        st.write(f"Top 3 documents related to your query '{user_input}':")
        for i, doc in enumerate(retrieved_documents):
            st.write(f"Document {i + 1}: {doc[:200]}...") # show the first 200 characters of the doc

        # Concatenate retrieved docs to form the context
        context = "\n\n".join(retrieved_documents)

        # Generate reponse using LLaMA
        #response = llama.chat({"prompt": f"Context: {context}\n\nUser: {user_input}\nBot:"})
        st.write(f"\nBot's Response: {response['text']}")

    else:
        st.write("No relevant documents were retrieved")