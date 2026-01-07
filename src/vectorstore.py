from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_vectorstore(chunks, persist_dir="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"Vectorstore created with {len(chunks)} chunks")
    return vectorstore


def load_vectorstore(persist_dir="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    print("Vectorstore loaded")
    return vectorstore


if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    
    docs = load_documents()
    chunks = chunk_documents(docs)
    
    vectorstore = create_vectorstore(chunks)
    
    query = "What season is rice grown?"
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:\n{doc.page_content[:200]}...\n")