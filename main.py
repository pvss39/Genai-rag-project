from src.loader import load_documents
from src.chunker import chunk_documents
from src.vectorstore import create_vectorstore, load_vectorstore
from src.chain import create_rag_chain
import os

def main():
    print("AGRICULTURE KNOWLEDGE ASSISTANT")
    print("-" * 40)
    
    if os.path.exists("chroma_db"):
        vectorstore = load_vectorstore()
    else:
        docs = load_documents()
        chunks = chunk_documents(docs)
        vectorstore = create_vectorstore(chunks)
    
    chain = create_rag_chain(vectorstore)
    
    print("Ask questions. Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ")
        
        if question == "quit":
            break
        
        answer = chain.invoke(question)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()