from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Original docs: {len(documents)}, Chunks: {len(chunks)}")
    
    return chunks


if __name__ == "__main__":
    from loader import load_documents
    
    docs = load_documents()
    chunks = chunk_documents(docs)
    
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:\n{chunk.page_content[:200]}...")