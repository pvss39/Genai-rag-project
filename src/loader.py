from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(docs_path: str = "data/docs"):
    """Load documents from the specified directory."""
    documents = []
    
    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)
        
        if filename.endswith(".txt"):
            loader = TextLoader(filepath)
            documents.extend(loader.load())
            print(f"✅ Loaded: {filename}")
            
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"✅ Loaded: {filename}")
            
        else:
            print(f"⚠️ Skipped: {filename} (unsupported format)")
    
    print(f"\n📄 Total documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_documents()
    
    
    if docs:
        print("\n--- PREVIEW ---")
        print(docs[0].page_content[:500])
