from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:""")
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    from vectorstore import create_vectorstore
    
    docs = load_documents()
    chunks = chunk_documents(docs)
    vectorstore = create_vectorstore(chunks)
    
    chain = create_rag_chain(vectorstore)
    
    question = "What season is rice grown in?"
    print(f"Question: {question}\n")
    
    answer = chain.invoke(question)
    print(f"Answer: {answer}")