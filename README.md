# GenAI RAG Project — Agriculture Knowledge Assistant

## About This Project

This is a personal project I built to deeply understand how **Retrieval-Augmented Generation (RAG)** works under the hood — not just using a pre-built tool, but actually writing every stage of the pipeline from scratch.

The idea is simple: instead of asking a generic AI chatbot a question and hoping it knows the answer, we give it a **specific document** (our knowledge base) and force it to answer **only from that**. This makes the answers accurate, reliable, and grounded in real data — not hallucinations.

For this project, the knowledge base is about **major Indian crops** — their growing seasons, climate needs, water requirements, and more. You ask a question like *"What climate does wheat grow in?"* and the system finds the right information from the document and gives you a precise, human-like answer.

The best part? It all runs **completely offline** — no cloud API charges, no internet needed during inference. Everything from the embeddings to the language model runs locally on your machine.

---

## The Problem This Solves

Standard LLMs (like ChatGPT) are trained on general internet data. They don't know about **your specific documents**, your company's internal knowledge, or niche domain data. And when they don't know something, they often make things up (hallucination).

RAG fixes this by:
1. Taking your own documents
2. Converting them into a searchable format (vector embeddings)
3. When a user asks a question — finding the most relevant parts of your documents
4. Passing that context to the LLM and saying *"answer only from this"*

This project implements that entire flow end-to-end, from raw text files to a working interactive assistant.

---

## How the Pipeline Works — Step by Step

### Stage 1: Load Documents
The system reads all `.txt` and `.pdf` files from the `data/docs/` folder. Right now it has a text file about major crops, but you can drop in any document and it works the same way.

### Stage 2: Chunk the Documents
A 50-page document can't be fed into an LLM all at once. So we split it into small, overlapping chunks of 500 characters each. The 50-character overlap between chunks ensures no information is lost at the boundaries.

### Stage 3: Create Vector Embeddings
Each chunk is converted into a list of numbers (a vector) using the `all-MiniLM-L6-v2` model from HuggingFace. This model understands language — so chunks that are *semantically similar* end up with *numerically similar* vectors. That's what makes the search smart, not just keyword-based.

### Stage 4: Store in Chroma Vector Database
All those vectors are stored in **Chroma**, a local vector database. This is persisted to disk so the indexing only happens once. Every time you run the app after that, it loads instantly.

### Stage 5: Answer Questions
When you type a question:
- The question is also converted to a vector
- Chroma finds the 2 most similar chunks from the knowledge base
- Those chunks + your question are combined into a prompt
- The prompt is sent to **Llama 3.2** (running locally via Ollama)
- Llama reads only that context and gives you a focused, accurate answer

---

## Project Structure

```
genai-rag-project/
│
├── main.py                   # Run this — it's the interactive CLI assistant
│
├── src/
│   ├── loader.py             # Reads documents from data/docs/ (txt + pdf)
│   ├── chunker.py            # Splits documents into overlapping chunks
│   ├── vectorstore.py        # Creates embeddings and manages Chroma DB
│   └── chain.py              # Wires everything into a runnable RAG chain
│
├── data/
│   └── docs/
│       └── major_crops_growing_conditions.txt   # The knowledge base
│
├── chroma_db/                # Auto-created vector index (don't commit this)
├── requirements.txt          # All Python dependencies
└── .env                      # Your API keys (never commit this)
```

Every module in `src/` is independently runnable for testing. No tangled dependencies.

---

## Tech Stack

| What | Tool | Why |
|---|---|---|
| LLM Orchestration | LangChain 0.3.25 | Chains together all pipeline stages cleanly |
| Language Model | Llama 3.2 via Ollama | Runs 100% locally, zero cost, full privacy |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Lightweight 384-dim model, fast and accurate |
| Vector Database | Chroma | Open-source, persists to disk, no server needed |
| Document Parsing | LangChain Community | Handles both .txt and .pdf out of the box |
| Environment Config | python-dotenv | Keeps API keys out of source code |
| Language | Python 3.10+ | — |

---

## Getting It Running

### What You Need First
- Python 3.10 or above
- [Ollama](https://ollama.com) installed on your machine
- Llama 3.2 pulled: run `ollama pull llama3.2` in your terminal

### Install Python dependencies
```bash
pip install -r requirements.txt
```

### Set up your environment file
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```
(These are optional — the core pipeline uses Ollama locally)

### Run it
```bash
python main.py
```

First run will index the documents and create the Chroma database. After that, every run loads instantly.

---

## Talking to It

Once running, you get an interactive prompt:

```
You: What season is rice grown in?
Answer: Rice is a Kharif crop, grown from June to September in warm and humid conditions with temperatures between 20 to 35 degrees Celsius.

You: What are the water requirements for sugarcane?
Answer: Sugarcane requires continuous water supply throughout its 12 to 18 month growing cycle. It thrives in hot and humid climates.

You: quit
```

Type `quit` to exit.

---

## Testing Individual Modules

You don't need to run the full app to test a specific stage:

```bash
python src/loader.py        # See what documents are loaded
python src/chunker.py       # See how text gets split into chunks
python src/vectorstore.py   # See similarity search working live
python src/chain.py         # Run the full RAG chain on a test question
```

---

## Knowledge Base — What It Knows Right Now

| Crop | Season | Climate | Water Needs |
|---|---|---|---|
| Rice | Kharif (Jun – Sep) | Warm & humid, 20–35°C | High |
| Wheat | Rabi (Oct – Mar) | Cool climate | Moderate |
| Cotton | 6 – 8 months | Warm, moderate rainfall | Moderate |
| Sugarcane | 12 – 18 months | Hot & humid | Continuous supply |
| Maize | Kharif or Rabi | Moderate temp, adaptable | Moderate |

**Want to extend it?** Drop any `.txt` or `.pdf` into `data/docs/`, delete the `chroma_db/` folder, and run `main.py` again. It will re-index everything automatically.

---

## Design Choices Worth Noting

**Why Ollama + Llama 3.2?**
Running inference locally means no per-token API charges and no data leaving your machine. Great for sensitive documents.

**Why all-MiniLM-L6-v2?**
It's one of the best lightweight embedding models for semantic similarity. Fast enough to run on CPU, accurate enough for production use.

**Why only k=2 retrieved chunks?**
For a focused knowledge base, 2 chunks are enough context without overwhelming the prompt. This keeps responses concise and relevant.

**Why persistent Chroma storage?**
Re-embedding documents on every run wastes compute. Persisting to disk means you only pay the indexing cost once.

---

## Dependencies

```
langchain==0.3.25
langchain-community==0.3.24
langchain-openai==0.3.17
chromadb==0.6.3
pypdf==5.5.0
python-dotenv==1.1.1
openai==1.78.1
```

---

---

# Resume Section

> Copy any of the below directly into your resume. Everything here is accurate and based on what this project actually does.

---

## Project Title for Resume

**GenAI RAG Pipeline — Domain-Specific Question Answering System**
`Python` · `LangChain` · `Chroma DB` · `HuggingFace` · `Ollama` · `Llama 3.2` · `RAG` · `Generative AI`

---

## One-Line Project Summary

> Built an end-to-end Retrieval-Augmented Generation (RAG) system in Python using LangChain, Chroma vector database, and a locally hosted Llama 3.2 LLM to answer domain-specific questions accurately from a custom knowledge base — with zero cloud API dependency.

---

## Resume Bullet Points (pick 4–5 that fit your role)

- Designed and implemented a complete **RAG (Retrieval-Augmented Generation) pipeline** from scratch using **LangChain**, covering document ingestion, recursive text chunking, vector embedding, semantic retrieval, and LLM-based answer generation

- Built a **persistent vector search layer** using **Chroma DB** with **HuggingFace `all-MiniLM-L6-v2`** embeddings, enabling documents to be indexed once and queried instantly on every subsequent run

- Integrated **Ollama + Llama 3.2** for fully **local LLM inference**, eliminating cloud API costs and ensuring complete data privacy during question-answering

- Implemented **constrained prompt engineering** to ground LLM responses strictly in retrieved document context, significantly reducing hallucinations in domain-specific Q&A

- Developed a **modular, testable architecture** with clean separation between pipeline stages (document loading, chunking, vectorstore, chain orchestration) — each module independently runnable for debugging and testing

- Supported **multi-format document ingestion** (`.txt` and `.pdf`) with an extensible knowledge base — new documents can be added and re-indexed without changing any code

- Built an **interactive CLI assistant** that smartly detects whether a vector index already exists and skips re-indexing, optimizing startup time for repeat sessions

---

## Skills This Project Demonstrates

**Generative AI & NLP:**
Retrieval-Augmented Generation (RAG), Prompt Engineering, Semantic Search, Vector Embeddings, Large Language Models (LLMs)

**Frameworks & Libraries:**
LangChain, HuggingFace Transformers, Chroma DB, Ollama, PyPDF, python-dotenv

**Software Engineering:**
Modular Python architecture, CLI application design, environment configuration management, persistent storage design

**AI Infrastructure:**
Local LLM deployment (Ollama), vector database design, embedding model selection, document preprocessing pipelines

---

## What to Say If Asked About It in an Interview

**"What is RAG and why did you build this?"**
> RAG stands for Retrieval-Augmented Generation. Standard LLMs are trained on general data and hallucinate when asked about specific or niche topics. RAG solves this by retrieving relevant chunks from your own documents and passing them as context to the LLM, so it answers only from verified information. I built this to understand how real-world AI assistants — like document chatbots — actually work under the hood.

**"What was the most interesting technical challenge?"**
> Designing the chunking strategy was more nuanced than I expected. If chunks are too small, you lose context. Too large and you hit token limits and introduce noise. I used recursive character splitting with overlapping windows (50-char overlap) to make sure no information is cut off at chunk boundaries.

**"Why did you use a local LLM instead of OpenAI?"**
> Privacy and cost. Running Llama 3.2 locally via Ollama means no data leaves the machine and there are no per-token API charges. For real-world use cases with sensitive documents — like medical records, legal contracts, or internal company data — this architecture is far more practical than relying on a cloud API.

**"How would you scale this?"**
> For production scale, I would swap Chroma for a managed vector database like Pinecone or Weaviate, move the LLM to a cloud endpoint with GPU backing, add a proper REST API layer (FastAPI), implement re-ranking after retrieval for higher accuracy, and add evaluation metrics to track answer quality over time.
