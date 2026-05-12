# Conversational AI RAG Application

A production-style Conversational AI application built using LangChain, OpenAI, Streamlit, ChromaDB, and SQLite.

This project enables users to chat with PDF documents using Retrieval Augmented Generation (RAG) while maintaining persistent conversational memory across sessions.

---

# Features

- Conversational chat interface using Streamlit
- Retrieval Augmented Generation (RAG)
- Persistent conversational memory using SQLite
- Semantic search over PDF documents
- Multi-session chat support
- Persistent vector database using ChromaDB
- OpenAI embeddings + LLM integration
- Multi-document PDF support
- Separate ingestion pipeline using `ingest.py`
- Production-style modular architecture

---

# Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- SQLite
- UV Package Manager

---

# Project Structure

```text
conversational_rag/
│
├── app.py
├── .env
├── pyproject.toml
├── uv.lock
├── chat_memory1.db
├── chroma_db/
│
├── src/
│   ├── __init__.py
│   ├── conversationalAI.py
│   ├── rag.py
│   ├── vectorstore.py
│   ├── database.py
│   ├── ingest.py
│   │
│   └── data/
│       └── *.pdf
```

---

# Architecture Overview

This project follows a production-style RAG architecture by separating:

- Document ingestion pipeline
- Vector database creation
- Conversational runtime
- Persistent memory layer

This improves:

- startup performance
- scalability
- maintainability
- extensibility

---

# How It Works

## 1. Document Ingestion

PDF documents are loaded from:

```text
src/data/
```

using LangChain `DirectoryLoader`.

The ingestion pipeline:

1. Loads PDFs
2. Splits documents into chunks
3. Generates embeddings
4. Stores vectors in ChromaDB

This process is handled separately by:

```text
src/ingest.py
```

---

## 2. Text Chunking

Documents are split into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

to improve semantic retrieval quality.

---

## 3. Embeddings

Text chunks are converted into vector embeddings using:

```python
OpenAIEmbeddings
```

---

## 4. Vector Database

Embeddings are stored persistently in:

```text
chroma_db/
```

using ChromaDB.

---

## 5. Retrieval

For each user query:

1. Relevant chunks are retrieved
2. Context is injected into the prompt
3. LLM generates context-aware responses

---

## 6. Conversational Memory

Chat history is stored in SQLite:

```text
chat_memory1.db
```

allowing persistent multi-session conversations.

---

# Installation

## Clone Repository

```bash
git clone <your_repo_url>
cd conversational_rag
```

---

## Create Virtual Environment

Using UV:

```bash
uv venv
```

Activate environment:

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
uv sync
```

or manually:

```bash
uv add streamlit
uv add langchain
uv add langchain-community
uv add langchain-openai
uv add langchain-text-splitters
uv add chromadb
uv add pypdf
uv add python-dotenv
uv add torch torchvision
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

# Running the Project

## Step 1 — Add PDFs

Place PDF files inside:

```text
src/data/
```

Example:

```text
src/data/
├── resume.pdf
├── notes.pdf
└── research/
    └── ai.pdf
```

---

## Step 2 — Run Document Ingestion

Generate embeddings and build the vector database:

```bash
uv run python src/ingest.py
```

This creates:

```text
chroma_db/
```

---

## Step 3 — Start the Application

```bash
uv run streamlit run app.py
```

Application will start on:

```text
http://localhost:8501
```

---

# Example Questions

- "Summarize the resume"
- "What skills are mentioned?"
- "What projects are listed?"
- "Explain the experience section"
- "What technologies are used?"
- "Who am I?" (conversation-aware memory)

---

# Current Limitations

- No streaming responses
- No authentication
- No metadata filtering
- Local deployment only
- No reranking pipeline
- No hybrid search

---

# Future Improvements

- Streaming token responses
- Hybrid search
- Semantic chunking
- Multi-query retrieval
- Reranking
- Source citations
- Docker deployment
- Cloud deployment
- Multi-user authentication
- Long-term memory
- Conversation summarization
- Agentic workflows

---

# Learning Concepts Covered

This project demonstrates:

- Retrieval Augmented Generation (RAG)
- Embeddings
- Vector Databases
- Semantic Search
- Conversational Memory
- Prompt Engineering
- LangChain Pipelines
- Persistent Chat Storage
- Production-style ingestion pipelines
- Modular AI application architecture

---

# License

MIT License

---

# Acknowledgements

Built using:

- LangChain
- Streamlit
- OpenAI
- ChromaDB
- UV Package Manager