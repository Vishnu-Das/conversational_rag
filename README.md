# Conversational AI RAG Application

A production-style Conversational AI application built using LangChain, OpenAI, Streamlit, ChromaDB, and SQLite.

This project enables users to chat with PDF documents using Retrieval Augmented Generation (RAG) while maintaining persistent conversational memory across sessions.

The application supports:

* multi-document PDF ingestion
* semantic retrieval
* streaming AI responses
* source citations
* persistent chat memory
* modular RAG architecture

---

# Features

## Core RAG Features

- Conversational Retrieval-Augmented Generation (RAG)
- Semantic search over PDF documents
- Persistent vector database using ChromaDB
- OpenAI embeddings + LLM integration
- Multi-PDF support
- Metadata-based document filtering
- Context-aware retrieval
- Source citations in responses

---

## Conversational Features

- Persistent chat memory using SQLite
- Multi-session conversations
- History-aware retrieval
- Follow-up question understanding
- ChatGPT-style streaming responses
- Typing cursor effect

---

## UI Features

- Streamlit chat interface
- Dynamic sidebar chat previews
- Single-click chat switching
- Document selection filter
- Responsive layout
- Fixed-width sidebar UI improvements

---

## Architecture Features

- Modular project structure
- Separate ingestion pipeline
- Persistent vector indexing
- Clean separation of concerns
- Scalable code organization
- Production-style architecture

---

# Tech Stack

* Python
* Streamlit
* LangChain
* OpenAI API
* ChromaDB
* SQLite
* UV Package Manager

---

# Architecture

```text
User Query
    ↓
History-Aware Query Rewriter
    ↓
Metadata Filtered Retriever
    ↓
Chroma Vector Search
    ↓
Context Injection
    ↓
OpenAI LLM
    ↓
Streaming Response
    ↓
Persistent Memory Storage
```
# Project Structure

```text
conversational_rag/
│
├── app.py
├── .env
├── pyproject.toml
├── uv.lock
├── README.md
├── chat_memory1.db
├── chroma_db/
│
├── src/
│   ├── __init__.py
│   │
│   ├── conversationalAI.py
│   ├── rag.py
│   ├── vectorstore.py
│   ├── database.py
│   ├── citations.py
│   ├── ingest.py
│   │
│   └── data/
│       ├── profile.pdf
│       ├── attention.pdf
│       └── *.pdf

---

# Architecture Overview

This project follows a production-style RAG architecture by separating:

* document ingestion
* vector indexing
* conversational runtime
* memory persistence
* citation formatting

This improves:

* startup performance
* scalability
* maintainability
* extensibility

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

## 6. Streaming Responses

Responses are streamed token-by-token for a real-time conversational experience similar to ChatGPT.

Implemented using:

```python
ChatOpenAI(streaming=True)
```

and Streamlit dynamic rendering.

---

## 7. Source Citations

Retrieved document sources are displayed alongside responses.

Citations include:

* source document
* page number
* chunk preview

Citation formatting logic is separated into:

```text
src/utils/citations.py
```

---

## 8. Conversational Memory

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

* "Summarize the resume"
* "What skills are mentioned?"
* "What projects are listed?"
* "Explain the experience section"
* "What technologies are used?"
* "Who am I?" (conversation-aware memory)

---

# Current Limitations

* No authentication
* No hybrid retrieval
* No reranking pipeline
* Local deployment only

---

# Features Covered

* Streaming Responses
* Typing Cursor Effect
* Source Citations
* Metadata Filtering
* Persistent Conversational Memory
* Multi-session Chat Support
* History-aware Retrieval
* Persistent Chroma Vector Database
* Modular Production-style Architecture
* Multi-PDF Support
* Separate Ingestion Pipeline
* Sidebar Chat Management
* Dynamic Chat Previews
* SQLite Chat Persistence
* ChromaDB Persistence
* Production-style Project Structure
* Hybrid Search BM25 + vector

  ---

# Limitations
* No authentication
* No hybrid retrieval
* No reranking pipeline
* Local deployment only

---

# Future Improvements

* Semantic chunking
* Multi-query retrieval (partially covered)
* Reranking
* Docker deployment
* FastAPI backend
* Cloud deployment
* Multi-user authentication
* Long-term memory (partially covered)
* Agentic workflows
* Evaluation pipeline

---

# Learning Concepts Covered

This project demonstrates:

* Retrieval Augmented Generation (RAG)
* Embeddings
* Vector Databases
* Semantic Search
* Conversational Memory
* Streaming LLM Responses
* Source Citation Systems
* Prompt Engineering
* LangChain Pipelines
* Persistent Chat Storage
* Production-style ingestion pipelines
* Modular AI application architecture
* Metada fitlering
* History-aware retrieval

---

# License

MIT License

---

# Acknowledgements

Built using:

* LangChain
* Streamlit
* OpenAI
* ChromaDB
* UV Package Manager

---

# References & Inspiration

This project architecture and improvements were inspired by modern RAG engineering patterns and production-ready conversational AI systems. ([Go Packages][1])

[1]: https://pkg.go.dev/github.com/zgsm-ai/chat-rag?utm_source=chatgpt.com "chat-rag command - github.com/zgsm-ai/chat-rag - Go Packages"
