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

<img width="1547" height="1238" alt="localhost_8501_ (2)" src="https://github.com/user-attachments/assets/7df5d2bf-f3cd-4efd-92e9-27b6e388bfe4" />

---

<img width="1557" height="1249" alt="localhost_8501_ (3)" src="https://github.com/user-attachments/assets/f094f50e-b7fe-4827-aefc-d325137da6e5" />

---

<img width="1564" height="1256" alt="localhost_8501_ (4)" src="https://github.com/user-attachments/assets/2b1ab3d0-9dbf-45ec-955f-d3e1ad2d8682" />

---

<img width="1550" height="973" alt="smith langchain com_o_de94ea1f-1328-4c89-b2de-15e49b9eb08e_projects_p_5ab235e2-224e-4cc6-b2f2-61a273f1b794_peek=20260516T104924Z019e3067-b812-7212-a24e-704f503738b3 peeked_trace=20260516T104921103687Z019e3067-abcf-" src="https://github.com/user-attachments/assets/39b0e105-02ca-42e6-b308-53db689c95a6" />

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
- Hybrid search using BM25
- CrossEncoder reranking
- Semantic Chunking

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

# Project Structure

```text
conversational_rag/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env
├── .env.example
├── README.md
├── pyproject.toml
├── uv.lock
│
├── chroma_db/
│   ├── chroma.sqlite3
│   └── ...
│
├── storage/
│   └── chat_memory1.db
│
├── eval/
│   ├── questions.json
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   └── evaluate_with_judge.py
│
├── notebooks/
│   └── experimentation.ipynb
│
├── tests/
│   ├── test_rag.py
│   ├── test_ingestion.py
│   └── test_vectorstore.py
│
└── src/
    │
    ├── __init__.py
    │
    ├── config.py
    ├── database.py
    ├── ingest.py
    ├── vectorstore.py
    │
    ├── data/
    │   ├── attention.pdf
    │   ├── Profile.pdf
    │   └── ...
    │
    ├── rag/
    │   │
    │   ├── __init__.py
    │   ├── llm.py
    │   ├── prompts.py
    │   ├── retrievers.py
    │   ├── pipeline.py
    │   ├── cache.py
    │   └── service.py
    │
    ├── reranker/
    │   ├── __init__.py
    │   └── cross_encoder.py
    │
    ├── helpers/
    │   └── deduplication.py
    │
    ├── services/
    │   ├── __init__.py
    │   ├── session_service.py
    │   └── upload_service.py
    │
    ├── ui/
    │   │
    │   ├── __init__.py
    │   ├── main.py
    │   ├── sidebar.py
    │   ├── welcome.py
    │   ├── chat.py
    │   └── styles.py
    │
    └── utils/
        └── citations.py

```

---

## Running with Docker Compose

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd conversational_rag
```

---

### 2. Create `.env`

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

---

### 3. Build and Start the Application

```bash
docker compose up --build
```

Open the app in your browser:

```text
http://localhost:8501
```

---

## PDF Storage

Place PDFs inside:

```text
src/data/
```

These files are mounted into the Docker container automatically.

---

## Automatic Ingestion

On first startup:

- If `chroma_db/` is empty
- The application automatically ingests all PDFs from `src/data/`

---

## Bulk Re-Ingestion

If you manually add new PDFs into `src/data/` after the container is already running, run:

```bash
docker compose exec conversational-rag \
.venv/bin/python -m src.ingest
```

This rebuilds the vector database with all available documents.

---

## Persistent Storage

The following directories are persisted locally using Docker volumes:

| Path | Purpose |
|---|---|
| `src/data/` | Uploaded PDFs |
| `chroma_db/` | Chroma vector database |
| `storage/` | SQLite chat history |

Your data remains available even after restarting containers.

---

## Stop the Application

```bash
docker compose down
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Vishnu-Das/conversational_rag
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
HF_TOKEN=your_huggingface_api_key (optional)
```

---

# Running the Project

## Step 1 — Add PDFs

### Place PDF files inside:

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
### Upload PDF

```text
file directly from the UI interface (works for single uploads)
```
---

## Step 2 — Run Document Ingestion

Generate embeddings and build the vector database (only for bulk document ingestion when placed directly in the data folder):

```bash
uv run python -m src.ingest
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

* "Summarize the resume."
* "What skills are mentioned?"
* "What projects are listed?"
* "Explain the experience section."
* "What technologies are used?"
* "Who am I?" (conversation-aware memory)

---

# Current Limitations

* No authentication
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
* Reranking CrossEncoder
* Semantic Chunking
* Document Deduplication
* Optimized runtime performance using caching:
  - Cached retrievers
  - Cached history-aware retrievers
  - Cached retrieval results
  - Cached reranker model
  - Cached document loading
* Refactored retriever creation and ingestion architecture
* incremental PDF ingestion pipeline
* PDF upload feature in Streamlit sidebar
* Single-document ingestion without rebuilding entire vector DB
* Automatic cache invalidation after new document ingestion
* Upload deduplication handling
* Uploader reset and upload success notifications
* Fixed repeated ingestion issues caused by Streamlit reruns
* Improved Streamlit session-state handling

  ---

# Future Improvements

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
* Metada filtering
* History-aware retrieval
* CrossEncoder reranking
* Context Compressor

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
