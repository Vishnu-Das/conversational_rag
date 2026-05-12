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
- Modular project structure
- Support for multiple PDFs
- Production-style architecture

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
│   │
│   └── data/
│       └── *.pdf
```

---

# How It Works

## 1. Document Loading

PDF documents are loaded from:

```text
src/data/
```

using LangChain `DirectoryLoader`.

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

# Running the Application

```bash
uv run streamlit run app.py
```

Application will start on:

```text
http://localhost:8501
```

---

# Adding PDFs

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

# Rebuilding the Vector Database

If new PDFs are added, delete the existing ChromaDB directory.

### Windows

```powershell
rmdir /s chroma_db
```

### Linux / Mac

```bash
rm -rf chroma_db
```

Then rerun the application.

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
- Re-indexing required for new documents
- Local deployment only

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
- Multi-user support
- Long-term memory
- Conversation summarization

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
- AI Application Architecture

---

# License

MIT License

---

# Acknowledgements

- LangChain
- Streamlit
- OpenAI
- ChromaDB
- UV Package Manager
