# Conversational RAG

Conversational RAG is an intelligent document assistant for grounded question answering and conversational interaction across PDF documents.

The system combines hybrid retrieval, multi-query search expansion, and cross-encoder reranking to improve retrieval accuracy and response quality.

It supports document-specific querying, streaming responses, source citations, automated evaluation pipelines, observability with tracing, and containerized deployment.

The project focuses on production-oriented RAG engineering concepts such as retrieval optimization, answer grounding, hallucination reduction, evaluation, and debugging workflows beyond basic vector search systems.

---

<img width="1547" height="1238" alt="localhost_8501_ (2)" src="https://github.com/user-attachments/assets/7df5d2bf-f3cd-4efd-92e9-27b6e388bfe4" />

---

<img width="1557" height="1249" alt="localhost_8501_ (3)" src="https://github.com/user-attachments/assets/f094f50e-b7fe-4827-aefc-d325137da6e5" />

---

<img width="1564" height="1256" alt="localhost_8501_ (4)" src="https://github.com/user-attachments/assets/2b1ab3d0-9dbf-45ec-955f-d3e1ad2d8682" />

---

## Features

- Conversational RAG with chat history awareness
- Hybrid Retrieval (Vector Search + BM25)
- MultiQuery Retrieval for improved recall
- Cross-Encoder Reranking
- Source citations with previews
- Streaming responses
- PDF upload and ingestion
- LangSmith observability and tracing
- Automated evaluation pipelines
- Dockerized deployment

---

## Architecture

```text
User Query
    ↓
History-Aware Query Rewriting
    ↓
MultiQuery Retrieval
    ↓
Hybrid Retrieval
(Vector + BM25)
    ↓
Deduplication
    ↓
Cross-Encoder Reranking
    ↓
Context Construction
    ↓
LLM Response Generation
    ↓
Streaming UI + Source Citations
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT |
| Framework | LangChain |
| Vector DB | ChromaDB |
| UI | Streamlit |
| Reranking | SentenceTransformers CrossEncoder |
| Observability | LangSmith |
| Containerization | Docker |
| Evaluation | Custom RAG Evaluation Pipelines |

---

## Evaluation Pipelines

The project includes automated evaluation scripts for validating retrieval and answer quality.

### Retrieval Evaluation

Validates:
- retrieval accuracy
- document filtering
- reranking quality
- expected keyword coverage

```bash
uv run python eval/evaluate_retrieval.py
```

---

### Answer Evaluation

Validates:
- generated answer quality
- semantic coverage
- source grounding

```bash
uv run python eval/evaluate_answers.py
```

---

### LLM-as-a-Judge Evaluation

Uses an LLM to evaluate:
- correctness
- groundedness
- hallucination risk
- completeness

```bash
uv run python eval/evaluate_with_judge.py
```

---

## Observability

Integrated LangSmith tracing for:

- query rewriting
- retrieval inspection
- reranking analysis
- prompt debugging
- token usage
- latency analysis
- end-to-end chain tracing

---

## Running Locally

### 1. Clone Repository

```bash
git clone https://github.com/Vishnu-Das/conversational_rag.git

cd conversational_rag
```

---

### 2. Create `.env`

```env
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token

LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=conversational-rag
```

---

### 3. Install Dependencies

```bash
uv sync
```

---

### 4. Run Application

```bash
uv run streamlit run app.py
```

---

## Running with Docker

### Build and Start

```bash
docker compose up --build
```

Application:

```text
http://localhost:8501
```

---

## Document Ingestion

Place PDFs inside:

```text
src/data/
```

If `chroma_db/` is empty, ingestion runs automatically during startup.

---

### Manual Bulk Re-Ingestion

```bash
docker compose exec conversational-rag \
.venv/bin/python -m src.ingest
```

---

## Project Structure

```text
src/
├── rag/            # Retrieval and generation pipeline
├── reranker/       # Cross-encoder reranking
├── ui/             # Streamlit UI
├── services/       # Upload/session services
├── helpers/        # Utility helpers
├── utils/          # Citation formatting
├── data/           # PDF documents
└── vectorstore.py  # ChromaDB integration

eval/
├── evaluate_retrieval.py
├── evaluate_answers.py
└── evaluate_with_judge.py
```

---

## Example Capabilities

- Ask questions across uploaded PDFs
- Restrict queries to specific documents
- Generate document summaries
- Extract key concepts
- View cited source chunks
- Analyze retrieval quality using LangSmith

---

## Future Improvements

- Parent-child retrieval
- Contextual compression
- Async ingestion
- FastAPI backend
- Feedback collection
- GraphRAG
- Multi-user authentication

---

## License

MIT License
