import os

from dotenv import load_dotenv

load_dotenv()

# =========================
# API KEYS
# =========================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

# =========================
# MODEL CONFIG
# =========================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gpt-3.5-turbo"
)

RERANKING_MODEL_NAME = os.getenv(
    "RERANKING_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# =========================
# CHUNKING CONFIG
# =========================

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", 1000)
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 200)
)

# =========================
# RETRIEVAL CONFIG
# =========================

INITIAL_RETRIEVAL_K = int(
    os.getenv("INITIAL_RETRIEVAL_K", 10)
)

RERANK_TOP_K = int(
    os.getenv("RERANK_TOP_K", 4)
)

# =========================
# VECTOR DATABASE
# =========================

VECTOR_DB_DIR = os.getenv(
    "VECTOR_DB_DIR",
    "chroma_db"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "pdf_docs"
)

# =========================
# DATA DIRECTORY
# =========================

DATA_DIR = os.getenv(
    "DATA_DIR",
    "src/data"
)

# =========================
# CHAT DIRECTORY
# =========================

CHAT_DB_PATH = os.getenv(
    "CHAT_DB_PATH",
    "storage/chat_memory1.db"
)

# =========================
# Parent-child retrieval configuration
# =========================
PARENT_CHUNK_SIZE = 2500
PARENT_CHUNK_OVERLAP = 200

CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

PARENT_CHILD_COLLECTION_NAME = (
    "parent_child_docs"
)

PARENT_DOCSTORE_DIR = (
    "storage/parent_docstore"
)

# =========================
# Retrieval strategy
# =========================
RETRIEVAL_STRATEGY = os.getenv(
    "RETRIEVAL_STRATEGY",
    "hybrid"
)

PARENT_CHILD_RETRIEVAL_K = 4