from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

MODEL_NAME = "gpt-3.5-turbo"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

VECTOR_DB_DIR = "chroma_db"

COLLECTION_NAME = "pdf_docs"

DATA_DIR = "src/data"

RERANK_TOP_K = 4
INITIAL_RETRIEVAL_K = 10