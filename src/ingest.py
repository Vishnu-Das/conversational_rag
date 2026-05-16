from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.vectorstore import (
    load_documents,
    load_and_split_documents,
    load_single_document,
    split_documents,
    load_vectorstore
)

from src.config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME
)

from src.rag.retrieval.parent_child.ingestion import (
    ingest_parent_child_documents
)


def ingest_documents():
    """
    Bulk ingestion.

    Builds:
    1. Standard semantic-chunk Chroma index for hybrid retrieval
    2. Parent-child index for parent-child retrieval
    """

    chunks = load_and_split_documents()

    embeddings = OpenAIEmbeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    raw_documents = load_documents()

    ingest_parent_child_documents(
        raw_documents
    )

    print("Documents ingested successfully.")


def ingest_single_document(file_path):
    """
    Incremental ingestion for uploaded PDF.

    Adds:
    1. Semantic chunks to existing hybrid vectorstore
    2. Raw document pages to parent-child index
    """

    vectorstore = load_vectorstore()

    raw_documents = load_single_document(
        file_path
    )

    chunks = split_documents(
        raw_documents
    )

    vectorstore.add_documents(
        chunks
    )

    ingest_parent_child_documents(
        raw_documents
    )

    print(f"Ingested: {file_path}")


if __name__ == "__main__":
    ingest_documents()