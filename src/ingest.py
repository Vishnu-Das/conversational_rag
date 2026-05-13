from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from src.vectorstore import (
    load_and_split_documents
)

from src.config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME
)


def ingest_documents():

    chunks = load_and_split_documents()

    embeddings = OpenAIEmbeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Documents ingested successfully.")


if __name__ == "__main__":
    ingest_documents()