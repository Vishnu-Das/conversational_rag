from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from src.vectorstore import (
    load_and_split_documents,
    load_single_document,
    split_documents,
    load_vectorstore
)

from src.config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME
)


def ingest_documents(): ## This function loads and splits documents, creates embeddings, and stores them in the Chroma vectorstore for later retrieval.

    chunks = load_and_split_documents()

    embeddings = OpenAIEmbeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Documents ingested successfully.")

def ingest_single_document(file_path):
    vectorstore = load_vectorstore()
    documents = load_single_document(
        file_path
    )
    chunks = split_documents(documents)
    vectorstore.add_documents(chunks)
    print(f"Ingested: {file_path}")


if __name__ == "__main__":
    ingest_documents()