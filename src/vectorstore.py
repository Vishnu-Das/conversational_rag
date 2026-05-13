import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

# from langchain_text_splitters import (
#     RecursiveCharacterTextSplitter
# )

from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    DATA_DIR
)

embeddings = OpenAIEmbeddings()


def load_and_split_documents(): ## This function loads documents from the specified directory, splits them into chunks, and returns the list of chunks.

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    # text_splitter = (
    #     RecursiveCharacterTextSplitter(
    #         chunk_size=CHUNK_SIZE,
    #         chunk_overlap=CHUNK_OVERLAP
    #     )
    # )

    text_splitter = SemanticChunker(embeddings=embeddings)

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


def load_vectorstore(): ## This function initializes the Chroma vectorstore with OpenAI embeddings and returns the vectorstore instance.

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


def get_available_documents():

    documents = []

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".pdf"):
                documents.append(file)

    return sorted(documents)