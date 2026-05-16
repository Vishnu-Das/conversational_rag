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

from langchain_core.documents import Document

from src.config import (
    # CHUNK_SIZE,
    # CHUNK_OVERLAP,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    DATA_DIR
)

embeddings = OpenAIEmbeddings()


def load_and_split_documents(): ## This function loads documents from the specified directory, splits them into chunks, and returns the list of chunks.

    documents = load_documents()

    # text_splitter = (
    #     RecursiveCharacterTextSplitter(
    #         chunk_size=CHUNK_SIZE,
    #         chunk_overlap=CHUNK_OVERLAP
    #     )
    # )

    return split_documents(documents)


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

def load_documents_from_vectorstore():

    vectorstore = load_vectorstore()
    data = vectorstore.get(
        include=["documents", "metadatas"]
    )
    documents = [
        Document(
            page_content=doc,
            metadata=meta
        )
        for doc, meta in zip(
            data["documents"],
            data["metadatas"]
        )
    ]
    return documents

def load_single_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(documents):

    text_splitter = SemanticChunker(
        embeddings=embeddings
    )
    return text_splitter.split_documents(
        documents
    )

def load_documents():

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    return loader.load()