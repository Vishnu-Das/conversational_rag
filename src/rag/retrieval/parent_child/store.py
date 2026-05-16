import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore

from src.config import (
    VECTOR_DB_DIR,
    PARENT_CHILD_COLLECTION_NAME,
    PARENT_DOCSTORE_DIR,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP
)


def get_parent_splitter():

    return RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )


def get_child_splitter():

    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )


def get_parent_child_vectorstore():

    embeddings = OpenAIEmbeddings()

    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        collection_name=PARENT_CHILD_COLLECTION_NAME,
        embedding_function=embeddings
    )


def get_parent_docstore():

    os.makedirs(
        PARENT_DOCSTORE_DIR,
        exist_ok=True
    )

    file_store = LocalFileStore(
        PARENT_DOCSTORE_DIR
    )

    return create_kv_docstore(
        file_store
    )