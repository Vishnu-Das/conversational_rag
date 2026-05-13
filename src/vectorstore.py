from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
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


def load_and_split_documents(): ## This function loads documents from the specified directory, splits them into chunks, and returns the list of chunks.

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


def load_vectorstore(): ## This function initializes the Chroma vectorstore with OpenAI embeddings and returns the vectorstore instance.

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectorstore

def get_available_documents(): ## This function retrieves the list of available documents from the loaded and split chunks, extracts their source filenames, and returns a sorted list of unique document names.

    chunks = load_and_split_documents()

    sources = set()

    for chunk in chunks:

        source = chunk.metadata.get("source")

        if source:

            filename = source.split("\\")[-1]

            sources.add(filename)

    return sorted(list(sources))