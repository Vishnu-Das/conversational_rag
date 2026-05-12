from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data"


def load_vectorstore():

    embeddings = OpenAIEmbeddings()

    persist_directory = "chroma_db"

    # Load existing vector DB
    if Path(persist_directory).exists():

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="pdf_docs"
        )

        return vectorstore

    # Load ALL PDFs
    loader = DirectoryLoader(
        path=str(DATA_PATH),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="pdf_docs"
    )

    return vectorstore