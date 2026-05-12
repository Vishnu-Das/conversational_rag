import shutil
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data"


def ingest_documents():

    if Path("chroma_db").exists():
        print("Deleting existing ChromaDB...")
        shutil.rmtree("chroma_db")

    print("Loading PDF documents...")

    loader = DirectoryLoader(
        path=str(DATA_PATH),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    print("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings...")

    embeddings = OpenAIEmbeddings()

    print("Creating ChromaDB vector store...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="pdf_docs"
    )

    print("Ingestion complete!")


if __name__ == "__main__":
    ingest_documents()