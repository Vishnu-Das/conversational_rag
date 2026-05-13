from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path


def load_vectorstore():

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="pdf_docs"
    )

    return vectorstore

def get_available_documents():

    data_dir = Path("src/data")
    pdfs = []
    for file in data_dir.rglob("*.pdf"):
        pdfs.append(file.name)
    return sorted(pdfs)