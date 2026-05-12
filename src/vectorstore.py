from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def load_vectorstore():

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="pdf_docs"
    )

    return vectorstore