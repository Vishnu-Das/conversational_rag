from langchain_classic.retrievers import (
    ParentDocumentRetriever
)

from src.rag.retrieval.parent_child.store import (
    get_parent_splitter,
    get_child_splitter,
    get_parent_child_vectorstore,
    get_parent_docstore
)


def get_parent_child_retriever():

    return ParentDocumentRetriever(
        vectorstore=get_parent_child_vectorstore(),
        docstore=get_parent_docstore(),
        child_splitter=get_child_splitter(),
        parent_splitter=get_parent_splitter()
    )


def ingest_parent_child_documents(
    documents
):

    if not documents:
        return
    retriever = get_parent_child_retriever()
    retriever.add_documents(
        documents
    )