from typing import List, Optional

from langchain_core.documents import Document
from langsmith import traceable

from src.rag.retrieval.base import BaseRetrievalStrategy

from src.rag.retrieval.parent_child.ingestion import (
    get_parent_child_retriever
)

from src.rag.retrieval.parent_child.store import (
    get_parent_child_vectorstore
)

from src.config import (
    PARENT_CHILD_RETRIEVAL_K
)


def normalize_source_name(source: str) -> str:

    if not source:
        return ""

    source = source.replace("\\", "/")

    return source.split("/")[-1]


def get_matching_sources(
    selected_document: str
):

    vectorstore = get_parent_child_vectorstore()

    data = vectorstore.get(
        include=["metadatas"]
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    matching_sources = list({
        metadata.get("source")
        for metadata in metadatas
        if normalize_source_name(
            metadata.get("source", "")
        ) == selected_document
    })

    return matching_sources


class ParentChildRetrievalStrategy(
    BaseRetrievalStrategy
):
    """
    Parent-child retrieval strategy.

    Retrieves small child chunks from Chroma,
    then returns larger parent chunks from docstore.
    """

    @traceable(
        name="Parent Child Retrieval",
        run_type="retriever"
    )
    def retrieve(
        self,
        query: str,
        chat_history: list,
        selected_document: Optional[str] = None
    ) -> List[Document]:

        _ = chat_history

        retriever = get_parent_child_retriever()

        search_kwargs = {
            "k": PARENT_CHILD_RETRIEVAL_K
        }

        if (
            selected_document
            and selected_document != "All Documents"
        ):

            matching_sources = get_matching_sources(
                selected_document
            )

            if not matching_sources:
                return []

            search_kwargs["filter"] = {
                "source": {
                    "$in": matching_sources
                }
            }

        retriever.search_kwargs = search_kwargs

        docs = retriever.invoke(
            query
        )

        for doc in docs:
            doc.metadata["retrieval_strategy"] = "parent_child"
            doc.metadata["retrieval_source"] = "parent_child"

        return docs