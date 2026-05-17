from typing import List, Optional

from langchain_core.documents import Document
from langsmith import traceable

from src.rag.retrieval.base import (
    BaseRetrievalStrategy
)

from src.rag.retrieval.hybrid.strategy import (
    HybridRetrievalStrategy
)

from src.rag.retrieval.parent_child.strategy import (
    ParentChildRetrievalStrategy
)


def deduplicate_retrieved_docs(
    documents: List[Document]
) -> List[Document]:

    seen = set()
    unique_docs = []

    for doc in documents:

        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content[:200]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


class FusionRetrievalStrategy(
    BaseRetrievalStrategy
):

    @traceable(
        name="Fusion Retrieval",
        run_type="retriever"
    )
    def retrieve(
        self,
        query: str,
        chat_history: list,
        selected_document: Optional[str] = None
    ) -> List[Document]:

        hybrid_strategy = HybridRetrievalStrategy()

        parent_child_strategy = ParentChildRetrievalStrategy()

        hybrid_docs = hybrid_strategy.retrieve(
            query=query,
            chat_history=chat_history,
            selected_document=selected_document
        )

        parent_child_docs = parent_child_strategy.retrieve(
            query=query,
            chat_history=chat_history,
            selected_document=selected_document
        )

        combined_docs = [
            *hybrid_docs,
            *parent_child_docs
        ]

        return deduplicate_retrieved_docs(
            combined_docs
        )