from typing import List, Optional

from langchain_core.documents import Document

from src.rag.retrieval.base import BaseRetrievalStrategy

from src.rag.retrievers import (
    get_history_aware_retriever
)


class HybridRetrievalStrategy(BaseRetrievalStrategy):
    """
    Current production retrieval strategy.

    Uses:
    - history-aware query rewriting
    - multi-query retrieval
    - vector retrieval
    - BM25 retrieval
    - ensemble retrieval
    """

    def retrieve(
        self,
        query: str,
        chat_history: list,
        selected_document: Optional[str] = None
    ) -> List[Document]:

        retriever = get_history_aware_retriever(
            selected_document
        )

        docs = retriever.invoke({
            "input": query,
            "chat_history": chat_history
        })

        return docs