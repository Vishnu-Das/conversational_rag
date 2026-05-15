from functools import lru_cache

from src.rag.retrievers import (
    get_history_aware_retriever,
    get_retriever,
    get_all_documents
)


@lru_cache(maxsize=100)
def cached_retrieval(
    user_input: str,
    selected_document: str
):

    history_aware_retriever = (
        get_history_aware_retriever(
            selected_document
        )
    )

    docs = history_aware_retriever.invoke({
        "input": user_input,
        "chat_history": []
    })

    return docs


def reset_rag_caches():
    get_retriever.cache_clear()
    get_history_aware_retriever.cache_clear()
    cached_retrieval.cache_clear()
    get_all_documents.clear()