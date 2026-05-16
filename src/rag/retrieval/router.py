from src.rag.retrieval.factory import (
    RetrievalStrategyFactory
)

from src.config import (
    DEFAULT_RETRIEVAL_STRATEGY
)


DOCUMENT_LEVEL_KEYWORDS = [
    "summarize",
    "summary",
    "summarise",
    "overview",
    "key concepts",
    "important topics",
    "main ideas",
    "study notes",
    "explain this document",
    "explain the document",
]

FACT_LOOKUP_KEYWORDS = [
    "what is",
    "define",
    "who is",
    "when",
    "where",
    "which",
]


def is_document_level_query(
    query: str
) -> bool:

    query = query.lower()

    return any(
        keyword in query
        for keyword in DOCUMENT_LEVEL_KEYWORDS
    )


def is_fact_lookup_query(
    query: str
) -> bool:

    query = query.lower()

    return any(
        query.startswith(keyword)
        for keyword in FACT_LOOKUP_KEYWORDS
    )


def route_retrieval_strategy(
    query: str,
    selected_document: str = None
) -> str:

    if is_document_level_query(query):

        return RetrievalStrategyFactory.PARENT_CHILD

    if is_fact_lookup_query(query):

        return RetrievalStrategyFactory.HYBRID

    if (
        selected_document
        and selected_document != "All Documents"
    ):

        return RetrievalStrategyFactory.PARENT_CHILD

    return DEFAULT_RETRIEVAL_STRATEGY