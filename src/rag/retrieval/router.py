from src.rag.retrieval.factory import (
    RetrievalStrategyFactory
)

from src.config import (
    DEFAULT_RETRIEVAL_STRATEGY
)

from src.rag.models import RouterResult


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

CONCEPTUAL_QUERY_KEYWORDS = [
    "explain",
    "how",
    "why",
    "architecture",
    "workflow",
    "compare",
    "difference",
    "advantages",
    "tradeoffs",
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

def is_conceptual_query(
    query: str
) -> bool:

    query = query.lower()

    return any(
        keyword in query
        for keyword in CONCEPTUAL_QUERY_KEYWORDS
    )

def build_router_result(
    strategy: str,
    reason: str,
    confidence: float,
    router_type: str = "rule_based"
) -> RouterResult:

    return RouterResult(
        strategy=strategy,
        reason=reason,
        confidence=confidence,
        router_type=router_type
    )


def route_retrieval_strategy(
    query: str,
    selected_document: str = None
) -> RouterResult:

    if is_document_level_query(query):

        return build_router_result(
            RetrievalStrategyFactory.PARENT_CHILD,
            "Document-level query detected.",
            1.0
        )

    if is_conceptual_query(query):

        return build_router_result(
            RetrievalStrategyFactory.FUSION,
            "Conceptual or comparative query detected.",
            1.0
        )

    if is_fact_lookup_query(query):

        return build_router_result(
            RetrievalStrategyFactory.HYBRID,
            "Factual lookup query detected.",
            1.0
        )

    if (
        selected_document
        and selected_document != "All Documents"
    ):

        return build_router_result(
            RetrievalStrategyFactory.PARENT_CHILD,
            "Specific document selected; using parent-child retrieval.",
            0.8
        )

    return build_router_result(
        DEFAULT_RETRIEVAL_STRATEGY,
        "No specific query pattern matched; using default strategy.",
        0.5
    )