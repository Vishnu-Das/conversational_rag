from src.config import (
    DEFAULT_RETRIEVAL_STRATEGY
)

from src.rag.models import (
    RouterResult
)

from src.rag.retrieval.factory import (
    RetrievalStrategyFactory
)

from src.rag.routing.base import (
    BaseRouterStrategy
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


class RuleBasedRouterStrategy(
    BaseRouterStrategy
):

    def route(
        self,
        query: str,
        selected_document: str | None = None
    ) -> RouterResult:

        query_lower = query.lower()

        if self._contains_any(
            query_lower,
            DOCUMENT_LEVEL_KEYWORDS
        ):

            return self._result(
                RetrievalStrategyFactory.PARENT_CHILD,
                "Document-level query detected.",
                1.0
            )

        if self._contains_any(
            query_lower,
            CONCEPTUAL_QUERY_KEYWORDS
        ):

            return self._result(
                RetrievalStrategyFactory.FUSION,
                "Conceptual or comparative query detected.",
                1.0
            )

        if self._starts_with_any(
            query_lower,
            FACT_LOOKUP_KEYWORDS
        ):

            return self._result(
                RetrievalStrategyFactory.HYBRID,
                "Factual lookup query detected.",
                1.0
            )

        if (
            selected_document
            and selected_document != "All Documents"
        ):

            return self._result(
                RetrievalStrategyFactory.PARENT_CHILD,
                "Specific document selected; using parent-child retrieval.",
                0.8
            )

        return self._result(
            DEFAULT_RETRIEVAL_STRATEGY,
            "No specific query pattern matched; using default strategy.",
            0.5
        )

    def _result(
        self,
        strategy: str,
        reason: str,
        confidence: float
    ) -> RouterResult:

        return RouterResult(
            strategy=strategy,
            reason=reason,
            confidence=confidence,
            router_type="rule_based"
        )

    def _contains_any(
        self,
        query: str,
        keywords: list[str]
    ) -> bool:

        return any(
            keyword in query
            for keyword in keywords
        )

    def _starts_with_any(
        self,
        query: str,
        keywords: list[str]
    ) -> bool:

        return any(
            query.startswith(keyword)
            for keyword in keywords
        )