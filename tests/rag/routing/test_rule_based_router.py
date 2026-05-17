from src.config import DEFAULT_RETRIEVAL_STRATEGY
from src.rag.retrieval.factory import RetrievalStrategyFactory
from src.rag.routing.rule_based import RuleBasedRouterStrategy


def test_routes_factual_query_to_hybrid():
    router = RuleBasedRouterStrategy()

    result = router.route(
        query="What is multi-head attention?",
        selected_document=None,
    )

    assert result.strategy == RetrievalStrategyFactory.HYBRID
    assert result.router_type == "rule_based"
    assert result.confidence == 1.0
    assert "Factual lookup" in result.reason


def test_routes_summary_query_to_parent_child():
    router = RuleBasedRouterStrategy()

    result = router.route(
        query="Summarize this document",
        selected_document=None,
    )

    assert result.strategy == RetrievalStrategyFactory.PARENT_CHILD
    assert result.router_type == "rule_based"
    assert result.confidence == 1.0
    assert "Document-level" in result.reason


def test_routes_conceptual_query_to_fusion():
    router = RuleBasedRouterStrategy()

    result = router.route(
        query="Explain transformer architecture",
        selected_document=None,
    )

    assert result.strategy == RetrievalStrategyFactory.FUSION
    assert result.router_type == "rule_based"
    assert result.confidence == 1.0
    assert "Conceptual" in result.reason


def test_routes_how_query_to_fusion():
    router = RuleBasedRouterStrategy()

    result = router.route(
        query="How does self-attention work?",
        selected_document=None,
    )

    assert result.strategy == RetrievalStrategyFactory.FUSION
    assert result.router_type == "rule_based"


def test_specific_document_selected_uses_parent_child_when_no_pattern_matches():
    router = RuleBasedRouterStrategy()

    result = router.route(
        query="Tell me more about this",
        selected_document="attention.pdf",
    )

    assert result.strategy == RetrievalStrategyFactory.PARENT_CHILD
    assert result.router_type == "rule_based"
    assert result.confidence == 0.8
    assert "Specific document selected" in result.reason


def test_all_documents_does_not_trigger_specific_document_routing():
    router = RuleBasedRouterStrategy()

    result = router.route(
        query="Tell me more about this",
        selected_document="All Documents",
    )

    assert result.strategy == DEFAULT_RETRIEVAL_STRATEGY
    assert result.router_type == "rule_based"
    assert result.confidence == 0.5


def test_unknown_query_uses_default_strategy():
    router = RuleBasedRouterStrategy()

    result = router.route(
        query="random unrelated text",
        selected_document=None,
    )

    assert result.strategy == DEFAULT_RETRIEVAL_STRATEGY
    assert result.router_type == "rule_based"
    assert result.confidence == 0.5
    assert "default strategy" in result.reason