from src.rag.models import RouterResult


def test_router_result_stores_values():
    result = RouterResult(
        strategy="hybrid",
        reason="Factual lookup query detected.",
        confidence=1.0,
        router_type="rule_based",
    )

    assert result.strategy == "hybrid"
    assert result.reason == "Factual lookup query detected."
    assert result.confidence == 1.0
    assert result.router_type == "rule_based"


def test_router_result_default_router_type_is_rule_based():
    result = RouterResult(
        strategy="hybrid",
        reason="Default test reason.",
    )

    assert result.router_type == "rule_based"
    assert result.confidence is None