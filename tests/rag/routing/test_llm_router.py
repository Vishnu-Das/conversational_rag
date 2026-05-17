from unittest.mock import Mock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from src.rag.retrieval.factory import RetrievalStrategyFactory
from src.rag.routing.llm import LLMRouterStrategy
from src.rag.routing.schemas import LLMRouterOutput


@patch("src.rag.routing.llm.RuleBasedRouterStrategy")
@patch("src.rag.routing.llm.ChatOpenAI")
def test_llm_router_returns_structured_router_result(
    mock_chat_openai,
    mock_rule_based_router,
):
    mock_llm = Mock()

    structured_output = LLMRouterOutput(
        strategy=RetrievalStrategyFactory.FUSION,
        reason="The query asks for conceptual explanation.",
        confidence=0.9,
    )

    mock_structured_llm = RunnableLambda(lambda _: structured_output)

    mock_chat_openai.return_value = mock_llm
    mock_llm.with_structured_output.return_value = mock_structured_llm

    router = LLMRouterStrategy()

    result = router.route(
        query="Explain transformer architecture",
        selected_document=None,
    )

    assert result.strategy == RetrievalStrategyFactory.FUSION
    assert result.reason == "The query asks for conceptual explanation."
    assert result.confidence == 0.9
    assert result.router_type == "llm"

    mock_llm.with_structured_output.assert_called_once_with(LLMRouterOutput)
    mock_rule_based_router.assert_called_once()


@patch("src.rag.routing.llm.RuleBasedRouterStrategy")
@patch("src.rag.routing.llm.ChatOpenAI")
def test_llm_router_falls_back_when_llm_raises_error(
    mock_chat_openai,
    mock_rule_based_router,
):
    mock_llm = Mock()

    def raise_llm_error(_):
        raise Exception("LLM unavailable")

    mock_structured_llm = RunnableLambda(raise_llm_error)

    mock_chat_openai.return_value = mock_llm
    mock_llm.with_structured_output.return_value = mock_structured_llm

    mock_fallback_result = Mock()
    mock_fallback_result.strategy = RetrievalStrategyFactory.HYBRID
    mock_fallback_result.reason = "Factual lookup query detected."
    mock_fallback_result.confidence = 1.0
    mock_fallback_result.router_type = "rule_based"

    mock_fallback_router = Mock()
    mock_fallback_router.route.return_value = mock_fallback_result
    mock_rule_based_router.return_value = mock_fallback_router

    router = LLMRouterStrategy()

    result = router.route(
        query="What is attention?",
        selected_document=None,
    )

    assert result.strategy == RetrievalStrategyFactory.HYBRID
    assert result.confidence == 1.0
    assert result.router_type == "llm_fallback"
    assert "LLM router failed" in result.reason
    assert "Fallback used" in result.reason
    assert "LLM unavailable" in result.reason
    assert "Fallback reason: Factual lookup query detected." in result.reason

    mock_fallback_router.route.assert_called_once_with(
        query="What is attention?",
        selected_document=None,
    )


@patch("src.rag.routing.llm.RuleBasedRouterStrategy")
@patch("src.rag.routing.llm.ChatOpenAI")
def test_llm_router_falls_back_when_strategy_is_invalid(
    mock_chat_openai,
    mock_rule_based_router,
):
    mock_llm = Mock()

    invalid_output = LLMRouterOutput(
        strategy="invalid_strategy",
        reason="Bad model output.",
        confidence=0.4,
    )

    mock_structured_llm = RunnableLambda(lambda _: invalid_output)

    mock_chat_openai.return_value = mock_llm
    mock_llm.with_structured_output.return_value = mock_structured_llm

    mock_fallback_result = Mock()
    mock_fallback_result.strategy = RetrievalStrategyFactory.PARENT_CHILD
    mock_fallback_result.reason = "Document-level query detected."
    mock_fallback_result.confidence = 1.0
    mock_fallback_result.router_type = "rule_based"

    mock_fallback_router = Mock()
    mock_fallback_router.route.return_value = mock_fallback_result
    mock_rule_based_router.return_value = mock_fallback_router

    router = LLMRouterStrategy()

    result = router.route(
        query="Summarize this document",
        selected_document="attention.pdf",
    )

    assert result.strategy == RetrievalStrategyFactory.PARENT_CHILD
    assert result.confidence == 1.0
    assert result.router_type == "llm_fallback"
    assert "Invalid strategy: invalid_strategy" in result.reason
    assert "Fallback reason: Document-level query detected." in result.reason

    mock_fallback_router.route.assert_called_once_with(
        query="Summarize this document",
        selected_document="attention.pdf",
    )