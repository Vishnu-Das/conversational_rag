from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from src.rag.service import stream_response


def make_ranked_doc(document, score=0.95, retrieval_source="hybrid"):
    return SimpleNamespace(
        document=document,
        score=score,
        retrieval_source=retrieval_source,
    )


@patch("src.rag.service.llm")
@patch("src.rag.service.qa_prompt")
@patch("src.rag.service.process_documents_with_scores")
@patch("src.rag.service.RetrievalStrategyFactory")
@patch("src.rag.service.RouterStrategyFactory")
@patch("src.rag.service.RETRIEVAL_STRATEGY", "auto")
def test_stream_response_uses_router_selected_strategy(
    mock_router_factory,
    mock_retrieval_factory,
    mock_process_documents_with_scores,
    mock_qa_prompt,
    mock_llm,
):
    doc = Document(
        page_content="Transformer architecture uses attention.",
        metadata={
            "source": "attention.pdf",
            "retrieval_strategy": "fusion",
        },
    )

    mock_router_result = Mock()
    mock_router_result.strategy = "fusion"
    mock_router_result.reason = "Conceptual query detected."
    mock_router_result.confidence = 0.9
    mock_router_result.router_type = "rule_based"

    mock_router = Mock()
    mock_router.route.return_value = mock_router_result
    mock_router_factory.get_router.return_value = mock_router

    mock_retrieval_strategy = Mock()
    mock_retrieval_strategy.retrieve.return_value = [doc]
    mock_retrieval_factory.get_strategy.return_value = mock_retrieval_strategy

    mock_process_documents_with_scores.return_value = [
        make_ranked_doc(
            document=doc,
            score=0.95,
            retrieval_source="fusion",
        )
    ]

    mock_messages = ["mock-message"]
    mock_qa_prompt.invoke.return_value = mock_messages

    mock_stream = iter(["answer chunk"])
    mock_llm.stream.return_value = mock_stream

    stream, docs, debug_info = stream_response(
        user_input="Explain transformer architecture",
        chat_history=[],
        selected_document=None,
    )

    assert stream == mock_stream
    assert docs == [doc]

    mock_router_factory.get_router.assert_called_once_with()
    mock_router.route.assert_called_once_with(
        query="Explain transformer architecture",
        selected_document=None,
    )

    mock_retrieval_factory.get_strategy.assert_called_once_with("fusion")
    mock_retrieval_strategy.retrieve.assert_called_once_with(
        query="Explain transformer architecture",
        chat_history=[],
        selected_document=None,
    )

    mock_process_documents_with_scores.assert_called_once_with(
        "Explain transformer architecture",
        [doc],
    )

    mock_qa_prompt.invoke.assert_called_once()
    mock_llm.stream.assert_called_once_with(mock_messages)

    assert debug_info.query == "Explain transformer architecture"
    assert debug_info.requested_strategy == "auto"
    assert debug_info.resolved_strategy == "fusion"
    assert debug_info.router_reason == "Conceptual query detected."
    assert debug_info.router_type == "rule_based"
    assert debug_info.router_confidence == 0.9
    assert debug_info.retrieved_docs_count == 1
    assert debug_info.final_docs_count == 1


@patch("src.rag.service.llm")
@patch("src.rag.service.qa_prompt")
@patch("src.rag.service.process_documents_with_scores")
@patch("src.rag.service.RetrievalStrategyFactory")
@patch("src.rag.service.RouterStrategyFactory")
@patch("src.rag.service.RETRIEVAL_STRATEGY", "hybrid")
def test_stream_response_skips_router_when_strategy_is_not_auto(
    mock_router_factory,
    mock_retrieval_factory,
    mock_process_documents_with_scores,
    mock_qa_prompt,
    mock_llm,
):
    doc = Document(
        page_content="Multi-head attention has queries, keys and values.",
        metadata={
            "source": "attention.pdf",
            "retrieval_strategy": "hybrid",
        },
    )

    mock_retrieval_strategy = Mock()
    mock_retrieval_strategy.retrieve.return_value = [doc]
    mock_retrieval_factory.get_strategy.return_value = mock_retrieval_strategy

    mock_process_documents_with_scores.return_value = [
        make_ranked_doc(
            document=doc,
            score=0.88,
            retrieval_source="hybrid",
        )
    ]

    mock_messages = ["mock-message"]
    mock_qa_prompt.invoke.return_value = mock_messages

    mock_stream = iter(["answer chunk"])
    mock_llm.stream.return_value = mock_stream

    stream, docs, debug_info = stream_response(
        user_input="What is multi-head attention?",
        chat_history=[],
        selected_document="attention.pdf",
    )

    assert stream == mock_stream
    assert docs == [doc]

    mock_router_factory.get_router.assert_not_called()

    mock_retrieval_factory.get_strategy.assert_called_once_with("hybrid")
    mock_retrieval_strategy.retrieve.assert_called_once_with(
        query="What is multi-head attention?",
        chat_history=[],
        selected_document="attention.pdf",
    )

    assert debug_info.query == "What is multi-head attention?"
    assert debug_info.selected_document == "attention.pdf"
    assert debug_info.requested_strategy == "hybrid"
    assert debug_info.resolved_strategy == "hybrid"
    assert debug_info.router_reason is None
    assert debug_info.router_type is None
    assert debug_info.router_confidence is None
    assert debug_info.retrieved_docs_count == 1
    assert debug_info.final_docs_count == 1


@patch("src.rag.service.llm")
@patch("src.rag.service.qa_prompt")
@patch("src.rag.service.process_documents_with_scores")
@patch("src.rag.service.RetrievalStrategyFactory")
@patch("src.rag.service.RouterStrategyFactory")
@patch("src.rag.service.RETRIEVAL_STRATEGY", "auto")
def test_stream_response_passes_selected_document_to_router_and_retriever(
    mock_router_factory,
    mock_retrieval_factory,
    mock_process_documents_with_scores,
    mock_qa_prompt,
    mock_llm,
):
    doc = Document(
        page_content="Document summary content.",
        metadata={
            "source": "attention.pdf",
            "retrieval_strategy": "parent_child",
        },
    )

    mock_router_result = Mock()
    mock_router_result.strategy = "parent_child"
    mock_router_result.reason = "Specific document selected."
    mock_router_result.confidence = 0.8
    mock_router_result.router_type = "rule_based"

    mock_router = Mock()
    mock_router.route.return_value = mock_router_result
    mock_router_factory.get_router.return_value = mock_router

    mock_retrieval_strategy = Mock()
    mock_retrieval_strategy.retrieve.return_value = [doc]
    mock_retrieval_factory.get_strategy.return_value = mock_retrieval_strategy

    mock_process_documents_with_scores.return_value = [
        make_ranked_doc(
            document=doc,
            score=0.91,
            retrieval_source="parent_child",
        )
    ]

    mock_qa_prompt.invoke.return_value = ["mock-message"]
    mock_llm.stream.return_value = iter(["answer chunk"])

    stream_response(
        user_input="Tell me more about this",
        chat_history=[],
        selected_document="attention.pdf",
    )

    mock_router.route.assert_called_once_with(
        query="Tell me more about this",
        selected_document="attention.pdf",
    )

    mock_retrieval_strategy.retrieve.assert_called_once_with(
        query="Tell me more about this",
        chat_history=[],
        selected_document="attention.pdf",
    )