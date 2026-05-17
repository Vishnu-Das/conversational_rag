from unittest.mock import Mock, patch

import pytest

from src.rag.retrieval.factory import RetrievalStrategyFactory


@patch("src.rag.retrieval.factory.HybridRetrievalStrategy")
def test_get_hybrid_strategy(mock_hybrid_strategy):
    mock_instance = Mock()
    mock_hybrid_strategy.return_value = mock_instance

    strategy = RetrievalStrategyFactory.get_strategy(
        RetrievalStrategyFactory.HYBRID
    )

    assert strategy == mock_instance
    mock_hybrid_strategy.assert_called_once_with()


@patch("src.rag.retrieval.factory.ParentChildRetrievalStrategy")
def test_get_parent_child_strategy(mock_parent_child_strategy):
    mock_instance = Mock()
    mock_parent_child_strategy.return_value = mock_instance

    strategy = RetrievalStrategyFactory.get_strategy(
        RetrievalStrategyFactory.PARENT_CHILD
    )

    assert strategy == mock_instance
    mock_parent_child_strategy.assert_called_once_with()


@patch("src.rag.retrieval.factory.FusionRetrievalStrategy")
def test_get_fusion_strategy(mock_fusion_strategy):
    mock_instance = Mock()
    mock_fusion_strategy.return_value = mock_instance

    strategy = RetrievalStrategyFactory.get_strategy(
        RetrievalStrategyFactory.FUSION
    )

    assert strategy == mock_instance
    mock_fusion_strategy.assert_called_once_with()


@patch("src.rag.retrieval.factory.HybridRetrievalStrategy")
def test_get_strategy_defaults_to_hybrid(mock_hybrid_strategy):
    mock_instance = Mock()
    mock_hybrid_strategy.return_value = mock_instance

    strategy = RetrievalStrategyFactory.get_strategy()

    assert strategy == mock_instance
    mock_hybrid_strategy.assert_called_once_with()


def test_get_unsupported_strategy_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported retrieval strategy"):
        RetrievalStrategyFactory.get_strategy("unsupported_strategy")