import pytest

from src.rag.routing.factory import RouterStrategyFactory
from src.rag.routing.rule_based import RuleBasedRouterStrategy


def test_get_rule_based_router():
    router = RouterStrategyFactory.get_router(
        RouterStrategyFactory.RULE_BASED
    )

    assert isinstance(router, RuleBasedRouterStrategy)


def test_get_unsupported_router_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported router type"):
        RouterStrategyFactory.get_router("unsupported_router")