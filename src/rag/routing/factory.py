from src.config import (
    ROUTER_TYPE
)

from src.rag.routing.base import (
    BaseRouterStrategy
)

from src.rag.routing.rule_based import (
    RuleBasedRouterStrategy
)

from src.rag.routing.llm import (
    LLMRouterStrategy
)


class RouterStrategyFactory:

    RULE_BASED = "rule_based"

    LLM = "llm"

    @staticmethod
    def get_router(
        router_type: str = ROUTER_TYPE
    ) -> BaseRouterStrategy:

        if router_type == RouterStrategyFactory.RULE_BASED:
            return RuleBasedRouterStrategy()

        if router_type == RouterStrategyFactory.LLM:
            return LLMRouterStrategy()

        raise ValueError(
            f"Unsupported router type: {router_type}"
        )