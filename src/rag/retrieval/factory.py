from src.rag.retrieval.base import (
    BaseRetrievalStrategy
)

from src.rag.retrieval.hybrid.strategy import (
    HybridRetrievalStrategy
)

from src.rag.retrieval.parent_child.strategy import (
    ParentChildRetrievalStrategy
)


class RetrievalStrategyFactory:

    HYBRID = "hybrid"

    PARENT_CHILD = "parent_child"

    @staticmethod
    def get_strategy(
        strategy_name: str = HYBRID
    ) -> BaseRetrievalStrategy:

        if (
            strategy_name
            == RetrievalStrategyFactory.HYBRID
        ):

            return HybridRetrievalStrategy()

        if (
            strategy_name
            == RetrievalStrategyFactory.PARENT_CHILD
        ):

            return ParentChildRetrievalStrategy()

        raise ValueError(
            f"Unsupported retrieval strategy: {strategy_name}"
        )