from langchain_openai import ChatOpenAI

from src.config import (
    ROUTER_MODEL_NAME
)

from src.rag.models import (
    RouterResult
)

from src.rag.prompts import (
    router_prompt
)

from src.rag.retrieval.factory import (
    RetrievalStrategyFactory
)

from src.rag.routing.base import (
    BaseRouterStrategy
)

from src.rag.routing.rule_based import (
    RuleBasedRouterStrategy
)

from src.rag.routing.schemas import (
    LLMRouterOutput
)


class LLMRouterStrategy(BaseRouterStrategy):

    VALID_STRATEGIES = {
        RetrievalStrategyFactory.HYBRID,
        RetrievalStrategyFactory.PARENT_CHILD,
        RetrievalStrategyFactory.FUSION
    }

    def __init__(self):

        self.llm = ChatOpenAI(
            model=ROUTER_MODEL_NAME,
            temperature=0
        )

        self.structured_llm = (
            self.llm.with_structured_output(
                LLMRouterOutput
            )
        )

        self.fallback_router = (
            RuleBasedRouterStrategy()
        )

    def route(
        self,
        query: str,
        selected_document: str | None = None
    ) -> RouterResult:

        try:

            chain = (
                router_prompt
                | self.structured_llm
            )

            result = chain.invoke({
                "query": query,
                "selected_document": selected_document
            })

            if result.strategy not in self.VALID_STRATEGIES:
                raise ValueError(
                    f"Invalid strategy: {result.strategy}"
                )

            return RouterResult(
                strategy=result.strategy,
                reason=result.reason,
                confidence=result.confidence,
                router_type="llm"
            )

        except Exception as error:

            fallback_result = (
                self.fallback_router.route(
                    query=query,
                    selected_document=selected_document
                )
            )

            fallback_result.reason = (
                "LLM router failed. "
                f"Fallback used. Error: {str(error)}. "
                f"Fallback reason: {fallback_result.reason}"
            )

            fallback_result.router_type = "llm_fallback"

            return fallback_result