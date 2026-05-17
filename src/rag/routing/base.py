from abc import ABC, abstractmethod

from src.rag.models import RouterResult


class BaseRouterStrategy(ABC):

    @abstractmethod
    def route(
        self,
        query: str,
        selected_document: str | None = None
    ) -> RouterResult:
        pass