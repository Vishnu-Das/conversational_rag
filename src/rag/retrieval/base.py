from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document


class BaseRetrievalStrategy(ABC):
    """
    Base interface for all retrieval strategies.

    Any retrieval strategy should implement this contract.
    Example:
    - Hybrid retrieval
    - Parent-child retrieval
    - Graph retrieval
    - Compression retrieval
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        chat_history: list,
        selected_document: Optional[str] = None
    ) -> List[Document]:
        pass