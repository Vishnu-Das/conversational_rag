from dataclasses import dataclass

from langchain_core.documents import (
    Document
)


@dataclass
class RankedDocument:

    document: Document

    score: float

    retrieval_strategy: str | None = None

    retrieval_source: str | None = None