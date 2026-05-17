from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RetrievalDebugInfo:

    query: str
    selected_document: str | None = None
    requested_strategy: str | None = None
    resolved_strategy: str | None = None
    router_reason: str | None = None
    router_type: str | None = None
    router_confidence: float | None = None
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    retrieved_docs_count: int = 0
    final_docs_count: int = 0

    retrieved_docs: List[Dict[str, Any]] = field(
        default_factory=list
    )

    final_docs: List[Dict[str, Any]] = field(
        default_factory=list
    )


def normalize_source_name(
    source: str
) -> str:

    if not source:
        return "Unknown"

    source = source.replace("\\", "/")

    return source.split("/")[-1]


def build_doc_debug_entry(
    doc,
    score=None,
    retrieval_source=None
):

    metadata = doc.metadata or {}

    return {
        "source": normalize_source_name(
            metadata.get("source", "")
        ),
        "page": (
            metadata.get("page")
            or metadata.get("page_label")
            or "N/A"
        ),
        "preview": (
            doc.page_content[:300]
            .replace("\n", " ")
            .strip()
        ),
        "score": score,
        "retrieval_source": retrieval_source
    }