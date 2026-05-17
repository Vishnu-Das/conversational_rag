from typing import List
import time

from src.rag.debug import (
    RetrievalDebugInfo,
    build_doc_debug_entry
)

from langchain_core.messages import BaseMessage
from langsmith import traceable

from src.rag.prompts import qa_prompt
from src.rag.llm import llm

from src.rag.retrievers import (
    get_all_documents
)

from src.rag.retrieval.factory import (
    RetrievalStrategyFactory
)

from src.config import RETRIEVAL_STRATEGY

from src.rag.retrieval.router import (
    route_retrieval_strategy
)

from src.rag.pipeline import (
    # process_documents,
    process_documents_with_scores
)


def normalize_source_name(source: str) -> str:

    if not source:
        return ""

    source = source.replace("\\", "/")

    return source.split("/")[-1]


@traceable( name="Get Selected Document Chunks", run_type="retriever" )
def get_selected_document_chunks(
    selected_document: str,
    max_chunks: int = 12
):

    all_documents = get_all_documents()

    docs = [
        doc
        for doc in all_documents
        if normalize_source_name(
            doc.metadata.get("source", "")
        ) == selected_document
    ]

    return docs[:max_chunks]


def build_context(docs):

    return "\n\n---\n\n".join([
        doc.page_content
        for doc in docs
    ])


@traceable(
    name="RAG Stream Response",
    run_type="chain"
)
def stream_response(
    user_input: str,
    chat_history: List[BaseMessage],
    selected_document: str = None
):
    debug_info = RetrievalDebugInfo(
        query=user_input,
        selected_document=selected_document,
        requested_strategy=RETRIEVAL_STRATEGY
    )
    total_start = time.perf_counter()

    strategy_name = RETRIEVAL_STRATEGY

    if RETRIEVAL_STRATEGY == "auto":
        strategy_name = route_retrieval_strategy(
            query=user_input,
            selected_document=selected_document
        )
    print(f"Using retrieval strategy: {strategy_name}")

    debug_info.resolved_strategy = strategy_name

    retrieval_strategy = (
        RetrievalStrategyFactory.get_strategy(
            strategy_name
        )
    )

    retrieval_start = time.perf_counter()

    retrieved_docs = retrieval_strategy.retrieve(
        query=user_input,
        chat_history=chat_history,
        selected_document=selected_document
    )

    debug_info.retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000,2)

    debug_info.retrieved_docs_count = len(
        retrieved_docs
    )

    debug_info.retrieved_docs = [
        build_doc_debug_entry(
            doc,
            retrieval_source=doc.metadata.get(
                "retrieval_strategy"
            )
        )
        for doc in retrieved_docs
    ]

    rerank_start = time.perf_counter()

    ranked_documents = (
        process_documents_with_scores(
            user_input,
            retrieved_docs
        )
    )

    docs = [
        ranked_doc.document
        for ranked_doc in ranked_documents
    ]
    
    debug_info.rerank_latency_ms = round((time.perf_counter() - rerank_start) * 1000,2)

    debug_info.final_docs_count = len(docs)

    debug_info.final_docs = [
        build_doc_debug_entry(
            ranked_doc.document,
            score=round(
                ranked_doc.score,
                4
            ),
            retrieval_source=ranked_doc.retrieval_source
        )
        for ranked_doc in ranked_documents
    ]

    # print("\n========== FINAL DOCS ==========")
    # for doc in docs:
    #     print(doc.metadata)
    # print("================================\n")

    context = build_context(
        docs
    )

    messages = qa_prompt.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
    })

    stream = llm.stream(messages)

    debug_info.total_latency_ms = round((time.perf_counter() - total_start) * 1000,2)

    return stream, docs, debug_info