from typing import List

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

from src.rag.pipeline import process_documents

from src.config import RETRIEVAL_STRATEGY

from src.rag.retrieval.router import (
    route_retrieval_strategy
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
    strategy_name = RETRIEVAL_STRATEGY

    if RETRIEVAL_STRATEGY == "auto":
        strategy_name = route_retrieval_strategy(
            query=user_input,
            selected_document=selected_document
        )
    print(f"Using retrieval strategy: {strategy_name}")

    retrieval_strategy = (
        RetrievalStrategyFactory.get_strategy(
            strategy_name
        )
    )

    retrieved_docs = retrieval_strategy.retrieve(
        query=user_input,
        chat_history=chat_history,
        selected_document=selected_document
    )

    docs = process_documents(
        user_input,
        retrieved_docs
    )

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

    return stream, docs