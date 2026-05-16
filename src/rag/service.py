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


DOCUMENT_LEVEL_KEYWORDS = [
    "summarize",
    "summary",
    "summarise",
    "overview",
    "key concepts",
    "important topics",
    "main ideas",
    "study notes",
    "explain this document",
    "explain the document",
]


def normalize_source_name(source: str) -> str:

    if not source:
        return ""

    source = source.replace("\\", "/")

    return source.split("/")[-1]


def is_document_level_request(
    user_input: str
) -> bool:

    user_input = user_input.lower()

    return any(
        keyword in user_input
        for keyword in DOCUMENT_LEVEL_KEYWORDS
    )


@traceable(
    name="Get Selected Document Chunks",
    run_type="retriever"
)
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

    is_doc_level = is_document_level_request(
        user_input
    )

    if (
        is_doc_level
        and selected_document
        and selected_document != "All Documents"
    ):

        docs = get_selected_document_chunks(
            selected_document
        )

    else:

        retrieval_strategy = (
            RetrievalStrategyFactory.get_strategy(
                RETRIEVAL_STRATEGY
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