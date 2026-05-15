from typing import List

from langchain_core.messages import (
    BaseMessage
)

from src.rag.prompts import (
    qa_prompt
)

from src.rag.llm import llm

from src.rag.retrievers import (
    get_history_aware_retriever
)

from src.rag.pipeline import (
    process_documents
)

from src.rag.cache import (
    cached_retrieval
)


def stream_response(
    user_input: str,
    chat_history: List[BaseMessage],
    selected_document: str = None
):

    history_aware_retriever = (
        get_history_aware_retriever(
            selected_document
        )
    )

    if not chat_history:

        retrieved_docs = cached_retrieval(
            user_input,
            selected_document or "All Documents"
        )

    else:

        retrieved_docs = (
            history_aware_retriever.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
        )

    docs = process_documents(
        user_input,
        retrieved_docs
    )

    context = "\n\n---\n\n".join([
        doc.page_content
        for doc in docs
    ])

    messages = qa_prompt.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
    })

    stream = llm.stream(messages)

    return stream, docs