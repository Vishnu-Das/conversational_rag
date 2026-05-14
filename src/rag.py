## This module implements the core Retrieval-Augmented Generation (RAG) logic for the application. It includes functions to retrieve relevant documents based on user queries and chat history, and to generate responses using a language model. The module also handles document filtering, deduplication, and reranking to ensure that the most relevant information is used in generating answers.

from turtle import st
from typing import List
import os
from functools import lru_cache
import streamlit as st

from langchain_core.messages import (
    BaseMessage
)

from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_community.retrievers import (
    BM25Retriever,
)

from langchain_classic.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever
)


from langchain_openai import ChatOpenAI

from src.vectorstore import (
    load_vectorstore,
    load_and_split_documents,
    load_documents_from_vectorstore
)

from src.config import (
    MODEL_NAME,
    RERANK_TOP_K,
    INITIAL_RETRIEVAL_K
)

from src.reranker import (
    rerank_documents
)

from src.helpers.deduplication import deduplicate_docs

@st.cache_resource
def get_all_documents(): ## This function loads all documents from the vectorstore and caches the result to avoid redundant loading on every query. It is used to enable filtering and retrieval based on document source.
    return load_documents_from_vectorstore()

## Load all documents from the vectorstore to enable filtering and retrieval based on document source. This is done at startup to avoid repeated loading on every query.
all_documents = get_all_documents()

## Prompt to reformulate the user's question in the context of the chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Given a chat history and the latest user question,
        formulate a standalone question that can be understood
        without the chat history.

        Do NOT answer the question.
        Just reformulate it if needed.
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{input}")
])

## Load the vectorstore at startup to avoid doing it on every query
vectorstore = load_vectorstore()

## Initialize the LLM with streaming enabled for real-time response generation
llm = ChatOpenAI(
    model=MODEL_NAME,  streaming=True
)

## Function to create a retriever based on the selected document filter.
@lru_cache(maxsize=10)
def get_retriever(selected_document=None):

    search_kwargs = {
        "k": INITIAL_RETRIEVAL_K
    }
    filtered_documents = all_documents
    if (
        selected_document
        and selected_document != "All Documents"
    ):
        search_kwargs["filter"] = {
            "source": selected_document
        }

        filtered_documents = [

            doc for doc in all_documents

            if os.path.basename(
                doc.metadata.get("source", "")
            ) == selected_document
        ]

    vector_retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )
    if not filtered_documents:
        return vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )

    bm25_retriever = BM25Retriever.from_documents(
        filtered_documents
    )
    bm25_retriever.k = 4
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever
        ],
        weights=[0.7, 0.3]
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )
    return multi_query_retriever

## Prompt template for the LLM to generate answers based on retrieved context and chat history.
prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the user's questions using ONLY the
        provided context and conversation history.

        Rules:
        - If the answer is not present in the context,
          clearly say you don't know.
        - Do NOT hallucinate or make up information.
        - Keep answers concise but informative.
        - Prefer factual answers grounded in the context.
        - Use conversation history when relevant.
        - If the user asks follow-up questions,
          understand them in context.

        Retrieved Context:
        {context}
        """
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    ("human", "{input}")

])

## Cache the history-aware retriever to avoid redundant computations when the same document filter is selected multiple times.
@lru_cache(maxsize=10)
def get_history_aware_retriever(selected_document: str):
        retriever = get_retriever(selected_document)

        # print("\n========== DEBUG ==========")
        # print("Selected Document:", selected_document)

        ## Create a history-aware retriever that reformulates the question

        history_aware_retriever = (
            create_history_aware_retriever(
                llm,
                retriever,
                contextualize_q_prompt
            )
        )

        return history_aware_retriever

@lru_cache(maxsize=100)
def cached_retrieval(user_input: str,selected_document: str):
    history_aware_retriever = (
        get_history_aware_retriever(
            selected_document
        )
    )
    docs = history_aware_retriever.invoke({
        "input": user_input,
        "chat_history": []
    })

    return docs

def reset_rag_caches():
    get_retriever.cache_clear()
    get_history_aware_retriever.cache_clear()
    cached_retrieval.cache_clear()
    get_all_documents.clear()

## Main function to handle user input, retrieve relevant documents,
## and generate a streamed response from the LLM
def stream_response(
    user_input: str,
    chat_history: List[BaseMessage],
    selected_document: str = None
):

    history_aware_retriever = get_history_aware_retriever(selected_document)

    if not chat_history:
        history_aware_docs = cached_retrieval(
            user_input,
            selected_document or "All Documents"
        )
    else:
        history_aware_docs = history_aware_retriever.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })

    ## Deduplicate retrieved documents to avoid redundancy in the context
    dedup_docs = deduplicate_docs(history_aware_docs)

    ## Rerank retrieved documents based on relevance to the question and chat history

    docs = rerank_documents(
        user_input,
        dedup_docs,
        top_k=RERANK_TOP_K
    )

    # print("\nRetrieved Documents:")
    # for doc in docs:
    #     print(
    #         doc.metadata.get("source")
    #     )
    # print("===========================\n")

    context = "\n\n---\n\n".join([
        doc.page_content for doc in docs
    ])

    messages = prompt.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
    })

    stream = llm.stream(messages)

    return stream, docs