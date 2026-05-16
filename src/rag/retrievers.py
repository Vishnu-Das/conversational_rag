import streamlit as st

from functools import lru_cache

from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever
)

from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from src.vectorstore import (
    load_vectorstore,
    load_documents_from_vectorstore
)

from src.rag.llm import llm

from src.rag.prompts import (
    contextualize_q_prompt
)

from src.config import (
    INITIAL_RETRIEVAL_K
)


def normalize_source_name(source: str) -> str:

    if not source:
        return ""

    source = source.replace("\\", "/")

    return source.split("/")[-1]


@st.cache_resource
def get_all_documents():

    return load_documents_from_vectorstore()


all_documents = get_all_documents()

vectorstore = load_vectorstore()


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

        filtered_documents = [
            doc for doc in all_documents
            if normalize_source_name(
                doc.metadata.get("source", "")
            ) == selected_document
        ]

        matching_sources = list({
            doc.metadata.get("source")
            for doc in filtered_documents
            if doc.metadata.get("source")
        })

        if matching_sources:

            search_kwargs["filter"] = {
                "source": {
                    "$in": matching_sources
                }
            }

    vector_retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )

    if not filtered_documents:

        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(
        filtered_documents
    )

    bm25_retriever.k = INITIAL_RETRIEVAL_K

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


@lru_cache(maxsize=10)
def get_history_aware_retriever(selected_document: str):

    retriever = get_retriever(
        selected_document
    )

    history_aware_retriever = (
        create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )
    )

    return history_aware_retriever