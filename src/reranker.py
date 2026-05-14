from sentence_transformers import CrossEncoder
from src.config import RERANKING_MODEL_NAME
from huggingface_hub import login
import streamlit as st
# from src.config import HF_TOKEN


# if HF_TOKEN:
#     login(token=HF_TOKEN)

@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKING_MODEL_NAME)


reranker_model = load_reranker()


## Reranking function that takes a query and a list of documents, computes relevance scores using the CrossEncoder model, and returns the top-k most relevant documents.
def rerank_documents(
    query,
    documents,
    top_k=4
):
    if not documents:
        return []
    pairs = [

        (
            query,
            doc.page_content
        )

        for doc in documents
    ]
    scores = reranker_model.predict(
        pairs
    )
    scored_docs = list(
        zip(documents, scores)
    )
    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )
    reranked_docs = [
        doc for doc, score in scored_docs[:top_k]
    ]
    return reranked_docs