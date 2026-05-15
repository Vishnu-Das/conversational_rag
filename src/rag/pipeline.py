from src.helpers.deduplication import (
    deduplicate_docs
)

from src.reranker import (
    rerank_documents
)

from src.config import (
    RERANK_TOP_K
)

# from src.compression import compress_documents


def process_documents(
    user_input,
    documents
):

    dedup_docs = deduplicate_docs(
        documents
    )

    reranked_docs = rerank_documents(
        user_input,
        dedup_docs,
        top_k=RERANK_TOP_K
    )

    # compressed_docs = compress_documents(
    #     user_input,
    #     reranked_docs
    # )

    return reranked_docs