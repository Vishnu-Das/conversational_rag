import json
import os
import sys

sys.path.append(os.getcwd())

from src.rag.retrievers import get_history_aware_retriever
from src.helpers.deduplication import deduplicate_docs
from src.reranker import rerank_documents
from src.config import RERANK_TOP_K


def normalize_source_name(source: str) -> str:
    if not source:
        return ""

    source = source.replace("\\", "/")
    return source.split("/")[-1]


def load_questions():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_case(case):
    question = case["question"]
    selected_document = case["selected_document"]
    expected_source = case["expected_source"]
    expected_keywords = case.get("expected_keywords", [])

    retriever = get_history_aware_retriever(
        selected_document
    )

    retrieved_docs = retriever.invoke({
        "input": question,
        "chat_history": []
    })

    dedup_docs = deduplicate_docs(
        retrieved_docs
    )

    reranked_docs = rerank_documents(
        question,
        dedup_docs,
        top_k=RERANK_TOP_K
    )

    retrieved_sources = [
        normalize_source_name(
            doc.metadata.get("source", "")
        )
        for doc in reranked_docs
    ]

    combined_text = " ".join([
        doc.page_content.lower()
        for doc in reranked_docs
    ])

    source_match = expected_source in retrieved_sources

    keyword_matches = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in combined_text
    ]

    keyword_score = (
        len(keyword_matches) / len(expected_keywords)
        if expected_keywords
        else 0
    )

    passed = source_match and keyword_score >= 0.5

    return {
        "question": question,
        "selected_document": selected_document,
        "retrieved_sources": retrieved_sources,
        "source_match": source_match,
        "keyword_matches": keyword_matches,
        "keyword_score": keyword_score,
        "passed": passed
    }


def main():
    cases = load_questions()

    results = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)

    passed_count = sum(
        1 for result in results
        if result["passed"]
    )

    print("\n===== RAG Retrieval Evaluation =====\n")

    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"

        print(status)
        print("Question:", result["question"])
        print("Selected Document:", result["selected_document"])
        print("Retrieved Sources:", result["retrieved_sources"])
        print("Keyword Matches:", result["keyword_matches"])
        print("Keyword Score:", result["keyword_score"])
        print("-" * 50)

    print(
        f"\nFinal Score: {passed_count}/{len(results)} passed"
    )


if __name__ == "__main__":
    main()