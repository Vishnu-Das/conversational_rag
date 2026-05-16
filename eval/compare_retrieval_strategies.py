import json
import os
import sys
import time

sys.path.append(os.getcwd())

from src.rag.retrieval.factory import (
    RetrievalStrategyFactory
)

from src.rag.pipeline import (
    process_documents
)

from src.rag.service import (
    build_context
)

from pathlib import Path

BASE_DIR = Path(__file__).parent


STRATEGIES = [
    RetrievalStrategyFactory.HYBRID,
    RetrievalStrategyFactory.PARENT_CHILD
]


def normalize_source_name(source: str) -> str:

    if not source:
        return ""

    source = source.replace("\\", "/")

    return source.split("/")[-1]


def load_cases():

    with open(
        BASE_DIR / "strategy_questions.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_strategy(
    case,
    strategy_name
):

    question = case["question"]
    selected_document = case["selected_document"]
    expected_source = case["expected_source"]
    expected_keywords = case.get(
        "expected_keywords",
        []
    )

    strategy = (
        RetrievalStrategyFactory.get_strategy(
            strategy_name
        )
    )

    start_time = time.perf_counter()

    retrieved_docs = strategy.retrieve(
        query=question,
        chat_history=[],
        selected_document=selected_document
    )

    final_docs = process_documents(
        question,
        retrieved_docs
    )

    elapsed_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    retrieved_sources = [
        normalize_source_name(
            doc.metadata.get("source", "")
        )
        for doc in final_docs
    ]

    combined_text = " ".join([
        doc.page_content.lower()
        for doc in final_docs
    ])

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

    source_match = expected_source in retrieved_sources

    context_length = len(
        build_context(final_docs)
    )

    passed = (
        source_match
        and keyword_score >= 0.5
    )

    return {
        "strategy": strategy_name,
        "question": question,
        "selected_document": selected_document,
        "doc_count": len(final_docs),
        "retrieved_sources": retrieved_sources,
        "source_match": source_match,
        "keyword_matches": keyword_matches,
        "keyword_score": keyword_score,
        "context_length": context_length,
        "latency_ms": elapsed_ms,
        "passed": passed
    }


def main():

    cases = load_cases()

    print(
        "\n===== Retrieval Strategy Benchmark =====\n"
    )

    total_results = []

    for case in cases:

        print(
            f"Question: {case['question']}"
        )

        print(
            f"Selected Document: {case['selected_document']}"
        )

        print("-" * 70)

        for strategy_name in STRATEGIES:

            result = evaluate_strategy(
                case,
                strategy_name
            )

            total_results.append(result)

            status = (
                "✅ PASS"
                if result["passed"]
                else "❌ FAIL"
            )

            print(status)
            print("Strategy:", result["strategy"])
            print("Doc Count:", result["doc_count"])
            print("Sources:", result["retrieved_sources"])
            print("Keyword Matches:", result["keyword_matches"])
            print("Keyword Score:", result["keyword_score"])
            print("Context Length:", result["context_length"])
            print("Latency:", result["latency_ms"], "ms")
            print()

        print("=" * 70)

    passed_count = sum(
        1 for result in total_results
        if result["passed"]
    )

    print(
        f"\nFinal Score: {passed_count}/{len(total_results)} passed"
    )


if __name__ == "__main__":
    main()