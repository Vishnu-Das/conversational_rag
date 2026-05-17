import json
import os
import sys

sys.path.append(os.getcwd())

from src.rag.service import stream_response


def normalize_source_name(source: str) -> str:
    if not source:
        return ""

    source = source.replace("\\", "/")
    return source.split("/")[-1]


def load_questions():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def drain_stream(stream):
    """
    Consume the stream so the full RAG pipeline runs.
    Retrieval/debug info is already returned separately.
    """
    for _ in stream:
        pass


def evaluate_case(case):
    question = case["question"]
    selected_document = case.get("selected_document")
    expected_source = case.get("expected_source")
    expected_strategy = case.get("expected_strategy")
    expected_keywords = case.get("expected_keywords", [])

    stream, sources, debug_info = stream_response(
        user_input=question,
        chat_history=[],
        selected_document=selected_document,
    )

    drain_stream(stream)

    retrieved_sources = [
        normalize_source_name(doc.metadata.get("source", ""))
        for doc in sources
    ]

    combined_text = " ".join(
        doc.page_content.lower()
        for doc in sources
    )

    source_match = (
        expected_source in retrieved_sources
        if expected_source
        else True
    )

    keyword_matches = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in combined_text
    ]

    keyword_score = (
        len(keyword_matches) / len(expected_keywords)
        if expected_keywords
        else 1.0
    )

    strategy_match = (
        debug_info.resolved_strategy == expected_strategy
        if expected_strategy
        else True
    )

    passed = (
        source_match
        and keyword_score >= 0.5
        and strategy_match
    )

    return {
        "question": question,
        "selected_document": selected_document,
        "expected_strategy": expected_strategy,
        "resolved_strategy": debug_info.resolved_strategy,
        "router_type": debug_info.router_type,
        "router_reason": debug_info.router_reason,
        "router_confidence": debug_info.router_confidence,
        "retrieved_sources": retrieved_sources,
        "source_match": source_match,
        "keyword_matches": keyword_matches,
        "keyword_score": keyword_score,
        "strategy_match": strategy_match,
        "retrieval_latency_ms": debug_info.retrieval_latency_ms,
        "rerank_latency_ms": debug_info.rerank_latency_ms,
        "total_latency_ms": debug_info.total_latency_ms,
        "final_docs_count": debug_info.final_docs_count,
        "passed": passed,
    }


def main():
    cases = load_questions()
    results = [evaluate_case(case) for case in cases]

    passed_count = sum(1 for result in results if result["passed"])

    print("\n===== RAG Retrieval + Routing Evaluation =====\n")

    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"

        print(status)
        print("Question:", result["question"])
        print("Selected Document:", result["selected_document"])
        print("Expected Strategy:", result["expected_strategy"])
        print("Resolved Strategy:", result["resolved_strategy"])
        print("Strategy Match:", result["strategy_match"])
        print("Router Type:", result["router_type"])
        print("Router Confidence:", result["router_confidence"])
        print("Router Reason:", result["router_reason"])
        print("Retrieved Sources:", result["retrieved_sources"])
        print("Source Match:", result["source_match"])
        print("Keyword Matches:", result["keyword_matches"])
        print("Keyword Score:", result["keyword_score"])
        print("Final Docs Count:", result["final_docs_count"])
        print("Retrieval Latency ms:", result["retrieval_latency_ms"])
        print("Rerank Latency ms:", result["rerank_latency_ms"])
        print("Total Latency ms:", result["total_latency_ms"])
        print("-" * 80)

    print(f"\nFinal Score: {passed_count}/{len(results)} passed")


if __name__ == "__main__":
    main()