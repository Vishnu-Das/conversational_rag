import json
import os
import sys

sys.path.append(os.getcwd())

from src.rag.service import stream_response


def load_questions():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stream_text(stream):
    answer = ""

    for chunk in stream:
        if chunk.content:
            answer += chunk.content

    return answer


def evaluate_answer(case):
    question = case["question"]
    selected_document = case.get("selected_document")
    expected_strategy = case.get("expected_strategy")
    expected_keywords = case.get("expected_keywords", [])

    stream, sources, debug_info = stream_response(
        user_input=question,
        chat_history=[],
        selected_document=selected_document,
    )

    answer = collect_stream_text(stream)
    answer_lower = answer.lower()

    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    ]

    keyword_score = (
        len(matched_keywords) / len(expected_keywords)
        if expected_keywords
        else 1.0
    )

    strategy_match = (
        debug_info.resolved_strategy == expected_strategy
        if expected_strategy
        else True
    )

    passed = keyword_score >= 0.5 and strategy_match

    return {
        "question": question,
        "selected_document": selected_document,
        "expected_strategy": expected_strategy,
        "resolved_strategy": debug_info.resolved_strategy,
        "router_type": debug_info.router_type,
        "router_reason": debug_info.router_reason,
        "router_confidence": debug_info.router_confidence,
        "answer": answer,
        "matched_keywords": matched_keywords,
        "keyword_score": keyword_score,
        "strategy_match": strategy_match,
        "source_count": len(sources),
        "retrieval_latency_ms": debug_info.retrieval_latency_ms,
        "rerank_latency_ms": debug_info.rerank_latency_ms,
        "total_latency_ms": debug_info.total_latency_ms,
        "passed": passed,
    }


def main():
    cases = load_questions()
    results = [evaluate_answer(case) for case in cases]

    passed_count = sum(1 for result in results if result["passed"])

    print("\n===== RAG Answer + Routing Evaluation =====\n")

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
        print("Source Count:", result["source_count"])
        print("Matched Keywords:", result["matched_keywords"])
        print("Keyword Score:", result["keyword_score"])
        print("Retrieval Latency ms:", result["retrieval_latency_ms"])
        print("Rerank Latency ms:", result["rerank_latency_ms"])
        print("Total Latency ms:", result["total_latency_ms"])
        print("Answer Preview:", result["answer"][:500])
        print("-" * 80)

    print(f"\nFinal Score: {passed_count}/{len(results)} passed")


if __name__ == "__main__":
    main()