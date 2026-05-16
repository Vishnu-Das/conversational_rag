import json
import os
import sys

sys.path.append(os.getcwd())

from src.rag.service import stream_response
from langchain_core.messages import HumanMessage


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
    selected_document = case["selected_document"]
    expected_keywords = case.get("expected_keywords", [])

    stream, sources = stream_response(
        user_input=question,
        chat_history=[],
        selected_document=selected_document
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
        else 0
    )

    passed = keyword_score >= 0.5

    return {
        "question": question,
        "selected_document": selected_document,
        "answer": answer,
        "matched_keywords": matched_keywords,
        "keyword_score": keyword_score,
        "passed": passed,
        "source_count": len(sources)
    }


def main():
    cases = load_questions()

    results = [
        evaluate_answer(case)
        for case in cases
    ]

    passed_count = sum(
        1 for result in results
        if result["passed"]
    )

    print("\n===== RAG Answer Evaluation =====\n")

    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"

        print(status)
        print("Question:", result["question"])
        print("Selected Document:", result["selected_document"])
        print("Source Count:", result["source_count"])
        print("Matched Keywords:", result["matched_keywords"])
        print("Keyword Score:", result["keyword_score"])
        print("Answer:", result["answer"][:500])
        print("-" * 70)

    print(
        f"\nFinal Score: {passed_count}/{len(results)} passed"
    )


if __name__ == "__main__":
    main()