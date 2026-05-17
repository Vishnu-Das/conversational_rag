import json
import os
import sys

sys.path.append(os.getcwd())

from langchain_openai import ChatOpenAI

from src.config import MAIN_MODEL_NAME
from src.rag.service import stream_response


judge_llm = ChatOpenAI(
    model=MAIN_MODEL_NAME,
    temperature=0,
)


def load_questions():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stream_text(stream):
    answer = ""

    for chunk in stream:
        if chunk.content:
            answer += chunk.content

    return answer


def build_context(sources):
    return "\n\n".join(
        doc.page_content
        for doc in sources
    )


def judge_answer(question, answer, context):
    prompt = f"""
You are an expert evaluator for Retrieval-Augmented Generation systems.

Evaluate whether the generated answer:
1. Correctly answers the question.
2. Is grounded in the retrieved context.
3. Avoids hallucinations.
4. Is reasonably complete.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Return STRICT JSON only in this format:

{{
  "score": 0,
  "grounded": true,
  "hallucination": false,
  "complete": true,
  "reason": "short explanation"
}}
"""

    response = judge_llm.invoke(prompt)
    return response.content


def evaluate_case(case):
    question = case["question"]
    selected_document = case.get("selected_document")
    expected_strategy = case.get("expected_strategy")

    stream, sources, debug_info = stream_response(
        user_input=question,
        chat_history=[],
        selected_document=selected_document,
    )

    answer = collect_stream_text(stream)
    context = build_context(sources)

    evaluation = judge_answer(
        question=question,
        answer=answer,
        context=context,
    )

    strategy_match = (
        debug_info.resolved_strategy == expected_strategy
        if expected_strategy
        else True
    )

    return {
        "question": question,
        "selected_document": selected_document,
        "expected_strategy": expected_strategy,
        "resolved_strategy": debug_info.resolved_strategy,
        "strategy_match": strategy_match,
        "router_type": debug_info.router_type,
        "router_reason": debug_info.router_reason,
        "router_confidence": debug_info.router_confidence,
        "source_count": len(sources),
        "answer": answer,
        "evaluation": evaluation,
        "retrieval_latency_ms": debug_info.retrieval_latency_ms,
        "rerank_latency_ms": debug_info.rerank_latency_ms,
        "total_latency_ms": debug_info.total_latency_ms,
    }


def main():
    cases = load_questions()

    print("\n===== LLM Judge + Routing Evaluation =====\n")

    for case in cases:
        result = evaluate_case(case)

        print("Question:")
        print(result["question"])

        print("\nSelected Document:")
        print(result["selected_document"])

        print("\nExpected Strategy:")
        print(result["expected_strategy"])

        print("\nResolved Strategy:")
        print(result["resolved_strategy"])

        print("\nStrategy Match:")
        print(result["strategy_match"])

        print("\nRouter Type:")
        print(result["router_type"])

        print("\nRouter Confidence:")
        print(result["router_confidence"])

        print("\nRouter Reason:")
        print(result["router_reason"])

        print("\nSource Count:")
        print(result["source_count"])

        print("\nLatency:")
        print("Retrieval:", result["retrieval_latency_ms"], "ms")
        print("Rerank:", result["rerank_latency_ms"], "ms")
        print("Total:", result["total_latency_ms"], "ms")

        print("\nAnswer:")
        print(result["answer"][:700])

        print("\nJudge Evaluation:")
        print(result["evaluation"])

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()