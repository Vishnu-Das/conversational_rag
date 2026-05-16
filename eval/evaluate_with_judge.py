import json
import os
import sys

sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.rag import stream_response
from src.config import MODEL_NAME


judge_llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0
)


def load_questions():

    with open(
        "eval/questions.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def collect_stream_text(stream):

    answer = ""

    for chunk in stream:

        if chunk.content:
            answer += chunk.content

    return answer


def build_context(sources):

    return "\n\n".join([
        doc.page_content
        for doc in sources
    ])


def judge_answer(
    question,
    answer,
    context
):

    prompt = f"""
You are an expert evaluator for RAG systems.

Your task is to evaluate whether the generated answer:
1. Correctly answers the question
2. Is grounded in the provided context
3. Avoids hallucinations
4. Is reasonably complete

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the answer.

Return STRICT JSON only:

{{
    "score": number between 0 and 10,
    "grounded": true/false,
    "hallucination": true/false,
    "reason": "short explanation"
}}
"""

    response = judge_llm.invoke(prompt)

    return response.content


def evaluate_case(case):

    question = case["question"]

    selected_document = (
        case["selected_document"]
    )

    stream, sources = stream_response(
        user_input=question,
        chat_history=[],
        selected_document=selected_document
    )

    answer = collect_stream_text(
        stream
    )

    context = build_context(
        sources
    )

    evaluation = judge_answer(
        question,
        answer,
        context
    )

    return {
        "question": question,
        "selected_document": selected_document,
        "answer": answer,
        "evaluation": evaluation
    }


def main():

    cases = load_questions()

    print(
        "\n===== LLM Judge Evaluation =====\n"
    )

    for case in cases:

        result = evaluate_case(case)

        print("Question:")
        print(result["question"])

        print("\nAnswer:")
        print(result["answer"][:700])

        print("\nJudge Evaluation:")
        print(result["evaluation"])

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()