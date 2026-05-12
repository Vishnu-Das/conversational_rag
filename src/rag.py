from typing import List

from langchain_core.messages import (
    BaseMessage,
    SystemMessage
)

from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_openai import ChatOpenAI

from src.vectorstore import load_vectorstore

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Given a chat history and the latest user question,
        formulate a standalone question that can be understood
        without the chat history.

        Do NOT answer the question.
        Just reformulate it if needed.
        """
    ),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")
])


vectorstore = load_vectorstore()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",  streaming=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the user's questions using ONLY the
        provided context and conversation history.

        Rules:
        - If the answer is not present in the context,
          clearly say you don't know.
        - Do NOT hallucinate or make up information.
        - Keep answers concise but informative.
        - Prefer factual answers grounded in the context.
        - Use conversation history when relevant.
        - If the user asks follow-up questions,
          understand them in context.

        Retrieved Context:
        {context}
        """
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    ("human", "{input}")

])

def stream_response(
    user_input: str,
    chat_history: List[BaseMessage]
):
    docs = history_aware_retriever.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
    context =  "\n\n---\n\n".join([
        doc.page_content for doc in docs
    ])
    messages = prompt.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
    })
    stream = llm.stream(messages)
    return stream, docs


def conversational_rag(
    user_input: str,
    chat_history: List[BaseMessage]
):

    docs = retriever.invoke(user_input)

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    messages = prompt.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
    })

    response = llm.stream(messages)

    return response, docs