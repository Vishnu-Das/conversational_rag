from typing import List

from langchain_core.messages import (
    BaseMessage,
    SystemMessage
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_openai import ChatOpenAI

from src.vectorstore import load_vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)

prompt = ChatPromptTemplate.from_messages([

    SystemMessage(
        content="""
You are a helpful conversational AI assistant.

Rules:
1. Use chat history as the primary source
   for conversational memory.

2. Use retrieved context only for
   answering factual questions about documents.

3. If the user specifies their identity,
   preferences, or personal details in chat,
   prefer those over retrieved document content.

4. Do not assume the user is the same person
   described in the retrieved documents.

5. If answer is unavailable in the chat history or retrieved context, say you don't know.

6. Very importantly : if the question is unrelated to the retrieved documents and chat history then say I dont know.
"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "system",
        "Retrieved Context:\n{context}"
    ),

    (
        "human",
        "{input}"
    )
])


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

    response = llm.invoke(messages)

    return response, docs