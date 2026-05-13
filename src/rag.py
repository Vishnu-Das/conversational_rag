from typing import List
import os

from langchain_core.messages import (
    BaseMessage
)

from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_community.retrievers import (
    BM25Retriever,
)

from langchain_classic.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever
)


from langchain_openai import ChatOpenAI

from src.vectorstore import (
    load_vectorstore,
    load_and_split_documents
)

from src.config import (
    MODEL_NAME,
    RERANK_TOP_K,
    INITIAL_RETRIEVAL_K
)

from src.reranker import (
    rerank_documents
)

## Load and split documents at startup to avoid doing it on every query
all_documents  = load_and_split_documents()

## Prompt to reformulate the user's question in the context of the chat history
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

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{input}")
])

## Load the vectorstore at startup to avoid doing it on every query
vectorstore = load_vectorstore()

## Initialize the LLM with streaming enabled for real-time response generation
llm = ChatOpenAI(
    model=MODEL_NAME,  streaming=True
)

## Function to create a retriever based on the selected document filter.
def get_retriever(selected_document=None):

    search_kwargs = {
        "k": INITIAL_RETRIEVAL_K
    }
    filtered_documents = all_documents
    if (
        selected_document
        and selected_document != "All Documents"
    ):
        search_kwargs["filter"] = {
            "source": selected_document
        }

        filtered_documents = [

            doc for doc in all_documents

            if os.path.basename(
                doc.metadata.get("source", "")
            ) == selected_document
        ]

    vector_retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )
    if not filtered_documents:
        return vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )

    bm25_retriever = BM25Retriever.from_documents(
        filtered_documents
    )
    bm25_retriever.k = 4
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever
        ],
        weights=[0.7, 0.3]
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )
    return multi_query_retriever

## Prompt template for the LLM to generate answers based on retrieved context and chat history.
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

## Main function to handle user input, retrieve relevant documents,
## and generate a streamed response from the LLM
def stream_response(
    user_input: str,
    chat_history: List[BaseMessage],
    selected_document: str = None
):

    retriever = get_retriever(selected_document)

    # print("\n========== DEBUG ==========")
    # print("Selected Document:", selected_document)

    ## Create a history-aware retriever that reformulates the question

    history_aware_retriever = (
        create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )
    )

    docs = history_aware_retriever.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })

    ## Rerank retrieved documents based on relevance to the question and chat history

    docs = rerank_documents(
        user_input,
        docs,
        top_k=RERANK_TOP_K
    )

    # print("\nRetrieved Documents:")
    # for doc in docs:
    #     print(
    #         doc.metadata.get("source")
    #     )
    # print("===========================\n")

    context = "\n\n---\n\n".join([
        doc.page_content for doc in docs
    ])

    messages = prompt.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": context
    })

    stream = llm.stream(messages)

    return stream, docs