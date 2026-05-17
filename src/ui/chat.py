import streamlit as st

from textwrap import dedent

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from src.database import save_message

from src.rag.service import stream_response

from src.utils.citations import (
    extract_sources
)

from src.ui.retrieval_inspector import (
    render_retrieval_inspector
)

from src.config import (
    ENABLE_RETRIEVAL_INSPECTOR
)

def render_chat_history():

    for msg in st.session_state.chat_history:

        if isinstance(msg, HumanMessage):

            st.markdown(
                dedent(f"""
                <div class="user-message">
                    <strong>🧑 You</strong><br><br>
                    {msg.content}
                </div>
                """),
                unsafe_allow_html=True
            )

        elif isinstance(msg, AIMessage):

            st.markdown(
                dedent(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong><br><br>
                    {msg.content}
                </div>
                """),
                unsafe_allow_html=True
            )

        elif isinstance(msg, SystemMessage):

            st.info(msg.content)


def handle_chat_input(selected_document):

    user_input = st.chat_input(
        "Ask a question about the PDF..."
    )

    if st.session_state.get("selected_prompt"):

        user_input = (
            st.session_state.selected_prompt
        )

        st.session_state.selected_prompt = None

    if not user_input:
        return None

    session_id = st.session_state.session_id

    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    save_message(
        session_id,
        "human",
        user_input
    )

    st.markdown(
        dedent(f"""
        <div class="user-message">
            <strong>🧑 You</strong><br><br>
            {user_input}
        </div>
        """),
        unsafe_allow_html=True
    )

    status_placeholder = st.empty()

    status_placeholder.markdown(
        dedent("""
        <div class="status-message">
            🔍 Retrieving and analyzing documents...
        </div>
        """),
        unsafe_allow_html=True
    )

    stream, sources, debug_info  = stream_response(
        user_input,
        st.session_state.chat_history,
        selected_document
    )

    st.session_state.last_retrieval_debug = (
        debug_info
    )

    assistant_response = ""

    response_placeholder = st.empty()

    status_placeholder.markdown(
        dedent("""
        <div class="status-message">
            ✍️ Generating response...
        </div>
        """),
        unsafe_allow_html=True
    )

    for chunk in stream:

        if chunk.content:

            status_placeholder.empty()

            assistant_response += chunk.content

            response_placeholder.markdown(
                dedent(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong><br><br>
                    {assistant_response}▌
                </div>
                """),
                unsafe_allow_html=True
            )

    response_placeholder.markdown(
        dedent(f"""
        <div class="assistant-message">
            <strong>🤖 Assistant</strong><br><br>
            {assistant_response}
        </div>
        """),
        unsafe_allow_html=True
    )

    formatted_sources = extract_sources(sources)

    if formatted_sources:

        with st.expander(
            f"📚 Sources ({len(formatted_sources)})",
            expanded=False
        ):

            for item in formatted_sources:

                source_html = f"""
                <div class="source-card">
                    <div class="source-title">
                        📄 {item['source']} — Page {item['page']}
                    </div>
                    <div class="source-preview-box">
                        {item['preview']}...
                    </div>
                </div>
                """

                st.markdown(
                    source_html,
                    unsafe_allow_html=True
                )
    if ENABLE_RETRIEVAL_INSPECTOR:
        render_retrieval_inspector(
            st.session_state.get("last_retrieval_debug")
        )

    save_message(
        session_id,
        "ai",
        assistant_response
    )

    st.session_state.chat_history.append(
        AIMessage(content=assistant_response)
    )

    st.session_state.is_processing_prompt = False

    return user_input