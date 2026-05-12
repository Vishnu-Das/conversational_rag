import uuid, os
import streamlit as st
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from src.database import (
    save_message,
    load_chat_history,
    get_all_sessions,
    get_chat_preview
)
from src.rag import conversational_rag
from src.rag import stream_response
from src.utils.citations import extract_sources

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def run_app():

    st.set_page_config(
        page_title="Conversational AI",
        page_icon="🤖",
        layout="wide"
    )

    st.markdown(
    """
    <style>

    div.stButton > button {
        width: 100%;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    </style>
    """,
    unsafe_allow_html=True
)

    st.title("Agent By Vishnu Das")

    # Sidebar
    st.sidebar.header("Chats")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []

    if st.sidebar.button("New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []

    st.sidebar.markdown("### Previous Chats")

    for sid in get_all_sessions():
        preview = get_chat_preview(sid)
        if st.sidebar.button(preview, key=sid):
            st.session_state.session_id = sid
            st.session_state.chat_history = load_chat_history(sid)

    session_id = st.session_state.session_id

    # Initial load
    if not st.session_state.chat_history:

        st.session_state.chat_history = load_chat_history(
            session_id
        )

    # Render history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)

    # User input
    user_input = st.chat_input(
        "Ask a question about the PDF..."
    )

    if user_input:
        st.chat_message("user").write(user_input)
        save_message(
            session_id,
            "human",
            user_input
        )
        st.session_state.chat_history.append(
            HumanMessage(content=user_input)
        )
        stream, sources = stream_response(
            user_input,
            st.session_state.chat_history
        )

        assistant_response = ""

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            for chunk in stream:
                if chunk.content:
                    assistant_response += chunk.content
                    response_placeholder.markdown(
                        assistant_response + "▌"
                    )
            response_placeholder.markdown(assistant_response)

        # Show sources
        formatted_sources = extract_sources(sources)

        if formatted_sources:
            with st.expander("Sources"):
                for item in formatted_sources:
                    st.markdown(
                        f"### {item['source']} "
                        f"(Page {item['page']})"
                    )
                    st.caption(
                        item["preview"] + "..."
                    )

        save_message(
            session_id,
            "ai",
            assistant_response
        )

        st.session_state.chat_history.append(
            AIMessage(content=assistant_response)
        )