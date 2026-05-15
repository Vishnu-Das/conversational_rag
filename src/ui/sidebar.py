import uuid
import streamlit as st

from src.database import (
    load_chat_history,
    get_all_sessions,
    get_chat_preview
)

from src.vectorstore import (
    get_available_documents
)

from src.services.upload_service import (
    handle_file_upload,
    render_upload_success
)


def render_sidebar():

    documents = get_available_documents()

    st.sidebar.markdown(
        "## 🤖 User Functions"
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "### 📂 Documents"
    )

    selected_document = st.sidebar.selectbox(
        "Filter Documents",
        ["All Documents"] + documents
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "### 📤 Upload PDF"
    )

    handle_file_upload()

    render_upload_success()

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 💬 Chats")

    if st.sidebar.button("+ New Chat"):

        st.session_state.session_id = (
            str(uuid.uuid4())
        )

        st.session_state.chat_history = []

        st.rerun()

    for sid in get_all_sessions():

        preview = get_chat_preview(sid)

        if st.sidebar.button(
            preview,
            key=sid
        ):

            st.session_state.session_id = sid

            st.session_state.chat_history = (
                load_chat_history(sid)
            )

            st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Built with LangChain • OpenAI • Chroma"
    )

    return selected_document