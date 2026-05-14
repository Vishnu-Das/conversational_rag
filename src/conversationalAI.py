from fileinput import filename
import uuid, os, time
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
from src.rag import (
    stream_response,
    reset_rag_caches
)
from src.utils.citations import extract_sources
from src.vectorstore import (
    get_available_documents
)
from src.config import (
    DATA_DIR
)
from src.ingest import ingest_single_document

def run_app(): ## This function sets up the Streamlit app, including page configuration, sidebar for chat sessions, and the main interface for displaying chat history and handling user input. It also manages the state of the current chat session and interacts with the response streaming and source extraction functions.

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

    st.title("Conversational AI with RAG")

    documents = get_available_documents()

    selected_document = st.sidebar.selectbox(
        "Filter Documents",
        ["All Documents"] + documents
    )

    # Sidebar
    st.sidebar.header("Chats")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = None

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:

        filename = uploaded_file.name

        if filename not in st.session_state.processed_uploads:

            save_path = os.path.join(
                DATA_DIR,
                uploaded_file.name
            )

            # Prevent duplicate ingestion
            if not os.path.exists(save_path):

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                ingest_single_document(save_path)

                reset_rag_caches()

                st.session_state.processed_uploads.add(
                    filename
                )

                st.session_state.upload_success = {
                    "message": f"{filename} uploaded successfully!",
                    "timestamp": time.time()
                }

                # Reset uploader widget
                st.session_state.uploader_key += 1

                st.rerun()

            else:
                st.sidebar.warning(
                    "File already exists."
                )
    if st.session_state.upload_success:

        st.sidebar.success(
            st.session_state.upload_success["message"]
        )

        time.sleep(3)

        st.session_state.upload_success = None

        st.rerun()

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
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

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
            st.session_state.chat_history,
            selected_document
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