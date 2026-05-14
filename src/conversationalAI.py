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

        section[data-testid="stSidebar"] {
            width: 320px !important;
        }

        div.stButton > button {
            width: 100%;
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            border-radius: 10px;
            margin-bottom: 6px;
            padding: 0.5rem;
        }

        .user-message {
            background-color: #3A3A3C; /* Balanced dark grey */
            padding: 14px;
            border-radius: 14px;
            margin-bottom: 12px;
            color: #FFFFFF;
        }

        .assistant-message {
            background-color: #1C1C1E; /* Deeper off-black */
            padding: 14px;
            border-radius: 14px;
            margin-bottom: 12px;
            border: 1px solid #2C2C2E;
            color: #F2F2F7;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 style='
            text-align: center;
            font-size: 3rem;
            margin-bottom: 1rem;
        '>
            🤖 Conversational AI with RAG
        </h1>
        """,
        unsafe_allow_html=True
    )

    documents = get_available_documents()

    # Sidebar
    st.sidebar.markdown(
        """
        ## 🤖 User Dashboard
        """
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📂 Documents")

    selected_document = st.sidebar.selectbox(
        "Filter Documents",
        ["All Documents"] + documents
    )

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = None
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Upload PDF")

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

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Chats")

    if st.sidebar.button("+  New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []

    for sid in get_all_sessions():
        preview = get_chat_preview(sid)
        if st.sidebar.button(preview, key=sid):
            st.session_state.session_id = sid
            st.session_state.chat_history = load_chat_history(sid)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built with LangChain • OpenAI • Chroma"
    )

    session_id = st.session_state.session_id

    # Initial load
    if not st.session_state.chat_history:

        st.session_state.chat_history = load_chat_history(
            session_id
        )
    
    # User input
    user_input = st.chat_input(
        "Ask a question about the PDF..."
    )

    # Welcome screen for empty chats
    if (
        len(st.session_state.chat_history) == 0 and not user_input
    ):

        st.markdown(
            """
            <div style="text-align: center; padding-top: 80px;">

            <h2>
                👋 Welcome
            </h2>

            <p style="font-size:18px; color: #9CA3AF;">
                Upload PDFs and start chatting with your documents.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div style="
                    padding:14px;
                    border-radius:12px;
                    background-color:#1E1E1E;
                    text-align:center;
                ">
                    <h3>📄</h3>
                    <p>Summarize PDFs</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div style="
                    padding:14px;
                    border-radius:12px;
                    background-color:#1E1E1E;
                    text-align:center;
                ">
                    <h3>🧠</h3>
                    <p>Ask Technical Questions</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                """
                <div style="
                    padding:14px;
                    border-radius:12px;
                    background-color:#1E1E1E;
                    text-align:center;
                ">
                    <h3>🔍</h3>
                    <p>Search Across Documents</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Render history
    for msg in st.session_state.chat_history:
        # Render history
        if isinstance(msg, HumanMessage):

            st.markdown(
                f"""
                <div class="user-message">
                    <strong>🧑 You</strong><br><br>
                    {msg.content}
                </div>
                """,
                unsafe_allow_html=True
            )

        elif isinstance(msg, AIMessage):

            st.markdown(
                f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong><br><br>
                    {msg.content}
                </div>
                """,
                unsafe_allow_html=True
            )

        elif isinstance(msg, SystemMessage):

            st.info(msg.content)

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

        with st.container():

            response_placeholder = st.empty()
            for chunk in stream:
                if chunk.content:
                    assistant_response += chunk.content
                    response_placeholder.markdown(
                        f'''
                        <div class="assistant-message">
                            <strong>🤖 Assistant</strong><br><br>
                            {assistant_response}▌
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            response_placeholder.markdown(
                f'''
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong><br><br>
                    {assistant_response}
                </div>
                ''',
                unsafe_allow_html=True
            )

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