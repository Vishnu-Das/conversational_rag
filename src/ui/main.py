import streamlit as st

from src.ui.styles import APP_STYLES

from src.ui.sidebar import (
    render_sidebar
)

from src.ui.welcome import (
    render_welcome_screen
)

from src.ui.chat import (
    render_chat_history,
    handle_chat_input
)

from src.services.session_service import (
    initialize_session
)

from src.database import (
    load_chat_history
)


def run_app():

    st.set_page_config(
        page_title="Conversational AI",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session()

    st.markdown(
        APP_STYLES,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 style='
            text-align:center;
            font-size:3rem;
            margin-bottom:1rem;
        '>
            🤖 Conversational AI with RAG
        </h1>
        """,
        unsafe_allow_html=True
    )

    session_id = st.session_state.session_id

    if not st.session_state.chat_history:

        st.session_state.chat_history = (
            load_chat_history(session_id)
        )

    selected_document = render_sidebar()

    show_welcome_screen = (
        len(st.session_state.chat_history) == 0
        and not st.session_state.get(
            "is_processing_prompt",
            False
        )
    )

    if show_welcome_screen:

        render_welcome_screen(session_id)

    render_chat_history()

    handle_chat_input(selected_document)