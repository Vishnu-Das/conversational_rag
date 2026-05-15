import uuid
import streamlit as st


def initialize_session():

    defaults = {
        "session_id": str(uuid.uuid4()),
        "chat_history": [],
        "processed_uploads": set(),
        "uploader_key": 0,
        "upload_success": None,
        "selected_prompt": None,
        "is_processing_prompt": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value