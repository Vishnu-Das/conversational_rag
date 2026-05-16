import os
import time
import streamlit as st

from src.config import DATA_DIR
from src.ingest import ingest_single_document
from src.rag.cache import reset_rag_caches


def handle_file_upload():

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is None:
        return

    filename = uploaded_file.name

    if filename in st.session_state.processed_uploads:
        return

    save_path = os.path.join(
        DATA_DIR,
        filename
    )

    if os.path.exists(save_path):

        st.sidebar.warning(
            "File already exists."
        )
        return

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

    st.session_state.uploader_key += 1

    st.rerun()


def render_upload_success():

    if not st.session_state.upload_success:
        return

    elapsed = (
        time.time()
        - st.session_state.upload_success["timestamp"]
    )

    if elapsed < 3:

        st.sidebar.success(
            st.session_state.upload_success["message"]
        )

    else:

        st.session_state.upload_success = None