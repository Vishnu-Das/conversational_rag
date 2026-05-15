import streamlit as st

SUGGESTED_PROMPTS = [
    "Summarize this document",
    "Explain the key concepts",
    "Generate study notes",
    "What are the important topics?",
]


def render_welcome_screen(session_id):

    st.markdown(
        """
        <div style="text-align:center; padding-top:80px;">

        <h2>👋 Welcome</h2>

        <p style="font-size:18px; color:#9CA3AF;">
            Upload PDFs and start chatting with your documents.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    cards = [
        ("📄", "Summarize PDFs"),
        ("🧠", "Ask Technical Questions"),
        ("🔍", "Search Across Documents")
    ]

    for col, (emoji, text) in zip(
        [col1, col2, col3],
        cards
    ):

        with col:

            st.markdown(
                f"""
                <div style="
                    padding:14px;
                    border-radius:12px;
                    background-color:#1E1E1E;
                    text-align:center;
                ">
                    <h3>{emoji}</h3>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### ✨ Try Asking")

    prompt_cols = st.columns(2)

    for index, prompt in enumerate(
        SUGGESTED_PROMPTS
    ):

        with prompt_cols[index % 2]:

            if st.button(
                prompt,
                key=f"prompt_{session_id}_{index}",
                use_container_width=True
            ):
                st.session_state.selected_prompt = prompt
                st.session_state.is_processing_prompt = True
                st.rerun()