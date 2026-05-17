import streamlit as st


def render_metric_card(
    label,
    value
):

    st.markdown(
        f"""
        <div style="
            background-color:#111827;
            border:1px solid #374151;
            border-radius:14px;
            padding:14px;
            min-height:90px;
        ">
            <div style="
                color:#9CA3AF;
                font-size:13px;
                margin-bottom:8px;
            ">
                {label}
            </div>
            <div style="
                color:#F9FAFB;
                font-size:24px;
                font-weight:700;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_chunk_card(
    doc,
    index
):

    source = doc.get(
        "source",
        "Unknown"
    )

    page = doc.get(
        "page",
        "N/A"
    )

    preview = doc.get(
        "preview",
        ""
    )

    score = doc.get(
        "score"
    )

    retrieval_source = doc.get(
        "retrieval_source",
        "N/A"
    )

    title = f"Chunk {index} • {source} • Page {page}"

    with st.expander(
        title,
        expanded=False
    ):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.caption(
                f"Source: {source}"
            )

        with col2:
            st.caption(
                f"Page: {page}"
            )

        with col3:
            st.caption(
                f"Retrieved By: {retrieval_source}"
            )

        with col4:
            if score is not None:
                st.caption(
                    f"Score: {score}"
                )
            else:
                st.caption(
                    "Score: N/A"
                )

        st.markdown(
            f"""
            <div style="
                max-height:180px;
                overflow-y:auto;
                background-color:#020617;
                border:1px solid #1F2937;
                border-radius:12px;
                padding:14px;
                font-size:14px;
                line-height:1.6;
                color:#E5E7EB;
                white-space:pre-wrap;
            ">
                {preview}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_chunk_section(
    title,
    docs
):

    st.markdown(
        f"### {title}"
    )

    if not docs:

        st.caption(
            "No documents available."
        )
        return

    st.markdown(
        f"""
        <div style="
            max-height:520px;
            overflow-y:auto;
            padding-right:6px;
        ">
        """,
        unsafe_allow_html=True
    )

    for index, doc in enumerate(
        docs,
        start=1
    ):

        render_chunk_card(
            doc,
            index
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


def render_retrieval_inspector(
    debug_info
):

    if not debug_info:
        return

    with st.expander(
        "🔎 Retrieval Inspector",
        expanded=False
    ):

        st.markdown(
            "## Retrieval Summary"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_metric_card(
                "Requested Strategy",
                debug_info.requested_strategy
            )

        with col2:
            render_metric_card(
                "Resolved Strategy",
                debug_info.resolved_strategy
            )

        with col3:
            render_metric_card(
                "Retrieved Docs",
                debug_info.retrieved_docs_count
            )

        with col4:
            render_metric_card(
                "Final Docs",
                debug_info.final_docs_count
            )

        st.markdown("")

        st.markdown(
            "## Latency Breakdown"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            render_metric_card(
                "Retrieval",
                f"{debug_info.retrieval_latency_ms} ms"
            )

        with col2:
            render_metric_card(
                "Reranking",
                f"{debug_info.rerank_latency_ms} ms"
            )

        with col3:
            render_metric_card(
                "Total",
                f"{debug_info.total_latency_ms} ms"
            )

        if debug_info.router_reason:

            st.markdown(
                "## Router Decision"
            )

            col1, col2 = st.columns(2)

            with col1:

                render_metric_card(
                    "Router Type",
                    debug_info.router_type
                )

            with col2:

                confidence = (
                    f"{round(debug_info.router_confidence * 100, 1)}%"
                    if debug_info.router_confidence
                    is not None
                    else "N/A"
                )

                render_metric_card(
                    "Router Confidence",
                    confidence
                )
            
            st.markdown("<br>", unsafe_allow_html=True)

            st.info(
                debug_info.router_reason
            )

        st.divider()

        tab1, tab2 = st.tabs([
            "Retrieved Chunks",
            "Final Chunks"
        ])

        with tab1:
            render_chunk_section(
                "Retrieved Chunks",
                debug_info.retrieved_docs
            )

        with tab2:
            render_chunk_section(
                "Final Chunks After Reranking",
                debug_info.final_docs
            )