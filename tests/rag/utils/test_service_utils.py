from unittest.mock import patch

from langchain_core.documents import Document

from src.rag.service import (
    build_context,
    get_selected_document_chunks,
    normalize_source_name,
)


def test_normalize_source_name_handles_windows_path():
    result = normalize_source_name(
        r"C:\Users\vishn\docs\attention.pdf"
    )

    assert result == "attention.pdf"


def test_normalize_source_name_handles_unix_path():
    result = normalize_source_name(
        "/home/user/docs/attention.pdf"
    )

    assert result == "attention.pdf"


def test_normalize_source_name_handles_empty_source():
    assert normalize_source_name("") == ""


def test_build_context_joins_documents_with_separator():
    docs = [
        Document(page_content="First chunk"),
        Document(page_content="Second chunk"),
    ]

    context = build_context(docs)

    assert context == "First chunk\n\n---\n\nSecond chunk"


@patch("src.rag.service.get_all_documents")
def test_get_selected_document_chunks_filters_by_normalized_source(
    mock_get_all_documents,
):
    matching_doc = Document(
        page_content="Matching document",
        metadata={"source": r"C:\docs\attention.pdf"},
    )

    non_matching_doc = Document(
        page_content="Other document",
        metadata={"source": r"C:\docs\other.pdf"},
    )

    mock_get_all_documents.return_value = [
        matching_doc,
        non_matching_doc,
    ]

    result = get_selected_document_chunks(
        selected_document="attention.pdf",
    )

    assert result == [matching_doc]


@patch("src.rag.service.get_all_documents")
def test_get_selected_document_chunks_respects_max_chunks(
    mock_get_all_documents,
):
    docs = [
        Document(
            page_content=f"Chunk {index}",
            metadata={"source": "attention.pdf"},
        )
        for index in range(5)
    ]

    mock_get_all_documents.return_value = docs

    result = get_selected_document_chunks(
        selected_document="attention.pdf",
        max_chunks=2,
    )

    assert result == docs[:2]