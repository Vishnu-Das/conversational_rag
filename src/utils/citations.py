import os
import html
import re


def extract_sources(documents):

    formatted_sources = []

    for doc in documents:

        source = os.path.basename(
            doc.metadata.get("source", "Unknown")
        )

        page = doc.metadata.get(
            "page",
            "N/A"
        )

        preview = doc.page_content

        # Remove ALL newlines/tabs
        preview = re.sub(
            r"[\n\r\t]+",
            " ",
            preview
        )

        # Remove extra spaces
        preview = re.sub(
            r"\s+",
            " ",
            preview
        ).strip()

        # Escape HTML safely
        preview = html.escape(preview)

        # Limit preview size
        preview = preview[:220]

        formatted_sources.append({
            "source": source,
            "page": page,
            "preview": preview
        })

    return formatted_sources