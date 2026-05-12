from pathlib import Path


def extract_sources(docs):
    formatted_sources = []
    displayed_sources = set()
    for doc in docs:
        source = doc.metadata.get(
            "source",
            "Unknown"
        )
        page = doc.metadata.get(
            "page",
            "Unknown"
        )
        source_name = Path(source).name
        unique_key = f"{source_name}-{page}"
        if unique_key not in displayed_sources:
            formatted_sources.append({
                "source": source_name,
                "page": page + 1,
                "preview": doc.page_content[:300]
            })
            displayed_sources.add(unique_key)
    return formatted_sources