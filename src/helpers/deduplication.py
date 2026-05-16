from typing import List
from langchain_core.documents import Document
from langsmith import traceable

@traceable(name="Deduplicate Documents")
def deduplicate_docs(docs: List[Document]) -> List[Document]:

    seen = set()
    unique_docs = []

    for doc in docs:

        # best unique key = content + source
        source = doc.metadata.get("source", "")
        key = (doc.page_content, source)

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs