from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor
)

from src.config import MODEL_NAME

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0
)

compressor = LLMChainExtractor.from_llm(llm)

def compress_documents(query, documents):
    if not documents:
        return []
    compressed_docs = compressor.compress_documents(
        documents=documents,
        query=query
    )
    return compressed_docs