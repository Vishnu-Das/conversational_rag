from langchain_openai import ChatOpenAI
from src.config import MODEL_NAME

llm = ChatOpenAI(
    model=MODEL_NAME,
    streaming=True
)