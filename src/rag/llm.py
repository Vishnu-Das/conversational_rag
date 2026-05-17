from langchain_openai import ChatOpenAI
from src.config import MAIN_MODEL_NAME

llm = ChatOpenAI(
    model=MAIN_MODEL_NAME,
    streaming=True
)