from langchain_openai import ChatOpenAI
from .config import settings

def get_llm(temperature: float = 0.7):
    """
    LLM Factory using configured settings.
    """
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_BASE_URL,
        temperature=temperature
    )
