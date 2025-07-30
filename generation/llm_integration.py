from langchain_openai import ChatOpenAI
from config.settings import settings

def get_llm():
    return ChatOpenAI(
        model_name=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.3
    )