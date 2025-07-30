from langchain_openai import OpenAIEmbeddings
from config.settings import settings

def get_embedding_model():
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )