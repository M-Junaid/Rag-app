from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 128
    VECTOR_DB_PATH: str = "./data/vectorstore"
    TOP_K: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings()