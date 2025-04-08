from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qdrant_path: str = "data/qdrant"
    collection: str = "chunks"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    llm_provider: str = "mock"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"
    llm_api_key: str = ""
