from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    summarization_model: str = "Falconsai/text_summarization"

    ollama_url: str = "http://localhost:11434"
    generation_ollama_model: str = "llama3.1"

    feedback_file_simple: str = "simple_feedback.csv"
    feedback_file_detailed: str = "detailed_feedback.csv"

    class Config:
        env_file = ".env"


settings = Settings()
