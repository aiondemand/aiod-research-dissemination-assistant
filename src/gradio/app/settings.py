from pydantic import AnyHttpUrl, BaseModel
from pydantic_settings import BaseSettings


class AIODKeycloakConfig(BaseModel):
    REALM: str
    CLIENT_ID: str
    CLIENT_SECRET: str
    SERVER_URL: AnyHttpUrl


class Settings(BaseSettings):
    summarization_model: str

    ollama_url: str
    generation_ollama_model: str

    aiod_keycloak: AIODKeycloakConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


settings = Settings()
