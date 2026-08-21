from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGORA_", extra="ignore")

    database_url: str = "postgresql://agora:agora@127.0.0.1:5433/agora"
    redis_url: str = "redis://127.0.0.1:6379/0"
    small_model: str = "gpt-5.6-luna"
    big_model: str = "gpt-5.6-terra"


@lru_cache
def get_settings() -> Settings:
    return Settings()
