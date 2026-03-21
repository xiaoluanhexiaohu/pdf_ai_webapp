from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "PDF AI 智能插图系统"
    ai_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    max_image_width_ratio: float = 0.42
    max_image_height_ratio: float = 0.28
    default_insert_gap: int = 12
    temp_dir: str = "temp"
    upload_dir: str = "uploads"
    output_dir: str = "outputs"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    (BASE_DIR / settings.temp_dir).mkdir(parents=True, exist_ok=True)
    (BASE_DIR / settings.upload_dir).mkdir(parents=True, exist_ok=True)
    (BASE_DIR / settings.output_dir).mkdir(parents=True, exist_ok=True)
    return settings
