from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    rate_limit: str = "1/second"

    class Config:
        env_file = ".env"


settings = Settings()
