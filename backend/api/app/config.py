"""Application configuration"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API settings
    api_host: str = "localhost"
    api_port: int = 8000
    environment: str = "development"
    secret_key: str = "your-secret-key-change-in-production"

    # Database
    database_url: str = "sqlite:///./wave_forecast.db"

    # NOAA
    noaa_api_key: str = ""
    noaa_base_url: str = "https://api.weather.gov"

    # NDBC (buoy data)
    ndbc_base_url: str = "https://www.ndbc.noaa.gov"

    # ML Models
    model_storage_bucket: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-west-2"

    # Redis/Celery
    redis_url: str = "redis://localhost:6379"


settings = Settings()
