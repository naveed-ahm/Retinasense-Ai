from pydantic_settings import BaseSettings
from pathlib import Path
import warnings


class Settings(BaseSettings):
    APP_NAME: str = "RetinaSense AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database — SQLite (dev) / PostgreSQL (prod)
    DATABASE_URL: str = "sqlite+aiosqlite:///./retinasense.db"
    POSTGRES_USER: str = "retinasense"
    POSTGRES_PASSWORD: str = "retinasense_secret"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "retinasense"

    @property
    def async_database_url(self) -> str:
        if self.DATABASE_URL.startswith("postgresql"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return self.DATABASE_URL

    # Auth
    SECRET_KEY: str = "retinasense-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 720

    # Paths
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent / "uploads")
    REPORT_DIR: str = str(Path(__file__).resolve().parent.parent / "reports")
    MODEL_PATH: str = str(Path(__file__).resolve().parent / "ai" / "models" / "retinasense_model.keras")

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://frontend:5173",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

if settings.SECRET_KEY == "retinasense-secret-key-change-in-production":
    warnings.warn(
        "SECRET_KEY is using the default value. Set SECRET_KEY env var in production!",
        UserWarning,
        stacklevel=2,
    )
