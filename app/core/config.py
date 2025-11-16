import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PATE-GAN Synthetic Data API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Airflow Settings
    AIRFLOW_API_URL: str = ""
    AIRFLOW_API_USER: str = ""
    AIRFLOW_API_PASS: str = ""

    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # DB Settings
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
    
    # File Storage
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    UPLOAD_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Model Training Defaults
    DEFAULT_LATENT_DIM: int = 128
    DEFAULT_TEACHER_EPOCHS: int = 100
    DEFAULT_STUDENT_EPOCHS: int = 100
    DEFAULT_GENERATOR_EPOCHS: int = 100
    DEFAULT_NUM_TEACHERS: int = 10
    DEFAULT_EPSILON: float = 1.0
    DEFAULT_DELTA: float = 1e-5
    DEFAULT_LEARNING_RATE: float = 0.0002
    DEFAULT_BATCH_SIZE: int = 64
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Path(self.DATA_DIR).mkdir(exist_ok=True)
        Path(self.MODEL_DIR).mkdir(exist_ok=True)

settings = Settings()