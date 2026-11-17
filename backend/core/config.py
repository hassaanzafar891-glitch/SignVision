from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    KEYPOINTS_DIR: str = "Keypoints"
    DATA_DIR: str = "PSL/data"
    SPEECH_DIR: str = "data/speech"
    CAMERA_INDEX: int = 0

    MONITORING_ENABLED: bool = True
    MONITORING_TOKEN: SecretStr = SecretStr("")
    MONITORING_WINDOW_SIZE: int = 1000
    MONITORING_MIN_SAMPLES: int = 100
    MONITORING_MIN_REFERENCE_SAMPLES: int = 100
    MONITORING_DRIFT_THRESHOLD: float = 0.1
    MONITORING_BASELINE_PATH: str = ""

    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "wow.992du@gmail.com"
    SMTP_PASS: str = ""
    CONTACT_TO: str = "wow.992du@gmail.com"

    class Config:
        env_file = ".env"

settings = Settings()
