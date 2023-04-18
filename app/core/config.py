import os
import secrets
from typing import Any, Dict, List, Optional, Union

import boto3
from fastapi.templating import Jinja2Templates
from pydantic import AnyHttpUrl, BaseSettings, root_validator

client = boto3.client("secretsmanager")
SECRET_NAME = f"{os.environ.get('ENVIRONMENT', 'staging')}/JWTSecret"
ROLLBAR_NAME = f"{os.environ.get('ENVIRONMENT', 'staging')}/RollbarToken"
SQL_NAME = f"{os.environ.get('ENVIRONMENT', 'staging')}/SqlalchemyDatabaseURI"
CONSUMER_NAME = f"{os.environ.get('ENVIRONMENT', 'staging')}/ConsumerID"
CONSUMER_SECRET_NAME = f"{os.environ.get('ENVIRONMENT', 'staging')}/ConsumerSecret"


class Settings(BaseSettings):
    ROLLBAR_TOKEN: str = "client.get_secret_value(SecretId=ROLLBAR_NAME)['SecretString']"
    ENVIRONMENT: str = "local"
    API_PREFIX: str = "/chatgpt/api/v1"
    ROOT_PATH: str = ""
    DOCS_URL: Optional[str] = "/docs"
    REDOC_URL: Optional[str] = None
    SECRET_KEY: str = "client.get_secret_value(SecretId=SECRET_NAME)['SecretString']"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    OAUTH_TOKEN_EXPIRE_MINUTES: int = 2
    SERVER_NAME: str = "Ai Assistant"
    PROJECT_NAME: str = "Ai Assistant"
    BACKEND_CORS_ORIGINS: list[Optional[str]] = []
    BACKEND_CORS_ORIGINS: List[str] = []
    SERVER_HOST: AnyHttpUrl = "http://localhost:8000"
    SQLALCHEMY_DATABASE_URI: str = (
        os.environ.get("SQLALCHEMY_DATABASE_URI", "postgresql://test:password@127.0.0.1:5432/test")
    )

    class Config:
        case_sensitive = True


settings = Settings()
