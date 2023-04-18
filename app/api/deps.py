from datetime import datetime, timezone
from typing import Generator

from dateutil.parser import parse
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app import models, schemas
from app.core.config import settings
from app.core.db.session import SessionLocal
from app.http_utils import (
    UnConnected,
    UnOAuth,
    UserNotFound,
)
from app.repositories import rds

reusable_oauth2 = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/v2/auth/access-token")


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

