from typing import Generator

from fastapi.security import OAuth2PasswordBearer

from app.core.config import settings
from app.core.db.session import SessionLocal

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_PREFIX}/v2/auth/access-token"
)


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
