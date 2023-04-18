from fastapi import APIRouter

from app.api.api_v1.endpoints import (
    health,
    chatgpt,
)

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chatgpt.router, prefix="/chatgpt", tags=["chatgpt"])
