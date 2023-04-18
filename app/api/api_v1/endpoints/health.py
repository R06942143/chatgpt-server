from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health() -> Any:
    return "health"
