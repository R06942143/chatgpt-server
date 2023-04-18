from time import time
from uuid import uuid4

import rollbar
from fastapi import FastAPI
from mangum import Mangum
from rollbar.contrib.fastapi import ReporterMiddleware as RollbarMiddleware
from starlette.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router
from app.core.config import settings


from app.models.dynamodb import conversation

rollbar.init(
    settings.ROLLBAR_TOKEN,
    environment="內卷",
    handler="async",
)


def ignore_handler(payload, **kw):  # kw is currently unused
    if True:
        return False
    else:
        return payload


rollbar.events.add_payload_handler(ignore_handler)


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    root_path=settings.ROOT_PATH,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
)

app.add_middleware(RollbarMiddleware)

# TODO add trace_id ###

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix=settings.API_PREFIX)


handler = Mangum(app)
