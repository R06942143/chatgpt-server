from time import time
from uuid import uuid4
import sys
import copy
import traceback
from app.core.logger import logger
from starlette.concurrency import iterate_in_threadpool
import json
import rollbar
from fastapi import FastAPI, Request, Response
from fastapi.security.utils import get_authorization_scheme_param
from rollbar.contrib.fastapi import ReporterMiddleware as RollbarMiddleware
from starlette.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router
from app.core.config import settings


from app.models.dynamodb import conversation  # noqa

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


@LogzioFlusher(logger)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    authorization: str = request.headers.get("Authorization")
    _, token = get_authorization_scheme_param(authorization)

    try:

        def fake_auth(token: str):
            return True

        user = fake_auth(token=token)
        request.state.user = user
    except Exception as e:  # noqa
        pass

    trace_id = str(uuid4())
    request_dict = {
        "s_service": "coobyGPT",
        "trace_id": trace_id,
        "path": request.url.path,
        "method": request.method,
        "env": settings.ENVIRONMENT,
        "origin": request.headers.get("origin"),
    }

    logger.info("request", extra=request_dict)
    start_time = time()
    try:
        response = await call_next(request)
    except Exception as error:
        error_dict = copy.deepcopy(request_dict)
        exc_type, exc_obj, tb = sys.exc_info()
        trace = traceback.format_exception(exc_type, exc_obj, tb)
        error_dict["trace"] = trace
        logger.error(repr(error), extra=error_dict)
        rollbar.report_exc_info()
        return Response(status_code=500)
    else:
        process_time = (time() - start_time) * 1000

        response_dict = copy.deepcopy(request_dict)
        response_dict["process_time_ms"] = process_time
        response_dict["status_code"] = response.status_code
        if response.status_code >= 400:
            response_body = [chunk async for chunk in response.body_iterator]
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            if response_body:
                try:
                    response_body = json.loads(response_body[0].decode())
                    if "message" in response_body:
                        response_dict["error_message"] = response_body["message"]
                    if "code" in response_body:
                        response_dict["error_code"] = response_body["code"]
                except:  # noqa
                    pass

        logger.info(f"Response status_code={response.status_code}", extra=response_dict)
        return response
