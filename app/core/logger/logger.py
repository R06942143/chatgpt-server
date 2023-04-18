import logging
import logging.config

from pydantic import BaseModel


class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "root"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "INFO"

    # Logging config
    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "logzioFormat": {"validate": False},
    }
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "logzio": {
            "class": "logzio.handler.LogzioHandler",
            "level": "INFO",
            "formatter": "logzioFormat",
            "logzio_type": "fastapi",
            "token": "nDhuvUGHBvDKpoUFaXXGadHcoFfNEZWw",
            "logs_drain_timeout": 5,
            "url": "https://listener-au.logz.io:8071",
        },
    }
    loggers = {
        "root": {"handlers": ["default", "logzio"], "level": LOG_LEVEL},
    }


logging.config.dictConfig(LogConfig().dict())
logger = logging.getLogger("ai_assistant")
