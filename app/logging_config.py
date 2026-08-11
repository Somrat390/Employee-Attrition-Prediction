"""Structured (JSON) logging for the API.

Logs go to stdout as one JSON object per line -- the standard approach for
containerized services (Docker/Kubernetes/Render/Railway all capture stdout
automatically and expect to feed it to a log aggregator), rather than
writing to a file inside the container that nothing outside it can see.

Three loggers are used elsewhere in the app, all going through this same
JSON formatter:
    "app.requests"    -- one line per HTTP request (method, path, status, duration)
    "app.predictions" -- one line per prediction (result, probability, inference time)
    "app.errors"      -- exceptions, with the traceback included
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Fields every LogRecord has by default -- anything else attached via
# logger.info(..., extra={...}) is "ours" and gets included in the JSON.
_STANDARD_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging():
    """Call once, at import time of app.main. Idempotent -- safe if uvicorn's
    --reload triggers a re-import."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn's own access log duplicates what our request-logging middleware
    # already logs (in plain text, not JSON) -- quiet it down so log output
    # stays consistently structured. uvicorn's error log (startup messages,
    # crashes) is left alone since that's genuinely separate information.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
