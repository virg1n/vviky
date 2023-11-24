import functools
import json
import logging
from contextlib import contextmanager
from http import HTTPStatus
from typing import Optional

from requests.exceptions import ConnectionError, ConnectTimeout, HTTPError, Timeout

logger = logging.getLogger(__name__)


@contextmanager
def error_handling_context(error_prefix: str):
    def log_error(msg: str):
        logger.error(f"{error_prefix} {msg}")

    try:
        yield
    except HTTPError as err:
        if err.response.status_code == HTTPStatus.SERVICE_UNAVAILABLE:
            json_msg = _get_json(err.response.text)
            if json_msg is not None and isinstance(json_msg, dict) and "error" in json_msg:
                log_error(json_msg["error"])
            else:
                log_error(err.response.text)
        else:
            raise err
    except (ConnectionError, ConnectTimeout):
        log_error(
            "Unable to communicate with the backend server."
            " Please verify that you have internet connectivity."
        )
    except Timeout:
        log_error("Request timed out")


def error_handling(error_prefix: str):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with error_handling_context(error_prefix):
                return func(*args, **kwargs)

        return wrapper

    return inner


def _get_json(msg: str) -> Optional[dict]:
    try:
        return json.loads(msg)
    except ValueError:
        return None
