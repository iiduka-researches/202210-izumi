import json
import sys
from functools import wraps
from textwrap import wrap
from traceback import format_exception
from typing import Callable
from requests import post

MAX_CHAR = 500


def notify(message: str) -> None:
    with open('./utils/line/token.json') as f:
        s = '\n'.join(f.readlines())
        d = json.loads(s)
    headers = dict(Authorization=('Bearer ' + d['token']))

    for m in wrap(message, width=MAX_CHAR):
        params = dict(message=m)
        post(url=d['url'], headers=headers, params=params)


def notify_error(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> object:
        try:
            result = func(*args, **kwargs)
            return result
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            notify(''.join((format_exception(exc_type, exc_value, exc_traceback))))
            raise Exception
    return wrapper
