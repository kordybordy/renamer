import os

from app_constants import GLOBAL_STYLESHEET
from app_logging import log_exception
from app_runtime import BASE_DIR


def load_stylesheet() -> str:
    style_path = os.path.join(BASE_DIR, "modern.qss")
    if os.path.exists(style_path):
        try:
            with open(style_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            log_exception(e)
    return GLOBAL_STYLESHEET
