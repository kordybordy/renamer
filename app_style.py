import os

from app_constants import GLOBAL_STYLESHEET
from app_logging import log_exception
from app_runtime import BASE_DIR


def load_stylesheet() -> str:
    for filename in ("modern.qss", "retro.qss"):
        style_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(style_path):
            continue
        try:
            with open(style_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            log_exception(e)
    return GLOBAL_STYLESHEET
