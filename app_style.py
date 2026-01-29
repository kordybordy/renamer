import os

from app_constants import GLOBAL_STYLESHEET
from app_logging import log_exception
from app_runtime import BASE_DIR


def load_stylesheet() -> str:
    retro_path = os.path.join(BASE_DIR, "retro.qss")
    if os.path.exists(retro_path):
        try:
            with open(retro_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            log_exception(e)
    return GLOBAL_STYLESHEET
