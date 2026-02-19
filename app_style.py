import json
import os
import re

from app_constants import GLOBAL_STYLESHEET
from app_logging import log_exception
from app_runtime import BASE_DIR


TOKEN_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def _load_theme_tokens(theme_name: str) -> dict[str, str]:
    tokens_path = os.path.join(BASE_DIR, "theme_tokens.json")
    if not os.path.exists(tokens_path):
        return {}
    try:
        with open(tokens_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tokens = payload.get(theme_name, {})
        return tokens if isinstance(tokens, dict) else {}
    except Exception as e:
        log_exception(e)
        return {}


def _compile_stylesheet(stylesheet: str, tokens: dict[str, str]) -> str:
    if not tokens:
        return stylesheet

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = tokens.get(key)
        if value is None:
            return match.group(0)
        return str(value)

    return TOKEN_PATTERN.sub(_replace, stylesheet)


def _find_unresolved_tokens(stylesheet: str) -> list[str]:
    return sorted({match.group(1) for match in TOKEN_PATTERN.finditer(stylesheet)})


def load_stylesheet() -> str:
    style_path = os.path.join(BASE_DIR, "modern.qss")
    if os.path.exists(style_path):
        try:
            with open(style_path, "r", encoding="utf-8") as f:
                stylesheet = f.read()
            if os.path.basename(style_path) == "modern.qss":
                tokens = _load_theme_tokens("modern")
                compiled_stylesheet = _compile_stylesheet(stylesheet, tokens)
                unresolved_tokens = _find_unresolved_tokens(compiled_stylesheet)
                if unresolved_tokens:
                    log_exception(
                        ValueError(
                            "Missing theme tokens for modern.qss: "
                            + ", ".join(unresolved_tokens)
                        )
                    )
                return compiled_stylesheet
            return stylesheet
        except Exception as e:
            log_exception(e)
    return GLOBAL_STYLESHEET
