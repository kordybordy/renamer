from __future__ import annotations

import logging
import os
from typing import Callable
from urllib.parse import urljoin

import requests
from openai import OpenAI


_DOTENV_LOADED = False


class OpenAIKeyMissingError(RuntimeError):
    pass


def _running_in_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


def _load_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def _resolve_openai_key() -> str:
    if "OPENAI_API_KEY" not in os.environ:
        if not _running_in_github_actions():
            _load_dotenv()
    try:
        return os.environ["OPENAI_API_KEY"]
    except KeyError as exc:
        logging.getLogger(__name__).warning(
            "OpenAI API key missing. Set OPENAI_API_KEY in your environment to enable OpenAI requests."
        )
        raise OpenAIKeyMissingError(
            "OpenAI API key not configured. Set the OPENAI_API_KEY environment variable."
        ) from exc


def _build_openai_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_openai(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float | None,
) -> str:
    key = _resolve_openai_key()
    client = OpenAI(api_key=key)
    payload = {
        "model": model,
        "messages": _build_openai_messages(system_prompt, user_prompt),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    resp = client.chat.completions.create(**payload)
    return resp.choices[0].message.content or ""


def call_openai_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5-nano",
    fallback_model: str = "gpt-4.1-mini",
    temperature: float | None = None,
    fallback_temperature: float | None = 0.0,
    log_info: Callable[[str], None] | None = None,
) -> str:
    try:
        return _call_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )
    except OpenAIKeyMissingError:
        raise
    except Exception:
        if log_info:
            log_info(f"OpenAI {model} failed; retrying with {fallback_model}")
    return _call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=fallback_model,
        temperature=fallback_temperature,
    )


def get_ollama_base_url() -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    if not base_url:
        return "http://127.0.0.1:11434"
    return base_url.rstrip("/")


def _ollama_url(path: str) -> str:
    return urljoin(f"{get_ollama_base_url()}/", path.lstrip("/"))


def get_ollama_generate_url() -> str:
    return _ollama_url("api/generate")


def build_ollama_generate_url(base_url: str | None) -> str:
    if not base_url:
        return get_ollama_generate_url()
    trimmed = base_url.strip().rstrip("/")
    if not trimmed:
        return get_ollama_generate_url()
    return urljoin(f"{trimmed}/", "api/generate")


def get_ollama_tags_url() -> str:
    return _ollama_url("api/tags")


def check_ollama_health(timeout: int = 2) -> bool:
    try:
        resp = requests.get(get_ollama_tags_url(), timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def call_ollama_chat(
    *,
    prompt: str,
    url: str | None = None,
    model: str = "qwen2.5:7b",
    timeout: int = 120,
) -> str:
    endpoint = url or get_ollama_generate_url()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    message = body.get("message", {})
    if message:
        return message.get("content", "") or ""
    return body.get("response", "") or ""
