from __future__ import annotations

import logging
import os
from typing import Callable

import requests
from openai import OpenAI


_DOTENV_LOADED = False


class OpenAIKeyMissingError(RuntimeError):
    pass


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


def call_ollama_chat(
    *,
    prompt: str,
    url: str,
    model: str = "qwen2.5:7b",
    timeout: int = 120,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    message = body.get("message", {})
    if message:
        return message.get("content", "") or ""
    return body.get("response", "") or ""
