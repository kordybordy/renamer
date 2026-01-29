from __future__ import annotations

import os
from typing import Callable

import requests
from openai import OpenAI


def _resolve_openai_key(api_key: str | None) -> str:
    return api_key or os.environ.get("OPENAI_API_KEY", "")


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
    api_key: str | None,
) -> str:
    key = _resolve_openai_key(api_key)
    if not key:
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY")
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
    api_key: str | None = None,
    log_info: Callable[[str], None] | None = None,
) -> str:
    try:
        return _call_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
    except Exception:
        if log_info:
            log_info(f"OpenAI {model} failed; retrying with {fallback_model}")
    return _call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=fallback_model,
        temperature=fallback_temperature,
        api_key=api_key,
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
