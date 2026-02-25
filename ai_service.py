from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

import requests
from openai import OpenAI

try:
    from websocket import WebSocketConnectionClosedException, create_connection
except Exception:  # pragma: no cover - optional dependency for websocket acceleration
    WebSocketConnectionClosedException = RuntimeError
    create_connection = None


_DOTENV_LOADED = False
_OLLAMA_REMOTE_FALLBACK_GENERATE_URL = (
    "https://contribute-roommates-gave-fame.trycloudflare.com/api/generate"
)


class OpenAIKeyMissingError(RuntimeError):
    pass


@dataclass
class AICallResult:
    provider: str
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    prompt_id: str = "default"
    prompt_version: str = "v1"
    request_id: str = ""


_OPENAI_MODEL_COST_USD_PER_1K = {
    "gpt-5-nano": {"prompt": 0.00005, "completion": 0.0004},
    "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
}


def estimate_text_tokens(text: str) -> int:
    # Lightweight fallback estimator when provider usage is unavailable.
    return max(1, len((text or "").strip()) // 4) if (text or "").strip() else 0


def _estimate_openai_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _OPENAI_MODEL_COST_USD_PER_1K.get(model)
    if not pricing:
        return 0.0
    prompt_cost = (prompt_tokens / 1000.0) * pricing["prompt"]
    completion_cost = (completion_tokens / 1000.0) * pricing["completion"]
    return round(prompt_cost + completion_cost, 8)


def append_ai_telemetry(record: dict, file_path: str | None = None) -> None:
    target = (file_path or os.environ.get("AI_TELEMETRY_PATH") or "ai_telemetry.jsonl").strip()
    if not target:
        return
    path = Path(target)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("created_at", datetime.now(UTC).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_telemetry_totals(
    *,
    tenant_id: str,
    date_yyyymmdd: str,
    file_path: str | None = None,
) -> dict[str, float]:
    target = (file_path or os.environ.get("AI_TELEMETRY_PATH") or "ai_telemetry.jsonl").strip()
    if not target:
        return {"tokens": 0.0, "cost_usd": 0.0, "requests": 0.0}
    path = Path(target)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path
    if not path.exists():
        return {"tokens": 0.0, "cost_usd": 0.0, "requests": 0.0}

    totals = {"tokens": 0.0, "cost_usd": 0.0, "requests": 0.0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if item.get("tenant_id") != tenant_id or item.get("date") != date_yyyymmdd:
                continue
            totals["tokens"] += float(item.get("total_tokens") or 0)
            totals["cost_usd"] += float(item.get("estimated_cost_usd") or 0)
            totals["requests"] += 1.0
    return totals


class _OpenAIResponsesWebSocketSession:
    """Thread-safe helper for persistent Responses API websocket calls."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.socket = None
        self.lock = threading.Lock()
        self.previous_response_id: str | None = None

    def _connect(self):
        if create_connection is None:
            raise RuntimeError("websocket-client is not installed")
        if self.socket is not None:
            return
        self.socket = create_connection(
            "wss://api.openai.com/v1/responses",
            header=[f"Authorization: Bearer {self.api_key}"],
            timeout=120,
        )

    def close(self):
        if self.socket is None:
            return
        try:
            self.socket.close()
        except Exception:
            pass
        self.socket = None
        self.previous_response_id = None

    @staticmethod
    def _extract_response_text(response: dict) -> str:
        output = response.get("output", []) if isinstance(response, dict) else []
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    text = part.get("text", "")
                    if text:
                        chunks.append(text)
        return "".join(chunks).strip()

    def _receive_response(self) -> tuple[str, str | None]:
        assert self.socket is not None
        output_chunks: list[str] = []
        while True:
            raw_message = self.socket.recv()
            if not raw_message:
                continue
            event = json.loads(raw_message)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    output_chunks.append(delta)
                continue
            if event_type == "response.completed":
                response = event.get("response", {})
                response_id = response.get("id") if isinstance(response, dict) else None
                text = "".join(output_chunks).strip()
                if not text:
                    text = self._extract_response_text(response)
                return text, response_id
            if event_type == "error":
                error = event.get("error", {}) if isinstance(event, dict) else {}
                message = (
                    error.get("message")
                    or event.get("message")
                    or "Unknown websocket responses API error"
                )
                code = error.get("code") if isinstance(error, dict) else None
                raise RuntimeError(f"{code or 'responses_error'}: {message}")

    def create_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
    ) -> str:
        payload: dict = {
            "type": "response.create",
            "model": model,
            "store": False,
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id

        with self.lock:
            self._connect()
            assert self.socket is not None
            try:
                self.socket.send(json.dumps(payload))
                text, response_id = self._receive_response()
            except WebSocketConnectionClosedException:
                self.close()
                self._connect()
                assert self.socket is not None
                payload.pop("previous_response_id", None)
                self.socket.send(json.dumps(payload))
                text, response_id = self._receive_response()
            except Exception as exc:
                if "previous_response_not_found" in str(exc):
                    self.previous_response_id = None
                    payload.pop("previous_response_id", None)
                    self.socket.send(json.dumps(payload))
                    text, response_id = self._receive_response()
                elif "websocket_connection_limit_reached" in str(exc):
                    self.close()
                    self._connect()
                    payload.pop("previous_response_id", None)
                    self.socket.send(json.dumps(payload))
                    text, response_id = self._receive_response()
                else:
                    raise

            self.previous_response_id = response_id
            return text


_OPENAI_WS_SESSION: _OpenAIResponsesWebSocketSession | None = None
_OPENAI_WS_LOCK = threading.Lock()


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
    max_tokens: int | None = None,
) -> AICallResult:
    key = _resolve_openai_key()
    client = OpenAI(api_key=key)
    payload = {
        "model": model,
        "messages": _build_openai_messages(system_prompt, user_prompt),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**payload)
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = estimate_text_tokens(system_prompt + "\n" + user_prompt)
    if completion_tokens is None:
        completion_tokens = estimate_text_tokens(content)
    total_tokens = prompt_tokens + completion_tokens
    request_id = getattr(resp, "id", "") or ""
    return AICallResult(
        provider="openai",
        model=model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=_estimate_openai_cost(model, prompt_tokens, completion_tokens),
        request_id=request_id,
    )


def _websocket_mode_enabled() -> bool:
    return os.environ.get("OPENAI_WEBSOCKET_MODE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _get_ws_session() -> _OpenAIResponsesWebSocketSession:
    global _OPENAI_WS_SESSION
    if _OPENAI_WS_SESSION is not None:
        return _OPENAI_WS_SESSION
    with _OPENAI_WS_LOCK:
        if _OPENAI_WS_SESSION is None:
            _OPENAI_WS_SESSION = _OpenAIResponsesWebSocketSession(_resolve_openai_key())
    return _OPENAI_WS_SESSION


def _call_openai_via_websocket(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float | None,
) -> AICallResult:
    session = _get_ws_session()
    content = session.create_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
    )
    prompt_tokens = estimate_text_tokens(system_prompt + "\n" + user_prompt)
    completion_tokens = estimate_text_tokens(content)
    return AICallResult(
        provider="openai",
        model=model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=_estimate_openai_cost(model, prompt_tokens, completion_tokens),
    )


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
    return call_openai_chat_result(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        fallback_model=fallback_model,
        temperature=temperature,
        fallback_temperature=fallback_temperature,
        log_info=log_info,
    ).content


def call_openai_chat_result(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5-nano",
    fallback_model: str = "gpt-4.1-mini",
    temperature: float | None = None,
    fallback_temperature: float | None = 0.0,
    max_tokens: int | None = None,
    prompt_id: str = "default",
    prompt_version: str = "v1",
    log_info: Callable[[str], None] | None = None,
) -> AICallResult:
    try:
        if _websocket_mode_enabled() and create_connection is not None:
            result = _call_openai_via_websocket(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
            )
            result.prompt_id = prompt_id
            result.prompt_version = prompt_version
            return result
        result = _call_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result.prompt_id = prompt_id
        result.prompt_version = prompt_version
        return result
    except OpenAIKeyMissingError:
        raise
    except Exception as exc:
        if log_info:
            log_info(f"OpenAI {model} failed; retrying with {fallback_model} ({exc})")
    result = _call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=fallback_model,
        temperature=fallback_temperature,
        max_tokens=max_tokens,
    )
    result.prompt_id = prompt_id
    result.prompt_version = prompt_version
    return result


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
    return call_ollama_chat_result(
        prompt=prompt,
        url=url,
        model=model,
        timeout=timeout,
    ).content


def call_ollama_chat_result(
    *,
    prompt: str,
    url: str | None = None,
    model: str = "qwen2.5:7b",
    timeout: int = 120,
    prompt_id: str = "default",
    prompt_version: str = "v1",
) -> AICallResult:
    endpoint = url or get_ollama_generate_url()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        parsed_endpoint = endpoint.lower()
        local_endpoints = (
            "127.0.0.1",
            "localhost",
            "0.0.0.0",
        )
        is_local_request = any(token in parsed_endpoint for token in local_endpoints)
        fallback_endpoint = os.environ.get(
            "OLLAMA_REMOTE_FALLBACK_GENERATE_URL",
            _OLLAMA_REMOTE_FALLBACK_GENERATE_URL,
        ).strip()
        if not is_local_request or not fallback_endpoint:
            raise
        resp = requests.post(fallback_endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
    body = resp.json()
    message = body.get("message", {})
    if message:
        content = message.get("content", "") or ""
    else:
        content = body.get("response", "") or ""
    prompt_tokens = estimate_text_tokens(prompt)
    completion_tokens = estimate_text_tokens(content)
    return AICallResult(
        provider="ollama",
        model=model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=0.0,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )
