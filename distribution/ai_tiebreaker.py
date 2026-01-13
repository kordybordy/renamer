from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from openai import OpenAI


def _parse_json_content(content: str) -> dict[str, Any] | None:
    raw = (content or "").strip()
    if not raw:
        return None

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _openai_client() -> OpenAI | None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _call_openai(prompt: str) -> str:
    client = _openai_client()
    if not client:
        return ""
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON API. Return only JSON matching the schema. "
                    "Never include explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_ollama(prompt: str) -> str:
    host = os.environ.get("OLLAMA_HOST", "https://ollama.renamer.win/")
    url = os.environ.get("OLLAMA_URL", f"{host.rstrip('/')}/api/generate")
    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    message = body.get("message", {})
    if message:
        return message.get("content", "") or ""
    return body.get("response", "") or ""


def build_prompt(doc: dict, candidates: list[dict]) -> str:
    return (
        "Pick the best matching folder index based on opposing parties and case numbers.\n"
        "Return strict JSON in this exact shape:\n"
        "{\n"
        "  \"best_index\": 0 | 1 | 2 | null,\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"reason\": \"short\"\n"
        "}\n"
        "Rules:\n"
        "- Only choose from the provided candidates list.\n"
        "- If none are plausible, best_index must be null.\n"
        "- Keep reason short.\n\n"
        f"DOC={json.dumps(doc, ensure_ascii=False)}\n"
        f"CANDIDATES={json.dumps(candidates, ensure_ascii=False)}"
    )


def choose_best_candidate(
    *,
    doc_summary: dict,
    candidates: list[dict],
    provider: str,
) -> dict[str, Any] | None:
    prompt = build_prompt(doc_summary, candidates)
    response_text = ""
    try:
        if provider == "ollama":
            response_text = _call_ollama(prompt)
        else:
            response_text = _call_openai(prompt)
    except Exception:
        return None

    parsed = _parse_json_content(response_text)
    if not parsed:
        return None
    if "best_index" not in parsed:
        return None
    return parsed
