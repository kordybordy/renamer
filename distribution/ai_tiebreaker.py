from __future__ import annotations

import json
import logging
import re
from typing import Any

from ai_service import OpenAIKeyMissingError, call_ollama_chat, call_openai_chat


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


def _call_openai(prompt: str) -> str:
    return call_openai_chat(
        system_prompt=(
            "You are a strict JSON API. Return only JSON matching the schema. "
            "Never include explanations."
        ),
        user_prompt=prompt,
        model="gpt-5-nano",
        fallback_model="gpt-4.1-mini",
        temperature=0.0,
        fallback_temperature=0.0,
        log_info=lambda message: logging.getLogger(__name__).info("[AI] %s", message),
    )


def _call_ollama(prompt: str) -> str:
    return call_ollama_chat(
        prompt=prompt,
        model="qwen2.5:7b",
        timeout=120,
    )


def build_prompt(doc: dict, candidates: list[dict]) -> str:
    return (
        "Pick the best matching folder based on opposing parties and case numbers.\n"
        "Return strict JSON in this exact shape:\n"
        "{\n"
        "  \"chosen_folder\": \"<folder_name from candidates>\" | null,\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"reason\": \"short\"\n"
        "}\n"
        "Rules:\n"
        "- Only choose from the provided candidates list.\n"
        "- If none are plausible, chosen_folder must be null.\n"
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
    except OpenAIKeyMissingError as exc:
        logging.getLogger(__name__).warning("AI tiebreaker skipped: %s", exc)
        return None
    except Exception as exc:
        logging.getLogger(__name__).exception(
            "AI tiebreaker failed (provider=%s): %s", provider, exc
        )
        return None

    parsed = _parse_json_content(response_text)
    if not parsed:
        return None
    if "chosen_folder" not in parsed:
        return None
    return parsed
