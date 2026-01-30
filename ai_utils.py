import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from ai_service import OpenAIKeyMissingError, call_ollama_chat, call_openai_chat
from config import SYSTEM_PROMPT, FILENAME_RULES
from logging_utils import log_exception, log_info
from text_utils import apply_meta_defaults, apply_party_order, build_filename, clean_party_name, normalize_person_to_given_surname


def call_openai_model(text: str) -> str:
    return call_openai_chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=text,
        model="gpt-5-nano",
        fallback_model="gpt-4.1-mini",
        temperature=None,
        fallback_temperature=0.0,
        log_info=lambda message: log_info(f"[AI] {message}"),
    )


def call_ollama_model(text: str) -> str:
    try:
        return call_ollama_chat(
            prompt=f"{SYSTEM_PROMPT}\n\n{text}",
            model="qwen2.5:7b",
            timeout=120,
        )
    except Exception as e:
        log_exception(e)
        return ""


def extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")
    snippet = raw[start : end + 1]
    return json.loads(snippet)


def parse_json_content(content: str, source: str) -> dict:
    raw = (content or "").strip()

    def attempt_parse(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    parsed = attempt_parse(raw)
    if parsed is None:
        try:
            parsed = extract_json_object(raw)
        except Exception:
            parsed = None

    if parsed is None:
        snippet = raw[:120]
        snippet_display = snippet or "empty response"
        raise ValueError(
            f"{source} did not return valid JSON. Received: '{snippet_display}'"
        )
    return parsed


def parse_ai_metadata(raw: str) -> dict:
    try:
        data = parse_json_content(raw, "AI response")
    except Exception as e:
        log_info(f"[AI] JSON parse failed: {e}. Raw response: {raw[:500]}")
        return {}

    meta: Dict[str, str] = {}

    def prepare_party(key: str):
        values = data.get(key)
        if not isinstance(values, list):
            return
        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            name = clean_party_name(value)
            name = normalize_person_to_given_surname(name) or name
            if not name:
                continue
            lower_name = name.lower()
            if "dwf poland jamka" in lower_name:
                continue
            if FILENAME_RULES.get("remove_raiffeisen") and "raiffeisen" in lower_name:
                continue
            cleaned.append(name)
        if FILENAME_RULES.get("primary_party_only"):
            cleaned = cleaned[:1]
        max_items = FILENAME_RULES.get("max_parties", len(cleaned))
        if cleaned:
            meta[key] = ", ".join(cleaned[:max_items])

    prepare_party("plaintiff")
    prepare_party("defendant")

    lt = data.get("letter_type")
    if isinstance(lt, str) and lt.strip():
        meta["letter_type"] = lt.strip()

    return meta


def query_backend_for_meta(target: str, ocr_text: str) -> dict:
    raw = ""
    try:
        if target == "ollama":
            raw = call_ollama_model(ocr_text)
        else:
            raw = call_openai_model(ocr_text)
        meta = parse_ai_metadata(raw)
        if meta:
            log_info(f"AI metadata extracted using {target}")
            return meta
    except OpenAIKeyMissingError as exc:
        log_info(str(exc))
        return {}
    except Exception as e:
        log_exception(e)
    return {}


def extract_metadata_ai_turbo(ocr_text: str, backends: list[str], attempts_per_backend: int = 2) -> dict:
    workers = max(1, len(backends) * attempts_per_backend)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for target in backends:
            for _ in range(attempts_per_backend):
                future = executor.submit(query_backend_for_meta, target, ocr_text)
                future_map[future] = target
        for fut in as_completed(future_map):
            try:
                meta = fut.result()
            except Exception as e:
                log_exception(e)
                continue
            if meta:
                for other in future_map:
                    if other is not fut:
                        other.cancel()
                return meta
    return {}


def extract_metadata_ai(ocr_text: str, backend: str, turbo: bool = False) -> dict:
    if not ocr_text.strip():
        return {}

    if turbo:
        backends = ["ollama", "openai"]
        meta = extract_metadata_ai_turbo(ocr_text, backends)
        if meta:
            return meta
    else:
        backends = [backend]
        if backend == "auto":
            backends = ["ollama", "openai"]

    for target in backends:
        meta = query_backend_for_meta(target, ocr_text)
        if meta:
            return meta

    return {}
