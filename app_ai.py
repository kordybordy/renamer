import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI

from app_constants import BASE_SYSTEM_PROMPT, FILENAME_RULES, OLLAMA_URL
from app_logging import log_exception, log_info
from app_text_utils import clean_party_name, normalize_person_to_given_surname


API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None


def build_system_prompt(custom_elements: dict[str, str]) -> str:
    extras = ""
    if custom_elements:
        extra_lines = [f'"{name}": "string"' for name in custom_elements]
        extras = ",\n  " + ",\n  ".join(extra_lines)
        details = "\n".join(
            [f'- {name}: {desc or "Return a concise string"}' for name, desc in custom_elements.items()]
        )
        guidance = f"\nCustom fields to add (as strings):\n{details}\n"
    else:
        guidance = ""
    return BASE_SYSTEM_PROMPT.replace(
        "}",
        f'{extras}\n}}',
        1,
    ) + guidance


def call_openai_model(text: str, prompt: str) -> str:
    """Call OpenAI with fallback models, returning the raw content."""

    if client is None:
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY")

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        log_info("OpenAI gpt-5-nano failed; retrying with gpt-4.1-mini")

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content


def call_ollama_model(text: str, prompt: str) -> str:
    """Call a local Ollama model using the same system prompt."""

    try:
        payload = {
            "model": "qwen2.5:7b",
            "prompt": f"{prompt}\n\n{text}",
            "stream": False,
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        message = body.get("message", {})
        if message:
            return message.get("content", "")
        return body.get("response", "")
    except Exception as e:
        log_exception(e)
        return ""


def parse_json_content(content: str, source: str) -> dict:
    """Parse JSON content from AI responses, stripping code fences if present."""

    raw = (content or "").strip()

    def attempt_parse(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    fence_match = re.search(r"```(?:json)?\\s*(.*?)\\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    parsed = attempt_parse(raw)
    if parsed is None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = attempt_parse(raw[start : end + 1])

    if parsed is None:
        snippet = raw[:120]
        raise ValueError(
            f"{source} did not return valid JSON. Received: '{snippet or 'empty response'}'"
        )
    return parsed


def parse_ai_metadata(raw: str, custom_keys: list[str]) -> dict:
    """Convert AI JSON into the meta structure expected by the app."""

    try:
        data = parse_json_content(raw, "AI response")
    except Exception:
        return {}

    meta: dict[str, str] = {}

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

    case_numbers = data.get("case_numbers")
    if isinstance(case_numbers, list) and case_numbers:
        first_case = next((c for c in case_numbers if isinstance(c, str) and c.strip()), "")
        if first_case:
            meta["case_number"] = first_case.strip()

    for key in custom_keys:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            meta[key] = val.strip()

    return meta


def query_backend_for_meta(target: str, ocr_text: str, custom_elements: dict[str, str]) -> dict:
    prompt = build_system_prompt(custom_elements)
    raw = ""
    try:
        if target == "ollama":
            raw = call_ollama_model(ocr_text, prompt)
        else:
            raw = call_openai_model(ocr_text, prompt)
        meta = parse_ai_metadata(raw, list(custom_elements.keys()))
        if meta:
            log_info(f"AI metadata extracted using {target}")
            return meta
    except Exception as e:
        log_exception(e)
    return {}


def extract_metadata_ai_turbo(
    ocr_text: str,
    backends: list[str],
    custom_elements: dict[str, str],
    attempts_per_backend: int = 2,
) -> dict:
    workers = max(1, len(backends) * attempts_per_backend)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for target in backends:
            for _ in range(attempts_per_backend):
                future = executor.submit(query_backend_for_meta, target, ocr_text, custom_elements)
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


def extract_metadata_ai(
    ocr_text: str,
    backend: str,
    custom_elements: dict[str, str],
    turbo: bool = False,
) -> dict:
    """Use AI backend to extract metadata; returns empty dict on failure."""

    if not ocr_text.strip():
        return {}

    if turbo:
        backends = ["ollama", "openai"]
        meta = extract_metadata_ai_turbo(ocr_text, backends, custom_elements)
        if meta:
            return meta
    else:
        backends = [backend]
        if backend == "auto":
            backends = ["ollama", "openai"]

    for target in backends:
        meta = query_backend_for_meta(target, ocr_text, custom_elements)
        if meta:
            return meta

    return {}
