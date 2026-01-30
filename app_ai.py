import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai_service import OpenAIKeyMissingError, call_ollama_chat, call_openai_chat
from app_constants import BASE_SYSTEM_PROMPT, FILENAME_RULES, OLLAMA_URL
from app_logging import log_exception, log_info
from app_text_utils import clean_party_name, normalize_person_to_given_surname


def build_system_prompt(custom_elements: dict[str, str]) -> str:
    extras = ""
    if custom_elements:
        extra_lines = [f'    "{name}": "string"' for name in custom_elements]
        extras = '"custom": {\n' + ",\n".join(extra_lines) + "\n  }"
        details = "\n".join(
            [f'- {name}: {desc or "Return a concise string"}' for name, desc in custom_elements.items()]
        )
        guidance = f"\nCustom fields to add under \"custom\" (as strings):\n{details}\n"
    else:
        guidance = ""
    prompt = BASE_SYSTEM_PROMPT
    if extras:
        prompt = prompt.replace('"custom": {}', extras, 1)
    return prompt + guidance


def call_openai_model(text: str, prompt: str) -> str:
    """Call OpenAI with fallback models, returning the raw content."""

    return call_openai_chat(
        system_prompt=prompt,
        user_prompt=text,
        model="gpt-5-nano",
        fallback_model="gpt-4.1-mini",
        temperature=None,
        fallback_temperature=0.0,
        log_info=lambda message: log_info(f"[AI] {message}"),
    )


def call_ollama_model(text: str, prompt: str) -> str:
    """Call a local Ollama model using the same system prompt."""

    try:
        return call_ollama_chat(
            prompt=f"{prompt}\n\n{text}",
            url=OLLAMA_URL,
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
        try:
            parsed = extract_json_object(raw)
        except Exception:
            parsed = None

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
    except Exception as e:
        log_info(f"[AI] JSON parse failed: {e}. Raw response: {raw[:500]}")
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

    custom_payload = data.get("custom", {}) if isinstance(data.get("custom"), dict) else {}
    for key in custom_keys:
        val = custom_payload.get(key)
        if not val:
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
            log_info(f"[AI] metadata extracted using {target}")
            return meta
    except OpenAIKeyMissingError as exc:
        log_info(f"[AI] {exc}")
        return {}
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
