import os
import re
from datetime import datetime

from app_constants import FILENAME_RULES


def normalize_polish(text: str) -> str:
    mapping = str.maketrans({
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ż": "z",
        "ź": "z",
    })
    normalized = (text or "").lower()
    normalized = re.sub(r"[\\-_,.;:()\\[\\]{}<>!?/\\\\]+", " ", normalized)
    normalized = re.sub(r"\\s+", " ", normalized).strip()
    normalized = normalized.translate(mapping)
    return normalized


def normalize_person_to_given_surname(s: str) -> str:
    """
    Returns 'Given Surname' (exactly 2 tokens), dropping middle names.
    Handles common OCR noise and hyphenated surnames.
    """
    if not s:
        return ""

    s = re.sub(r"\\s+", " ", s.strip())
    s = s.strip(" ,.;:")

    parts = s.split(" ")
    if len(parts) == 1:
        return s

    first = parts[0]
    rest = parts[1:]
    is_all_caps = (first.upper() == first) and any(ch.isalpha() for ch in first)

    if is_all_caps and len(parts) >= 2:
        surname = first.title()
        given = rest[0].title()
        return f"{given} {surname}"

    if len(parts) == 2:
        likely_surname_first_suffixes = (
            "ski",
            "ska",
            "cki",
            "cka",
            "dzki",
            "dzka",
            "wicz",
            "owicz",
            "ewicz",
            "icz",
            "czyk",
            "czak",
            "czuk",
            "uk",
            "ak",
            "ek",
            "arz",
            "asz",
            "ysz",
            "ów",
            "owa",
            "ewna",
        )
        first_lower = first.lower()
        if first_lower.endswith(likely_surname_first_suffixes):
            surname = first.title()
            given = rest[0].title()
            return f"{given} {surname}"

    given = parts[0].title()
    surname = parts[-1].title()
    return f"{given} {surname}"


def clean_party_name(raw: str) -> str:
    name = raw.strip().strip("-:;•")
    name = re.sub(r"^[\\d.\\)]+\\s*", "", name)

    address_markers = [
        r"\\b\\d{2}-\\d{3}\\b",
        r"\\bul\\.?\\b",
        r"\\bal\\.?\\b",
        r"\\bpl\\.?\\b",
        r"\\bplac\\b",
        r"\\baleja\\b",
    ]
    for marker in address_markers:
        match = re.search(marker, name, flags=re.IGNORECASE)
        if match:
            name = name[: match.start()].rstrip(",; -")
            break
    name = re.sub(r"\\(\\s*z domu[^)]*\\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(?i)\\bz domu\\b.*", "", name)
    name = name.replace("(", "").replace(")", "")
    name = re.sub(r"(?i)\\bpesel[:\\s]*\\d[\\d\\s]{9,}\\d\\b", "", name)
    name = re.sub(r"\\b\\d{11}\\b", "", name)
    name = re.sub(r"\\s{2,}", " ", name)
    name = name.strip()
    max_tokens = FILENAME_RULES.get("person_token_limit")
    if max_tokens and max_tokens > 0:
        tokens = name.split()
        if len(tokens) > max_tokens:
            if max_tokens == 2 and len(tokens) >= 2:
                name = " ".join([tokens[0], tokens[-1]])
            else:
                name = " ".join(tokens[:max_tokens])
    return name[:80]


def format_party_name(name: str, surname_first: bool) -> str:
    normalized = normalize_person_to_given_surname(name) or name
    tokens = [tok for tok in normalized.split() if tok]
    if len(tokens) != 2:
        return normalized

    joined = " ".join(tokens)
    if re.search(r"[\\d/]", joined) or "." in joined:
        return normalized

    given, surname = tokens
    if surname_first:
        return f"{surname} {given}".strip()
    return f"{given} {surname}".strip()


def normalize_target_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\\\/:*?\"<>|]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def requirements_from_template(template: list[str], custom_elements: dict[str, str] | None = None) -> dict:
    requirements = {
        "plaintiff": True if "plaintiff" in template else False,
        "defendant": True if "defendant" in template else False,
        "letter_type": True if "letter_type" in template else False,
        "date": True if "date" in template else False,
        "case_number": True if "case_number" in template else False,
    }
    if custom_elements:
        for key in custom_elements:
            requirements[key] = True
    return requirements


def apply_meta_defaults(meta: dict, requirements: dict) -> dict:
    meta = meta.copy()
    if requirements.get("date") and "date" not in meta:
        meta["date"] = datetime.now().strftime("%Y-%m-%d")
    if requirements.get("letter_type"):
        meta.setdefault("letter_type", FILENAME_RULES.get("default_letter_type", "letter"))
    if requirements.get("plaintiff"):
        meta.setdefault("plaintiff", "Plaintiff")
    if requirements.get("defendant"):
        meta.setdefault("defendant", "Defendant")
    if requirements.get("case_number"):
        meta.setdefault("case_number", "Case-Number")
    for key in requirements:
        if key not in ("plaintiff", "defendant", "letter_type", "date", "case_number"):
            meta.setdefault(key, key.replace("_", " ").title())
    return meta


def format_party_field(value, surname_first: bool) -> str:
    names: list[str] = []
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        names = [str(item).strip() for item in value if str(item).strip()]
    if not names:
        return value if isinstance(value, str) else ""

    formatted = [format_party_name(name, surname_first) for name in names]
    return ", ".join(formatted)


def defendant_from_filename(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[_\\-]+", " ", base)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    return cleaned


def apply_party_order(meta: dict, *, plaintiff_surname_first: bool, defendant_surname_first: bool) -> dict:
    meta = meta.copy()
    if "plaintiff" in meta:
        meta["plaintiff"] = format_party_field(
            meta.get("plaintiff"), plaintiff_surname_first
        )
    if "defendant" in meta:
        meta["defendant"] = format_party_field(
            meta.get("defendant"), defendant_surname_first
        )
    return meta


def build_filename(meta: dict, template_elements: list[str]) -> str:
    parts: list[str] = []
    for element in template_elements:
        value = meta.get(element, "")
        if not value and element == "date":
            value = datetime.now().strftime("%Y-%m-%d")
        if value:
            parts.append(str(value).strip())
    filename = " - ".join(parts)
    if not filename:
        filename = "document"
    filename = normalize_target_filename(filename)
    return filename
