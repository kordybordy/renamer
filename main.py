
# --------- HARD-CODED API KEY (EDIT THIS LINE!) ----------
API_KEY = "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"
# ----------------------------------------------------------

import sys
import os
import re
import json
import shutil
import subprocess
import tempfile
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from PIL import Image
import pytesseract

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox,
    QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QListWidget, QListWidgetItem, QTextEdit, QProgressBar,
    QStatusBar, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt6.QtGui import QPixmap, QIcon

AI_BACKEND = os.environ.get("AI_BACKEND", "openai")  # openai | ollama | auto
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.renamer.win/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", urljoin(OLLAMA_HOST, "api/generate"))

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

# ===============================
# FILENAME POLICY
# ===============================

FILENAME_RULES = {
    "remove_raiffeisen": True,        # always remove Raiffeisen from parties
    "max_parties": 10,                # limit number of names in filename
    "primary_party_only": False,      # include all detected parties per side
    "surname_first": True,            # SURNAME Name
    "use_commas": True,               # comma-separated parties
    "replace_slash_only": True,       # only replace "/" → "_"
    "force_letter_type": True,        # GUI overrides AI letter type
    "default_letter_type": "pozew",   # fallback if AI unsure
}


DEFAULT_TEMPLATE_ELEMENTS = ["date", "plaintiff", "defendant", "letter_type"]


# ===============================
# Logging
# ===============================
import traceback
from datetime import datetime

LOG_FILE = os.path.join(os.path.expanduser("~"), "renamer_error.log")
DISTRIBUTION_LOG_FILE = os.path.join(os.path.expanduser("~"), "renamer_distribution.log")

ACCENT_COLOR = "#4F7CFF"
BACKGROUND_COLOR = "#1E1E1E"
PANEL_COLOR = "#252526"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#B0B0B0"
BORDER_COLOR = "#333333"

GLOBAL_STYLESHEET = f"""
* {{
    font-family: 'Segoe UI', sans-serif;
    color: {TEXT_PRIMARY};
}}

QWidget {{
    background-color: {BACKGROUND_COLOR};
}}

QLineEdit, QComboBox, QListWidget, QTableWidget, QTextEdit, QSpinBox {{
    background-color: {PANEL_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 6px;
    color: {TEXT_PRIMARY};
}}

QLabel {{
    color: {TEXT_PRIMARY};
}}

QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    background: {PANEL_COLOR};
    border-radius: 10px;
    padding: 6px;
}}

QTabBar::tab {{
    background: {PANEL_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-bottom: none;
    padding: 8px 16px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 4px;
}}

QTabBar::tab:selected {{
    background: {ACCENT_COLOR};
    color: {TEXT_PRIMARY};
}}

QTabBar::tab:hover {{
    border-color: {ACCENT_COLOR};
}}

QPushButton {{
    background-color: {ACCENT_COLOR};
    border: 1px solid {ACCENT_COLOR};
    color: {TEXT_PRIMARY};
    padding: 10px 14px;
    border-radius: 8px;
    font-weight: 600;
}}

QPushButton:hover {{
    box-shadow: 0 0 8px {ACCENT_COLOR};
}}

QPushButton:disabled {{
    background-color: {BORDER_COLOR};
    border-color: {BORDER_COLOR};
    color: {TEXT_SECONDARY};
}}

QProgressBar {{
    background: {PANEL_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_COLOR};
    border-radius: 6px;
}}
"""

def log_exception(e: Exception):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(datetime.now().isoformat() + "\n")
        f.write(str(e) + "\n")
        f.write(traceback.format_exc())
        f.flush()
        os.fsync(f.fileno())


def log_info(message: str):
    entry = f"{datetime.now().isoformat()} INFO {message}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(entry)


def append_distribution_log(entry: str):
    with open(DISTRIBUTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
        f.flush()
        os.fsync(f.fileno())


def show_friendly_error(
    parent: QWidget,
    title: str,
    friendly: str,
    details: str,
    *,
    icon: QMessageBox.Icon = QMessageBox.Icon.Critical,
):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(friendly)
    if details:
        box.setInformativeText("Show technical details below if you need them.")
        box.setDetailedText(details)
    box.setIcon(icon)
    box.exec()


# ===============================
# Normalization helpers
# ===============================

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
    normalized = normalized.translate(mapping)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

# ===============================
# BASE DIR
# ===============================
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# POPPLER (GLOBAL!)
# ===============================
POPPLER_PATH = os.path.join(BASE_DIR, "poppler", "Library", "bin")
PDFTOPPM_EXE = os.path.join(POPPLER_PATH, "pdftoppm.exe")
os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

if not os.path.exists(PDFTOPPM_EXE):
    raise RuntimeError(f"pdftoppm.exe not found: {PDFTOPPM_EXE}")

# ===============================
# Tesseract configuration (CRITICAL)
# ===============================

def configure_tesseract() -> str:
    import sys, os, pytesseract

    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    tesseract_dir = os.path.join(base_path, "tesseract")
    tessdata_dir = os.path.join(tesseract_dir, "tessdata")
    tesseract_exe = os.path.join(tesseract_dir, "tesseract.exe")

    if not os.path.exists(tesseract_exe):
        raise RuntimeError(f"Tesseract EXE not found: {tesseract_exe}")

    if not os.path.exists(tessdata_dir):
        raise RuntimeError(f"Tessdata folder not found: {tessdata_dir}")

    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    os.environ["TESSDATA_PREFIX"] = tessdata_dir

    return tessdata_dir


TESSDATA_DIR = configure_tesseract()

# ==========================================================
# AI SYSTEM PROMPT — returns structured JSON
# ==========================================================

SYSTEM_PROMPT = """
Return strict JSON in this exact shape:

{
  "plaintiff": ["Given Surname", ...],
  "defendant": ["Given Surname", ...],
  "case_numbers": ["I C 1234/25", ...],
  "letter_type": "Pozew" | "Pozew + Postanowienie" |
                 "Postanowienie" | "Portal" | "Korespondencja" |
                 "Unknown" | "Zawiadomienie" |
                 "Odpowiedź na pozew" | "Wniosek" | "Replika"
}

Rules:
- Ignore DWF Poland Jamka and Raiffeisen Bank (do not include them in any party list).
- Each list item MUST be EXACTLY TWO WORDS: "Given Surname".
  If the person has multiple given names, KEEP ONLY THE FIRST given name.
  Examples:
    "Szymon Hubert Marciniak" -> "Szymon Marciniak"
    "Katarzyna Magdalena Obałek" -> "Katarzyna Obałek"
- Never include PESEL, addresses, or IDs.
- Extract ALL case numbers.
- Preserve Polish letters.
- No commentary. Output JSON only.
"""


@dataclass
class NamingOptions:
    template_elements: list[str]
    ocr_enabled: bool
    ocr_char_limit: int
    ocr_dpi: int
    ocr_pages: int
    plaintiff_surname_first: bool
    defendant_surname_first: bool
    turbo_mode: bool


def normalize_person_to_given_surname(s: str) -> str:
    """
    Returns 'Given Surname' (exactly 2 tokens), dropping middle names.
    Handles common OCR noise and hyphenated surnames.
    """
    if not s:
        return ""

    # Trim + collapse whitespace
    s = re.sub(r"\s+", " ", s.strip())

    # Remove trailing commas/periods
    s = s.strip(" ,.;:")

    parts = s.split(" ")
    if len(parts) == 1:
        return s

    # If AI accidentally returns "SURNAME Given ..." (common when OCR shows all-caps surname first)
    # Heuristic: first token ALL CAPS (incl Polish), and later tokens not all-caps -> treat first as surname.
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

    # Default: treat first as given, last as surname, drop middle names
    given = parts[0].title()
    surname = parts[-1].title()
    return f"{given} {surname}"


def clean_party_name(raw: str) -> str:
    """Normalize party names extracted from OCR text while stripping address tails."""

    name = raw.strip().strip("-:;•")
    name = re.sub(r"^[\d.\)]+\s*", "", name)

    # Remove obvious address fragments that often trail an entity name
    address_markers = [
        r"\b\d{2}-\d{3}\b",  # postal code
        r"\bul\.?\b",
        r"\bal\.?\b",
        r"\bpl\.?\b",
        r"\bplac\b",
        r"\baleja\b",
    ]
    for marker in address_markers:
        match = re.search(marker, name, flags=re.IGNORECASE)
        if match:
            name = name[: match.start()].rstrip(",; -")
            break
    name = re.sub(r"\(\s*z domu[^)]*\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(?i)\bz domu\b.*", "", name)
    name = name.replace("(", "").replace(")", "")
    name = re.sub(r"(?i)\bpesel[:\s]*\d[\d\s]{9,}\d\b", "", name)
    name = re.sub(r"\b\d{11}\b", "", name)
    name = re.sub(r"\s{2,}", " ", name)
    name = name.strip()
    max_tokens = FILENAME_RULES.get("person_token_limit")
    if max_tokens and max_tokens > 0:
        tokens = name.split()
        if len(tokens) > max_tokens:
            if max_tokens == 2 and len(tokens) >= 2:
                # Preserve probable given + surname by keeping first and last tokens
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
    if re.search(r"[\d/]", joined) or "." in joined:
        return normalized

    given, surname = tokens
    if surname_first:
        return f"{surname} {given}".strip()
    return f"{given} {surname}".strip()


def normalize_target_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def extract_text_ocr(pdf_path: str, char_limit: int, dpi: int, pages: int) -> str:
    output_dir = tempfile.mkdtemp(prefix="ocr_")
    try:
        cmd = [
            PDFTOPPM_EXE,
            "-png",
            "-f",
            "1",
            "-l",
            str(max(1, pages)),
            "-r",
            str(dpi),
            pdf_path,
            os.path.join(output_dir, "page"),
        ]
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        image_paths = sorted(glob.glob(os.path.join(output_dir, "page-*.png")))
        if not image_paths:
            # Fallback for environments where pdftoppm defaults to PPM
            image_paths = sorted(glob.glob(os.path.join(output_dir, "page-*.ppm")))

        if not image_paths:
            raise RuntimeError(
                f"pdftoppm produced no output images for '{os.path.basename(pdf_path)}'"
            )

        text_chunks: list[str] = []
        for image_file in image_paths:
            chunk = pytesseract.image_to_string(Image.open(image_file), lang="pol")
            text_chunks.append(chunk)
            if len("".join(text_chunks)) >= char_limit:
                break
        text = "".join(text_chunks)[:char_limit]
        return text
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


OCR_CACHE: Dict[str, dict] = {}
OCR_CACHE_LOCK = threading.Lock()


def get_ocr_text(pdf_path: str, char_limit: int, dpi: int, pages: int) -> str:
    normalized_pages = max(1, pages)
    with OCR_CACHE_LOCK:
        cached = OCR_CACHE.get(pdf_path)
        if (
            cached
            and cached.get("dpi") == dpi
            and cached.get("pages") == normalized_pages
            and cached.get("char_limit", 0) >= char_limit
        ):
            cached_text = cached.get("ocr_text", "")[:char_limit]
            log_info(
                f"Reusing cached OCR for '{os.path.basename(pdf_path)}' "
                f"(pages={normalized_pages}, dpi={dpi}, char_limit={char_limit})"
            )
            return cached_text
    text = extract_text_ocr(pdf_path, char_limit, dpi, normalized_pages)
    with OCR_CACHE_LOCK:
        OCR_CACHE[pdf_path] = {
            "ocr_text": text,
            "char_limit": max(char_limit, len(text)),
            "dpi": dpi,
            "pages": normalized_pages,
        }
    return text


def call_openai_model(text: str) -> str:
    """Call OpenAI with fallback models, returning the raw content."""

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content


def call_ollama_model(text: str) -> str:
    """Call a local Ollama model using the same system prompt."""

    try:
        payload = {
            "model": "qwen2.5:7b",
            "prompt": f"{SYSTEM_PROMPT}\n\n{text}",
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

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
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


def parse_ai_metadata(raw: str) -> dict:
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
    """Use AI backend to extract metadata; returns empty dict on failure."""

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


def requirements_from_template(template: list[str]) -> dict:
    requirements = {
        "plaintiff": True if "plaintiff" in template else False,
        "defendant": True if "defendant" in template else False,
        "letter_type": True if "letter_type" in template else False,
        "date": True if "date" in template else False,
    }
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


def apply_party_order(meta: dict, options: NamingOptions) -> dict:
    meta = meta.copy()
    if "plaintiff" in meta:
        meta["plaintiff"] = format_party_field(
            meta.get("plaintiff"), options.plaintiff_surname_first
        )
    if "defendant" in meta:
        meta["defendant"] = format_party_field(
            meta.get("defendant"), options.defendant_surname_first
        )
    return meta


def build_filename(meta: dict, options: NamingOptions) -> str:
    parts: list[str] = []
    for element in options.template_elements:
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


@dataclass
class CaseFolderInfo:
    path: str
    tokens: List[str]
    full: str


class DistributionManager:
    def __init__(self, normalizer: Callable[[str], str]):
        self.normalizer = normalizer

    def build_case_index(self, case_root: str) -> List[CaseFolderInfo]:
        entries: List[CaseFolderInfo] = []
        for name in os.listdir(case_root):
            full_path = os.path.join(case_root, name)
            if not os.path.isdir(full_path):
                continue
            normalized = self.normalizer(name)
            tokens = [tok for tok in normalized.split(" ") if tok]
            entries.append(CaseFolderInfo(path=full_path, tokens=tokens, full=normalized))
        return entries

    def _defendant_tokens(self, defendant: str) -> Tuple[List[str], str]:
        normalized = self.normalizer(defendant)
        tokens = [tok for tok in normalized.split(" ") if tok]
        surname = tokens[-1] if tokens else ""
        return tokens, surname

    def find_matches(self, defendants: List[str], case_index: List[CaseFolderInfo]) -> List[CaseFolderInfo]:
        normalized_defendants = []
        for defendant in defendants:
            tokens, surname = self._defendant_tokens(defendant)
            if tokens:
                normalized_defendants.append((tokens, surname))

        matches: Dict[str, CaseFolderInfo] = {}
        for folder in case_index:
            folder_token_set = set(folder.tokens)
            for tokens, surname in normalized_defendants:
                if surname and surname in folder_token_set:
                    matches.setdefault(folder.path, folder)
                    break
                if surname and folder_token_set.issuperset(tokens):
                    matches.setdefault(folder.path, folder)
                    break
        return list(matches.values())

    def copy_pdf(self, source_path: str, target_dir: str, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        candidate = filename
        counter = 1
        os.makedirs(target_dir, exist_ok=True)
        while os.path.exists(os.path.join(target_dir, candidate)):
            candidate = f"{base} ({counter}){ext}"
            counter += 1
        shutil.copy2(source_path, os.path.join(target_dir, candidate))
        return candidate


class FileProcessWorker(QThread):
    finished = pyqtSignal(int, dict)
    failed = pyqtSignal(int, Exception)

    def __init__(
        self,
        index: int,
        pdf_path: str,
        options: NamingOptions,
        stop_event: threading.Event,
        backend: str,
    ):
        super().__init__()
        self.index = index
        self.pdf_path = pdf_path
        self.options = options
        self.stop_event = stop_event
        self.backend = backend
        self.requirements = requirements_from_template(options.template_elements)

    def run(self):
        try:
            if self.stop_event.is_set():
                return
            log_info(
                f"[Worker {self.index + 1}] Starting OCR for '{os.path.basename(self.pdf_path)}' "
                f"(pages={self.options.ocr_pages}, dpi={self.options.ocr_dpi}, "
                f"char_limit={self.options.ocr_char_limit}, backend={self.backend})"
            )
            ocr_text = get_ocr_text(
                self.pdf_path,
                self.options.ocr_char_limit,
                self.options.ocr_dpi,
                self.options.ocr_pages,
            ) if self.options.ocr_enabled else ""
            char_count = len(ocr_text)
            if char_count:
                log_info(f"[Worker {self.index + 1}] OCR extracted {char_count} characters")
            else:
                log_info(
                    f"[Worker {self.index + 1}] OCR returned no text; placeholders likely in output"
                )
            ai_meta = extract_metadata_ai(ocr_text, self.backend, self.options.turbo_mode)
            meta = ai_meta or {}
            defaults_applied = [
                key for key in self.requirements if key not in meta or not meta.get(key)
            ]
            meta = apply_meta_defaults(meta, self.requirements)
            meta = apply_party_order(meta, self.options)

            if defaults_applied:
                log_info(
                    f"[Worker {self.index + 1}] Applied defaults for missing fields: {', '.join(defaults_applied)}"
                )
            log_info(
                f"[Worker {self.index + 1}] Extracted meta: {json.dumps(meta, ensure_ascii=False)}"
            )

            if self.stop_event.is_set():
                return

            filename = build_filename(meta, self.options)

            log_info(
                f"[Worker {self.index + 1}] Proposed filename: {filename} (backend={self.backend})"
            )

            self.finished.emit(
                self.index,
                {
                    "ocr_text": ocr_text,
                    "meta": meta,
                    "filename": filename,
                    "char_count": len(ocr_text),
                },
            )
        except Exception as e:
            log_exception(e)
            self.failed.emit(self.index, e)


class DistributionWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, gui_ref, input_dir: str, pdf_files: List[str], case_index: List[CaseFolderInfo]):
        super().__init__()
        self.gui_ref = gui_ref
        self.input_dir = input_dir
        self.pdf_files = pdf_files
        self.case_index = case_index

    def run(self):
        processed = 0
        total = len(self.pdf_files)
        try:
            for pdf in self.pdf_files:
                pdf_path = os.path.join(self.input_dir, pdf)
                status_text = f"Processing {pdf}"
                self.progress.emit(processed, total, status_text)
                log_lines = [f"PDF: {pdf}"]
                try:
                    result = self.gui_ref.get_or_generate_distribution_result(pdf_path, pdf)
                except Exception as e:
                    log_exception(e)
                    log_lines.append("Status: error while extracting defendants")
                    self.log_ready.emit("\n".join(log_lines))
                    processed += 1
                    self.progress.emit(processed, total, f"{status_text} (error)")
                    continue

                defendants = self.gui_ref.get_defendants_from_result(result)
                if defendants:
                    log_lines.append(f"Defendants: {', '.join(defendants)}")
                    primary_defendant = defendants[0]
                else:
                    log_lines.append("Defendants: none detected")
                    primary_defendant = "—"

                status_text = f"Processing {pdf} → {primary_defendant}"
                matches: List[CaseFolderInfo] = []
                if defendants:
                    try:
                        matches = self.gui_ref.distribution_manager.find_matches(defendants, self.case_index)
                    except Exception as e:
                        log_exception(e)

                if matches:
                    log_lines.append("Matched folders:")
                    for match in matches:
                        log_lines.append(f" - {os.path.basename(match.path)}")
                    if len(matches) > 1:
                        log_lines.append("Note: multiple matches detected, copied to all.")
                    for match in matches:
                        try:
                            copied_name = self.gui_ref.distribution_manager.copy_pdf(pdf_path, match.path, pdf)
                            log_lines.append(
                                f"Action: copied to {os.path.basename(match.path)} as {copied_name}"
                            )
                        except Exception as e:
                            log_exception(e)
                            log_lines.append(
                                f"Action: failed to copy to {os.path.basename(match.path)} ({e})"
                            )
                else:
                    log_lines.append("Status: no matching case folder found")

                self.log_ready.emit("\n".join(log_lines))
                processed += 1
                self.progress.emit(processed, total, status_text)
        except Exception as e:
            log_exception(e)
            self.log_ready.emit(f"Unexpected error during distribution: {e}")
        finally:
            self.finished.emit()


class RenamerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Renamer")
        self.setGeometry(200, 200, 1000, 760)

        self.settings = QSettings("Renamer", "Renamer")

        # State
        self.pdf_files = []
        self.current_index = 0
        self.ocr_text = ""
        self.meta = {}
        self.file_results: dict[int, dict] = {}
        self.active_workers: dict[int, FileProcessWorker] = {}
        self.failed_indices: set[int] = set()
        self.max_parallel_workers = 3
        self.stop_event = threading.Event()
        self.ui_ready = False
        self.distribution_manager = DistributionManager(normalize_polish)
        self.distribution_meta_cache: Dict[str, dict] = {}
        self.distribution_worker: DistributionWorker | None = None

        # Widgets used across the UI
        self.preview_value = QLabel("—")
        self.preview_value.setStyleSheet("font-weight: 600;")

        # ---------- Layout ----------
        root_layout = QVBoxLayout()

        header = QHBoxLayout()
        logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            logo_label = QLabel()
            logo_label.setPixmap(pixmap.scaled(QSize(40, 40), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            header.addWidget(logo_label)
        title_col = QVBoxLayout()
        title_label = QLabel("Renamer")
        title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        subtitle_label = QLabel("Smart document naming")
        subtitle_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        title_col.addWidget(title_label)
        title_col.addWidget(subtitle_label)
        header.addLayout(title_col)
        header.addStretch()
        root_layout.addLayout(header)

        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.settings_tab = QWidget()
        self.distribution_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.settings_tab, "AI Filename Settings")
        self.tabs.addTab(self.distribution_tab, "Distribute PDFs to Case Folders")
        root_layout.addWidget(self.tabs)

        self.main_layout = QVBoxLayout()
        self.main_tab.setLayout(self.main_layout)
        self.settings_layout = QVBoxLayout()
        self.settings_tab.setLayout(self.settings_layout)
        self.distribution_layout = QVBoxLayout()
        self.distribution_tab.setLayout(self.distribution_layout)

        # Distribution tab UI
        dist_input_row = QHBoxLayout()
        dist_input_row.addWidget(QLabel("Folder containing PDFs to distribute:"))
        self.distribution_input_edit = QLineEdit()
        dist_input_row.addWidget(self.distribution_input_edit)
        self.distribution_input_button = QPushButton("Browse")
        self.distribution_input_button.clicked.connect(self.choose_distribution_input)
        dist_input_row.addWidget(self.distribution_input_button)
        self.distribution_layout.addLayout(dist_input_row)

        dist_cases_row = QHBoxLayout()
        dist_cases_row.addWidget(QLabel("Case folders root:"))
        self.case_root_edit = QLineEdit()
        dist_cases_row.addWidget(self.case_root_edit)
        self.case_root_button = QPushButton("Browse")
        self.case_root_button.clicked.connect(self.choose_case_root)
        dist_cases_row.addWidget(self.case_root_button)
        self.distribution_layout.addLayout(dist_cases_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.copy_mode_checkbox = QCheckBox("Copy files (default, mandatory)")
        self.copy_mode_checkbox.setChecked(True)
        self.copy_mode_checkbox.setEnabled(False)
        mode_row.addWidget(self.copy_mode_checkbox)
        mode_row.addStretch()
        self.distribution_layout.addLayout(mode_row)

        dist_controls = QHBoxLayout()
        self.distribution_status_label = QLabel("Idle")
        dist_controls.addWidget(self.distribution_status_label)
        dist_controls.addStretch()
        self.distribution_progress = QProgressBar()
        self.distribution_progress.setRange(0, 1)
        self.distribution_progress.setValue(0)
        self.distribution_progress.setTextVisible(True)
        dist_controls.addWidget(self.distribution_progress)
        self.distribute_button = QPushButton("Distribute")
        self.distribute_button.clicked.connect(self.on_distribute_clicked)
        dist_controls.addWidget(self.distribute_button)
        self.distribution_layout.addLayout(dist_controls)

        self.distribution_layout.addWidget(QLabel("Distribution log:"))
        self.distribution_log_view = QTextEdit()
        self.distribution_log_view.setReadOnly(True)
        self.distribution_log_view.setPlaceholderText(
            "Processing details will appear here. Copies are logged to disk as well."
        )
        self.distribution_log_view.setMinimumHeight(200)
        self.distribution_layout.addWidget(self.distribution_log_view)

        # Input folder
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Input folder:"))
        self.input_edit = QLineEdit()
        h1.addWidget(self.input_edit)
        btn_input = QPushButton("Browse")
        btn_input.clicked.connect(self.choose_input)
        h1.addWidget(btn_input)
        self.main_layout.addLayout(h1)

        # Output folder
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Output folder:"))
        self.output_edit = QLineEdit()
        h2.addWidget(self.output_edit)
        btn_output = QPushButton("Browse")
        btn_output.clicked.connect(self.choose_output)
        h2.addWidget(btn_output)
        self.main_layout.addLayout(h2)

        play_row = QHBoxLayout()
        self.play_button = QPushButton("▶ Generate")
        self.play_button.setStyleSheet("font-size: 16px; padding: 12px; font-weight: bold;")
        self.play_button.clicked.connect(self.start_processing_clicked)
        play_row.addStretch()
        play_row.addWidget(self.play_button)
        play_row.addStretch()
        self.main_layout.addLayout(play_row)

        # OCR options
        h3b = QHBoxLayout()
        self.run_ocr_checkbox = QCheckBox("Run OCR")
        self.run_ocr_checkbox.setChecked(True)
        self.run_ocr_checkbox.toggled.connect(self.update_preview)
        h3b.addWidget(self.run_ocr_checkbox)

        h3b.addWidget(QLabel("Max characters:"))
        self.char_limit_spin = QSpinBox()
        self.char_limit_spin.setRange(100, 10000)
        self.char_limit_spin.setSingleStep(100)
        self.char_limit_spin.setValue(1500)
        self.char_limit_spin.valueChanged.connect(self.update_preview)
        h3b.addWidget(self.char_limit_spin)

        h3b.addWidget(QLabel("OCR DPI:"))
        self.ocr_dpi_spin = QSpinBox()
        self.ocr_dpi_spin.setRange(72, 600)
        self.ocr_dpi_spin.setValue(300)
        self.ocr_dpi_spin.valueChanged.connect(self.update_preview)
        h3b.addWidget(self.ocr_dpi_spin)

        h3b.addWidget(QLabel("Pages to scan:"))
        self.ocr_pages_spin = QSpinBox()
        self.ocr_pages_spin.setRange(1, 50)
        self.ocr_pages_spin.setValue(1)
        self.ocr_pages_spin.valueChanged.connect(self.update_preview)
        h3b.addWidget(self.ocr_pages_spin)

        self.char_count_label = QLabel("Characters retrieved: 0")
        h3b.addWidget(self.char_count_label)
        self.settings_layout.addLayout(h3b)

        self.settings_layout.addWidget(QLabel("Filename template (ordered elements):"))
        template_builder = QHBoxLayout()

        selector_col = QVBoxLayout()
        selector_row = QHBoxLayout()
        self.template_selector = QComboBox()
        self.template_selector.addItem("Date (today)", "date")
        self.template_selector.addItem("Plaintiff", "plaintiff")
        self.template_selector.addItem("Defendant", "defendant")
        self.template_selector.addItem("Letter type", "letter_type")
        selector_row.addWidget(self.template_selector)
        add_template_btn = QPushButton("Add element")
        add_template_btn.clicked.connect(self.add_template_element)
        selector_row.addWidget(add_template_btn)
        selector_col.addLayout(selector_row)

        template_builder.addLayout(selector_col)

        list_col = QHBoxLayout()
        self.template_list = QListWidget()
        self.template_list.setSelectionMode(
            self.template_list.SelectionMode.SingleSelection
        )
        self.template_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.template_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.template_list.model().rowsMoved.connect(lambda *_: self.update_preview())
        for element in DEFAULT_TEMPLATE_ELEMENTS:
            self.add_template_item(element, refresh=False)
        self.update_preview()
        list_col.addWidget(self.template_list)

        buttons_col = QVBoxLayout()
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_selected_template_element)

        buttons_col.addWidget(remove_btn)
        buttons_col.addStretch()
        list_col.addLayout(buttons_col)

        template_builder.addLayout(list_col)
        self.settings_layout.addLayout(template_builder)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("AI Engine:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "OpenAI (cloud)",
            "Local AI (Ollama)",
            "Auto (Local → Cloud)",
        ])
        self.backend_combo.setCurrentIndex(1)
        self.backend_combo.setToolTip("OpenAI = cloud (cost). Ollama = local (free). Auto tries local then cloud.")
        self.backend_combo.currentIndexChanged.connect(self.check_ollama_status)
        backend_row.addWidget(self.backend_combo)
        self.ollama_badge = QLabel("")
        backend_row.addWidget(self.ollama_badge)
        backend_row.addStretch()
        self.settings_layout.addLayout(backend_row)

        turbo_row = QHBoxLayout()
        self.turbo_mode_checkbox = QCheckBox("Turbo mode (parallel AI queries)")
        self.turbo_mode_checkbox.setToolTip("Send a couple of requests to each backend and keep the first valid answer.")
        turbo_row.addWidget(self.turbo_mode_checkbox)
        turbo_row.addStretch()
        self.settings_layout.addLayout(turbo_row)

        name_order_row = QHBoxLayout()
        name_order_row.addWidget(QLabel("Plaintiff order:"))
        self.plaintiff_order_combo = QComboBox()
        self.plaintiff_order_combo.addItem("Surname Name", True)
        self.plaintiff_order_combo.addItem("Name Surname", False)
        self.plaintiff_order_combo.currentIndexChanged.connect(self.update_preview)
        name_order_row.addWidget(self.plaintiff_order_combo)

        name_order_row.addWidget(QLabel("Defendant order:"))
        self.defendant_order_combo = QComboBox()
        self.defendant_order_combo.addItem("Surname Name", True)
        self.defendant_order_combo.addItem("Name Surname", False)
        self.defendant_order_combo.currentIndexChanged.connect(self.update_preview)
        name_order_row.addWidget(self.defendant_order_combo)

        name_order_row.addStretch()
        self.settings_layout.addLayout(name_order_row)
        
        self.main_layout.addWidget(QLabel("Files and proposed names:"))
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["PDF file", "Proposed filename"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.setSelectionBehavior(self.file_table.SelectionBehavior.SelectRows)
        self.file_table.setEditTriggers(self.file_table.EditTrigger.NoEditTriggers)
        self.file_table.cellClicked.connect(self.on_row_selected)
        self.main_layout.addWidget(self.file_table)

        # Filename editing
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Proposed filename:"))
        self.filename_edit = QLineEdit()
        self.filename_edit.editingFinished.connect(self.update_filename_for_current_row)
        h4.addWidget(self.filename_edit)
        self.main_layout.addLayout(h4)

        preview_row = QHBoxLayout()
        preview_label = QLabel("Live preview:")
        preview_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        preview_row.addWidget(preview_label)
        preview_row.addWidget(self.preview_value)
        preview_row.addStretch()
        self.main_layout.addLayout(preview_row)

        self.ocr_preview_label = QLabel("OCR text sent to AI:")
        self.ocr_preview_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.ocr_preview = QTextEdit()
        self.ocr_preview.setReadOnly(True)
        self.ocr_preview.setPlaceholderText(
            "The OCR excerpt forwarded to the AI/backend will appear here."
        )
        self.ocr_preview.setMinimumHeight(140)
        self.main_layout.addWidget(self.ocr_preview_label)
        self.main_layout.addWidget(self.ocr_preview)

        # Buttons row
        h5 = QHBoxLayout()

        btn_process = QPushButton("✎ Rename File")
        btn_process.clicked.connect(self.process_this_file)
        self.btn_process = btn_process

        btn_all = QPushButton("⏩ Rename All")
        btn_all.clicked.connect(self.process_all_files_safe)
        self.btn_all = btn_all

        btn_quit = QPushButton("Quit")
        btn_quit.clicked.connect(self.close)

        h5.addWidget(btn_process)
        h5.addWidget(btn_all)
        h5.addWidget(btn_quit)
        self.main_layout.addLayout(h5)

        copyright_label = QLabel("Copyright 2025-2026 Przemek1337 all rights reserved")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.main_layout.addWidget(copyright_label)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Waiting for input…")
        self.spinner_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.spinner_label)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.progress_bar)
        status_widget.setLayout(status_layout)
        self.status_bar.addPermanentWidget(status_widget, 1)
        root_layout.addWidget(self.status_bar)

        self.setLayout(root_layout)

        self.processing_enabled = False
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self.animate_spinner)
        self.spinner_state = 0

        self.load_settings()
        self.ui_ready = True
        self.update_preview()
        self.check_ollama_status()

    # ------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------

    def load_settings(self):
        self.input_edit.setText(self.settings.value("input_folder", ""))
        self.output_edit.setText(self.settings.value("output_folder", ""))
        self.distribution_input_edit.setText(
            self.settings.value("distribution_input_folder", self.settings.value("input_folder", ""))
        )
        self.case_root_edit.setText(self.settings.value("case_root_folder", ""))
        default_order = FILENAME_RULES.get("surname_first", True)
        turbo_mode = self.settings.value("turbo_mode", False)
        self.turbo_mode_checkbox.setChecked(str(turbo_mode).lower() == "true")
        plaintiff_order = self.settings.value("plaintiff_surname_first", default_order)
        defendant_order = self.settings.value("defendant_surname_first", default_order)
        plaintiff_order_bool = str(plaintiff_order).lower() == "true"
        defendant_order_bool = str(defendant_order).lower() == "true"
        self.plaintiff_order_combo.setCurrentIndex(0 if plaintiff_order_bool else 1)
        self.defendant_order_combo.setCurrentIndex(0 if defendant_order_bool else 1)
        saved_template = self.settings.value("template", [])
        if isinstance(saved_template, str):
            saved_template = json.loads(saved_template) if saved_template else []
        if saved_template:
            self.template_list.clear()
            for element in saved_template:
                self.add_template_item(element)

        # Automatically reload PDFs from the last used folder if it still exists.
        saved_input = self.input_edit.text()
        if saved_input and os.path.isdir(saved_input):
            self.load_pdfs()

    def save_settings(self):
        self.settings.setValue("input_folder", self.input_edit.text())
        self.settings.setValue("output_folder", self.output_edit.text())
        self.settings.setValue("distribution_input_folder", self.distribution_input_edit.text())
        self.settings.setValue("case_root_folder", self.case_root_edit.text())
        self.settings.setValue("template", self.get_template_elements())
        self.settings.setValue("turbo_mode", self.turbo_mode_checkbox.isChecked())
        self.settings.setValue(
            "plaintiff_surname_first", bool(self.plaintiff_order_combo.currentData())
        )
        self.settings.setValue(
            "defendant_surname_first", bool(self.defendant_order_combo.currentData())
        )

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def log_activity(self, message: str):
        log_info(message)

    def set_status(self, text: str):
        if not text:
            text = "Working…"
        self.status_label.setText(text)

    def animate_spinner(self):
        dots = "." * (self.spinner_state % 4)
        self.spinner_label.setText(f"⏳{dots}")
        self.spinner_state += 1

    def update_processing_progress(self, total: int = None, processed_override: int = None):
        total_files = total if total is not None else len(self.pdf_files)
        if total_files <= 0:
            total_files = 1
        processed = (
            processed_override
            if processed_override is not None
            else len(self.file_results) + len(self.failed_indices)
        )
        processed = min(processed, total_files)
        self.progress_bar.setRange(0, total_files)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat(f"{processed}/{total_files} processed")
        self.progress_bar.setTextVisible(True)

    def start_processing_ui(self, status: str = "Processing…", total: int = None):
        self.set_status(status)
        self.update_processing_progress(total)
        self.spinner_timer.start(300)
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(True)

    def stop_processing_ui(self, status: str = "Idle"):
        self.set_status(status)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.progress_bar.setTextVisible(False)
        self.spinner_timer.stop()
        self.spinner_label.setText("")
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(False)

    def start_distribution_ui(self, total: int):
        self.distribute_button.setDisabled(True)
        self.distribution_input_button.setDisabled(True)
        self.case_root_button.setDisabled(True)
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(True)
        self.distribution_status_label.setText("Processing…")
        self.distribution_progress.setRange(0, max(1, total))
        self.distribution_progress.setValue(0)

    def stop_distribution_ui(self, status: str = "Idle"):
        self.distribute_button.setDisabled(False)
        self.distribution_input_button.setDisabled(False)
        self.case_root_button.setDisabled(False)
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(False)
        self.distribution_status_label.setText(status)
        self.distribution_progress.setRange(0, 1)
        self.distribution_progress.setValue(0)

    def check_ollama_status(self):
        if self.backend_combo.currentIndex() != 1:
            self.ollama_badge.setText("")
            return
        try:
            resp = requests.get(urljoin(OLLAMA_HOST, "api/tags"), timeout=2)
            ok = resp.status_code == 200
        except Exception:
            ok = False
        if ok:
            self.ollama_badge.setText("🟢 Connected")
            self.ollama_badge.setStyleSheet("color: #7CFC00;")
        else:
            self.ollama_badge.setText("🔴 Offline")
            self.ollama_badge.setStyleSheet("color: #FF6B6B;")

    # ------------------------------------------------------
    # Folder Selection
    # ------------------------------------------------------

    def choose_input(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder")
            if not folder:
                return
            self.input_edit.setText(folder)
            self.load_pdfs()
        except Exception as e:
            log_exception(e)
            show_friendly_error(
                self,
                "Folder error",
                "Renamer could not open the selected input folder.",
                traceback.format_exc(),
            )

    def choose_output(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if not folder:
                return
            self.output_edit.setText(folder)
        except Exception as e:
            log_exception(e)
            show_friendly_error(
                self,
                "Folder error",
                "Renamer could not open the selected output folder.",
                traceback.format_exc(),
            )

    def choose_distribution_input(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder to Distribute")
            if not folder:
                return
            self.distribution_input_edit.setText(folder)
        except Exception as e:
            log_exception(e)
            show_friendly_error(
                self,
                "Folder error",
                "Renamer could not open the selected distribution input folder.",
                traceback.format_exc(),
            )

    def choose_case_root(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Case Folders Root")
            if not folder:
                return
            self.case_root_edit.setText(folder)
        except Exception as e:
            log_exception(e)
            show_friendly_error(
                self,
                "Folder error",
                "Renamer could not open the selected case root folder.",
                traceback.format_exc(),
            )

    def append_distribution_log_message(self, message: str):
        if message:
            self.distribution_log_view.append(message)
            append_distribution_log(message)

    def handle_distribution_progress(self, processed: int, total: int, status_text: str):
        total = max(1, total)
        processed = min(processed, total)
        self.distribution_status_label.setText(status_text or "Processing…")
        self.distribution_progress.setRange(0, total)
        self.distribution_progress.setValue(processed)
        self.distribution_progress.setFormat(f"{processed}/{total}")
        self.distribution_progress.setTextVisible(True)

    def handle_distribution_log(self, message: str):
        self.append_distribution_log_message(message)

    def handle_distribution_finished(self):
        self.distribution_status_label.setText("Finished")
        if self.distribution_progress.maximum() > 0:
            self.distribution_progress.setValue(self.distribution_progress.maximum())
        self.stop_distribution_ui("Finished")
        QMessageBox.information(self, "Distribution complete", "Finished distributing PDFs.")
        self.distribution_worker = None

    # ------------------------------------------------------
    # Load PDFs
    # ------------------------------------------------------

    def load_pdfs(self):
        folder = self.input_edit.text()
        if not os.path.isdir(folder):
            show_friendly_error(
                self,
                "Input folder unavailable",
                "Renamer could not find the selected input folder.",
                f"Checked path: {folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        self.stop_event.clear()
        self.pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
        self.current_index = 0
        self.file_results.clear()
        self.active_workers.clear()
        self.failed_indices.clear()
        self.log_activity(f"Loaded {len(self.pdf_files)} PDF files")
        self.set_status("Waiting for generate…")

        self.processing_enabled = False

        self.file_table.setRowCount(0)
        for idx, filename in enumerate(self.pdf_files):
            self.file_table.insertRow(idx)
            self.file_table.setItem(idx, 0, QTableWidgetItem(filename))
            self.file_table.setItem(idx, 1, QTableWidgetItem(filename))

        if not self.pdf_files:
            show_friendly_error(
                self,
                "No PDFs detected",
                "Renamer did not find any PDF files in this folder.",
                f"Looked in: {folder}",
                icon=QMessageBox.Icon.Information,
            )
            return

        self.file_table.selectRow(0)

    # ------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------

    def parse_defendant_field(self, value) -> list[str]:
        names: list[str] = []
        if isinstance(value, str):
            names = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, list):
            names = [str(item).strip() for item in value if str(item).strip()]
        cleaned: list[str] = []
        for name in names:
            lower_name = name.lower()
            if lower_name == "defendant":
                continue
            if name not in cleaned:
                cleaned.append(name)
        return cleaned

    def get_defendants_from_result(self, result: dict) -> list[str]:
        source_meta = result.get("raw_meta") or result.get("meta") or {}
        names = self.parse_defendant_field(source_meta.get("defendant"))
        return names

    def get_or_generate_distribution_result(self, pdf_path: str, filename: str) -> dict:
        if pdf_path in self.distribution_meta_cache:
            return self.distribution_meta_cache[pdf_path]

        try:
            if os.path.abspath(os.path.dirname(pdf_path)) == os.path.abspath(self.input_edit.text()):
                if filename in self.pdf_files:
                    idx = self.pdf_files.index(filename)
                    cached = self.file_results.get(idx)
                    if cached:
                        result = {
                            "meta": cached.get("meta", {}),
                            "raw_meta": cached.get("meta", {}),
                            "ocr_text": cached.get("ocr_text", ""),
                        }
                        self.distribution_meta_cache[pdf_path] = result
                        return result
        except Exception as e:
            log_exception(e)

        options = self.build_options()
        requirements = requirements_from_template(options.template_elements)
        self.log_activity(f"[Distribution] Running OCR/AI for '{filename}'")
        ocr_text = ""
        try:
            if options.ocr_enabled:
                ocr_text = get_ocr_text(
                    pdf_path,
                    options.ocr_char_limit,
                    options.ocr_dpi,
                    options.ocr_pages,
                )
        except Exception as e:
            log_exception(e)
            ocr_text = ""

        ai_meta = extract_metadata_ai(ocr_text, self.get_ai_backend(), options.turbo_mode)
        meta = apply_meta_defaults(ai_meta or {}, requirements)
        meta = apply_party_order(meta, options)
        result = {"meta": meta, "raw_meta": ai_meta or {}, "ocr_text": ocr_text}
        self.distribution_meta_cache[pdf_path] = result
        return result

    def on_distribute_clicked(self):
        input_dir = self.distribution_input_edit.text() or self.input_edit.text()
        case_root = self.case_root_edit.text()

        if self.distribution_worker and self.distribution_worker.isRunning():
            show_friendly_error(
                self,
                "Distribution in progress",
                "Please wait for the current distribution run to finish.",
                "Distribution worker already running.",
                icon=QMessageBox.Icon.Information,
            )
            return

        if not os.path.isdir(input_dir):
            show_friendly_error(
                self,
                "Input folder unavailable",
                "Renamer could not find the selected distribution input folder.",
                f"Checked path: {input_dir}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not os.path.isdir(case_root):
            show_friendly_error(
                self,
                "Case root unavailable",
                "Renamer could not find the selected case root folder.",
                f"Checked path: {case_root}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            show_friendly_error(
                self,
                "No PDFs detected",
                "Renamer did not find any PDF files to distribute.",
                f"Looked in: {input_dir}",
                icon=QMessageBox.Icon.Information,
            )
            return

        try:
            case_index = self.distribution_manager.build_case_index(case_root)
        except Exception as e:
            log_exception(e)
            show_friendly_error(
                self,
                "Case root error",
                "Renamer could not read the case folders.",
                traceback.format_exc(),
            )
            return

        if not case_index:
            show_friendly_error(
                self,
                "No case folders",
                "Renamer did not find any case folders inside the selected root.",
                f"Checked path: {case_root}",
                icon=QMessageBox.Icon.Information,
            )
            return

        self.distribution_log_view.clear()
        self.start_distribution_ui(len(pdf_files))
        self.distribution_worker = DistributionWorker(self, input_dir, pdf_files, case_index)
        self.distribution_worker.progress.connect(self.handle_distribution_progress)
        self.distribution_worker.log_ready.connect(self.handle_distribution_log)
        self.distribution_worker.finished.connect(self.handle_distribution_finished)
        self.distribution_worker.start()

    def get_ai_backend(self) -> str:
        idx = self.backend_combo.currentIndex()
        if idx == 1:
            return "ollama"
        if idx == 2:
            return "auto"
        return "openai"

    def build_options(self) -> NamingOptions:
        return NamingOptions(
            template_elements=self.get_template_elements(),
            ocr_enabled=self.run_ocr_checkbox.isChecked(),
            ocr_char_limit=self.char_limit_spin.value(),
            ocr_dpi=self.ocr_dpi_spin.value(),
            ocr_pages=self.ocr_pages_spin.value(),
            plaintiff_surname_first=bool(self.plaintiff_order_combo.currentData()),
            defendant_surname_first=bool(self.defendant_order_combo.currentData()),
            turbo_mode=self.turbo_mode_checkbox.isChecked(),
        )

    def display_name_for_element(self, element: str) -> str:
        mapping = {
            "date": "Date (today)",
            "plaintiff": "Plaintiff",
            "defendant": "Defendant",
            "letter_type": "Letter type",
        }
        return mapping.get(element, element)

    def add_template_item(self, element: str, refresh: bool = True):
        item = QListWidgetItem(self.display_name_for_element(element))
        item.setData(Qt.ItemDataRole.UserRole, element)
        self.template_list.addItem(item)
        if refresh:
            self.update_preview()

    def add_template_element(self):
        element = self.template_selector.currentData()
        if not element:
            return
        self.add_template_item(element)

    def remove_selected_template_element(self):
        row = self.template_list.currentRow()
        if row >= 0:
            self.template_list.takeItem(row)
            self.update_preview()

    def move_template_element_up(self):
        row = self.template_list.currentRow()
        if row <= 0:
            return
        item = self.template_list.takeItem(row)
        self.template_list.insertItem(row - 1, item)
        self.template_list.setCurrentRow(row - 1)

    def move_template_element_down(self):
        row = self.template_list.currentRow()
        if row < 0 or row >= self.template_list.count() - 1:
            return
        item = self.template_list.takeItem(row)
        self.template_list.insertItem(row + 1, item)
        self.template_list.setCurrentRow(row + 1)

    def get_template_elements(self) -> list[str]:
        elements: list[str] = []
        for idx in range(self.template_list.count()):
            item = self.template_list.item(idx)
            value = item.data(Qt.ItemDataRole.UserRole)
            if value:
                elements.append(value)

        if not elements:
            elements = DEFAULT_TEMPLATE_ELEMENTS[:]
            for element in elements:
                self.add_template_item(element, refresh=False)

        return elements

    def update_ocr_preview(self, text: str):
        if hasattr(self, "ocr_preview"):
            self.ocr_preview.setPlainText(text or "")

    def update_preview(self):
        if not getattr(self, "ui_ready", False):
            return
        options = self.build_options()
        meta = self.meta or {}
        if self.current_index in self.file_results:
            meta = self.file_results[self.current_index].get("meta", meta)
        requirements = requirements_from_template(options.template_elements)
        meta = apply_meta_defaults(meta, requirements)
        meta = apply_party_order(meta, options)
        filename = build_filename(meta, options)
        display_name = filename or "—"
        self.preview_value.setText(display_name)
        if filename:
            self.filename_edit.blockSignals(True)
            self.filename_edit.setText(filename)
            self.filename_edit.blockSignals(False)
            if self.current_index < self.file_table.rowCount():
                self.file_table.setItem(self.current_index, 1, QTableWidgetItem(filename))

    # ------------------------------------------------------
    # Process One File
    # ------------------------------------------------------

    def process_current_file(self):
        if not self.pdf_files or not self.processing_enabled:
            return

        folder = self.input_edit.text()
        pdf = self.pdf_files[self.current_index]
        pdf_path = os.path.join(folder, pdf)
        self.start_worker_for_index(self.current_index, pdf_path)

    # ------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------

    def start_processing_clicked(self):
        if self.processing_enabled and self.active_workers:
            self.stop_and_reprocess()
            return
        if not self.pdf_files:
            log_info("Generate clicked with no files queued")
            show_friendly_error(
                self,
                "No files queued",
                "Select an input folder with PDFs before generating names.",
                "Nothing to process yet.",
                icon=QMessageBox.Icon.Information,
            )
            return

        self.stop_event.clear()
        self.active_workers.clear()
        self.failed_indices.clear()
        self.file_results.clear()
        self.ocr_text = ""
        self.meta = {}
        self.filename_edit.clear()
        self.char_count_label.setText("Characters retrieved: 0")
        self.update_ocr_preview("")
        for row in range(self.file_table.rowCount()):
            source_item = self.file_table.item(row, 0)
            current_name = source_item.text() if source_item else ""
            self.file_table.setItem(row, 1, QTableWidgetItem(current_name))
        self.current_index = 0
        self.processing_enabled = True
        log_info(
            f"Starting generation for {len(self.pdf_files)} files using backend {self.get_ai_backend()}"
        )
        self.start_processing_ui("Generating proposals…", total=len(self.pdf_files))
        self.start_parallel_processing()

    def stop_and_reprocess(self):
        self.stop_event.set()
        log_info("Stopping current processing and resetting state")

        for worker in list(self.active_workers.values()):
            worker.requestInterruption()

        self.active_workers.clear()
        self.file_results.clear()
        self.failed_indices.clear()

        self.ocr_text = ""
        self.meta = {}
        self.filename_edit.clear()
        self.char_count_label.setText("Characters retrieved: 0")
        self.update_ocr_preview("")

        for row in range(self.file_table.rowCount()):
            current_name = self.file_table.item(row, 0).text()
            self.file_table.setItem(row, 1, QTableWidgetItem(current_name))

        self.stop_event.clear()
        self.current_index = 0
        self.processing_enabled = False
        if self.pdf_files:
            self.file_table.selectRow(0)
        self.stop_processing_ui("Reset")

    def process_this_file(self):
        out_folder = self.output_edit.text()
        if not os.path.isdir(out_folder):
            show_friendly_error(
                self,
                "Output folder missing",
                "Please choose where renamed files should be saved.",
                f"Checked path: {out_folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not self.pdf_files or (self.current_index in self.active_workers):
            return

        self.stop_event.clear()

        self.update_filename_for_current_row()
        self.start_processing_ui("Renaming current file…", total=1)

        pdf_name = self.pdf_files[self.current_index]
        inp = os.path.join(self.input_edit.text(), pdf_name)

        proposed = self.file_table.item(self.current_index, 1)
        target_name_raw = proposed.text() if proposed else self.filename_edit.text()
        target_name = normalize_target_filename(target_name_raw)
        if not target_name:
            show_friendly_error(
                self,
                "No filename",
                "Proposed filename cannot be empty.",
                "Filename validation stopped the operation.",
                icon=QMessageBox.Icon.Warning,
            )
            return

        out = os.path.join(out_folder, target_name)

        try:
            shutil.move(inp, out)
            if self.current_index in self.file_results:
                self.file_results[self.current_index]["filename"] = target_name
            self.update_processing_progress(total=1, processed_override=1)
            QMessageBox.information(self, "Done", f"Renamed:\n{out}")
        except Exception as e:
            log_exception(e)
            show_friendly_error(
                self,
                "Rename failed",
                "Renamer could not move the file to the output folder.",
                traceback.format_exc(),
            )
        finally:
            self.stop_processing_ui("Idle")

    def process_all_files_safe(self):
        out_folder = self.output_edit.text()
        if not os.path.isdir(out_folder):
            show_friendly_error(
                self,
                "Output folder missing",
                "Please choose where renamed files should be saved.",
                f"Checked path: {out_folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not self.pdf_files:
            return

        self.stop_event.clear()
        self.start_processing_ui("Renaming all files…", total=len(self.pdf_files))
        for idx, pdf_name in enumerate(self.pdf_files[:]):
            try:
                result = self.file_results.get(idx)
                if result is None:
                    result = self.generate_result_for_index(idx)
                    self.file_results[idx] = result

                self.current_index = idx
                self.apply_cached_result(idx, self.file_results[idx])
                proposed_item = self.file_table.item(idx, 1)
                raw_name = proposed_item.text() if proposed_item else result.get("filename", pdf_name)
                target_name = normalize_target_filename(raw_name)
                if not target_name:
                    raise ValueError(f"Empty or invalid filename for {pdf_name}")
                self.file_results[idx]["filename"] = target_name

                inp_path = os.path.join(self.input_edit.text(), pdf_name)
                out_path = os.path.join(out_folder, target_name)
                shutil.move(inp_path, out_path)
                self.update_processing_progress(
                    total=len(self.pdf_files), processed_override=idx + 1
                )
            except Exception as e:
                log_exception(e)
                show_friendly_error(
                    self,
                    "File error",
                    "Renamer hit a problem while renaming one of the files.",
                    traceback.format_exc(),
                    icon=QMessageBox.Icon.Warning,
                )
                continue

        QMessageBox.information(self, "Done", "All files processed.")
        self.stop_processing_ui("Idle")

    def process_all(self):
        out_folder = self.output_edit.text()
        if not os.path.isdir(out_folder):
            show_friendly_error(
                self,
                "Output folder missing",
                "Please choose where renamed files should be saved.",
                f"Checked path: {out_folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not self.pdf_files:
            return

        for idx, pdf_name in enumerate(self.pdf_files[:]):  # iterate over COPY
            result = self.file_results.get(idx)
            if result is None:
                result = self.generate_result_for_index(idx)
                self.file_results[idx] = result
            self.apply_cached_result(idx, result)

            inp = os.path.join(self.input_edit.text(), pdf_name)
            out = os.path.join(out_folder, self.filename_edit.text())
            shutil.move(inp, out)

        QMessageBox.information(self, "Done", "All files processed.")

    # Helpers
    def handle_worker_finished(self, index: int, result: dict):
        self.active_workers.pop(index, None)
        if self.stop_event.is_set():
            return
        self.file_results[index] = result
        self.apply_cached_result(index, result)
        self.update_processing_progress()
        self.start_parallel_processing()
        self.log_activity(
            f"✓ Processed file {index + 1} of {len(self.pdf_files)} (chars: {result.get('char_count', 0)})"
        )
        if not self.active_workers:
            self.stop_processing_ui("Idle")
            log_info("All queued workers completed")

    def handle_worker_failed(self, index: int, error: Exception):
        self.active_workers.pop(index, None)
        if self.stop_event.is_set():
            return
        self.failed_indices.add(index)
        log_exception(error)
        log_info(f"Worker {index} failed: {error}")
        show_friendly_error(
            self,
            "Processing failed",
            "Renamer could not finish processing one of the files.",
            traceback.format_exc(),
        )
        self.update_processing_progress()
        self.start_parallel_processing()
        if not self.active_workers:
            self.stop_processing_ui("Idle")

    def on_row_selected(self, row: int, _col: int):
        self.current_index = row
        item = self.file_table.item(row, 0)
        if item:
            self.file_table.scrollToItem(item)

        if row in self.file_results:
            self.apply_cached_result(row, self.file_results[row])
        else:
            item = self.file_table.item(row, 1)
            current_name = item.text() if item else ""
            self.filename_edit.setText(current_name)
            if not self.processing_enabled:
                self.ocr_text = ""
                self.meta = {}
                self.char_count_label.setText("Characters retrieved: 0")
                self.update_ocr_preview("")
            if self.processing_enabled:
                self.process_current_file()
        self.update_preview()

    def generate_result_for_index(self, index: int) -> dict:
        pdf = self.pdf_files[index]
        pdf_path = os.path.join(self.input_edit.text(), pdf)
        global AI_BACKEND
        AI_BACKEND = self.get_ai_backend()
        options = self.build_options()
        requirements = requirements_from_template(options.template_elements)
        self.log_activity(
            f"[UI] Starting OCR for '{pdf}' (pages={options.ocr_pages}, dpi={options.ocr_dpi}, "
            f"char_limit={options.ocr_char_limit}, backend={AI_BACKEND})"
        )
        ocr_text = get_ocr_text(
            pdf_path,
            options.ocr_char_limit,
            options.ocr_dpi,
            options.ocr_pages,
        ) if options.ocr_enabled else ""

        char_count = len(ocr_text)
        self.log_activity(f"[UI] OCR extracted {char_count} characters for '{pdf}'")
        if char_count == 0:
            self.log_activity(
                f"[UI] No OCR text for '{pdf}'; filenames will rely on placeholders/defaults"
            )

        ai_meta = extract_metadata_ai(ocr_text, self.get_ai_backend(), options.turbo_mode)
        meta = ai_meta or {}
        defaults_applied = [key for key in requirements if key not in meta or not meta.get(key)]
        meta = apply_meta_defaults(meta, requirements)
        meta = apply_party_order(meta, options)

        if defaults_applied:
            self.log_activity(
                f"[UI] Applied defaults for missing fields: {', '.join(defaults_applied)}"
            )
        self.log_activity(f"[UI] Extracted meta: {json.dumps(meta, ensure_ascii=False)}")

        filename = build_filename(meta, options)

        self.log_activity(f"[UI] Proposed filename: {filename} (backend={AI_BACKEND})")

        return {
            "ocr_text": ocr_text,
            "meta": meta,
            "filename": filename,
            "char_count": len(ocr_text),
        }

    def apply_cached_result(self, index: int, cached: dict):
        if index == self.current_index:
            self.ocr_text = cached.get("ocr_text", "")
            self.meta = cached.get("meta", {})
            self.filename_edit.setText(cached.get("filename", ""))
            self.char_count_label.setText(f"Characters retrieved: {cached.get('char_count', 0)}")
            self.update_ocr_preview(self.ocr_text)

        self.file_table.setItem(index, 1, QTableWidgetItem(cached.get("filename", "")))
        self.update_preview()

    def update_filename_for_current_row(self):
        if not self.pdf_files:
            return
        if self.current_index >= self.file_table.rowCount():
            return

        current_text = normalize_target_filename(self.filename_edit.text())
        self.filename_edit.setText(current_text)
        self.file_table.setItem(self.current_index, 1, QTableWidgetItem(current_text))

        if self.current_index in self.file_results:
            self.file_results[self.current_index]["filename"] = current_text

    # Background processing
    def start_worker_for_index(self, index: int, pdf_path: str):
        if self.stop_event.is_set():
            return
        if index in self.failed_indices:
            return
        if index in self.active_workers or index in self.file_results:
            return

        global AI_BACKEND
        AI_BACKEND = self.get_ai_backend()
        options = self.build_options()
        worker = FileProcessWorker(
            index=index,
            pdf_path=pdf_path,
            options=options,
            stop_event=self.stop_event,
            backend=self.get_ai_backend(),
        )
        worker.finished.connect(self.handle_worker_finished)
        worker.failed.connect(self.handle_worker_failed)
        self.active_workers[index] = worker
        self.set_status(
            f"Running OCR ({options.ocr_pages} page(s) @ {options.ocr_dpi} DPI) for file {index + 1}…"
        )
        self.log_activity(
            f"→ Processing file {index + 1} with backend {self.get_ai_backend()}"
        )
        worker.start()

    def start_parallel_processing(self):
        if not self.processing_enabled:
            log_info("Parallel processing skipped: disabled state")
            return
        if self.stop_event.is_set():
            log_info("Parallel processing halted due to stop event")
            return
        if not self.pdf_files:
            log_info("Parallel processing skipped: no files loaded")
            return

        started_any = False
        for idx in range(len(self.pdf_files)):
            if len(self.active_workers) >= self.max_parallel_workers:
                break
            if idx in self.file_results or idx in self.active_workers or idx in self.failed_indices:
                continue
            pdf_path = os.path.join(self.input_edit.text(), self.pdf_files[idx])
            self.start_worker_for_index(idx, pdf_path)
            started_any = True

        if not started_any and not self.active_workers:
            log_info("No workers started; marking UI idle")
            self.stop_processing_ui("Idle")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Renamer")
    app.setApplicationDisplayName("Renamer")
    app.setStyleSheet(GLOBAL_STYLESHEET)
    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
    icon_path = os.path.join(BASE_DIR, "assets", "logo.ico")
    icon_file = icon_path if os.path.exists(icon_path) else logo_path
    if os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))
    gui = RenamerGUI()
    if os.path.exists(icon_file):
        gui.setWindowIcon(QIcon(icon_file))
    gui.show()
    sys.exit(app.exec())
