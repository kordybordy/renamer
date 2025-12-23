import os
import sys
from urllib.parse import urljoin


# --------- HARD-CODED API KEY (EDIT THIS LINE!) ----------
API_KEY = "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"
# ----------------------------------------------------------

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AI_BACKEND_DEFAULT = os.environ.get("AI_BACKEND", "openai")  # openai | ollama | auto
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.renamer.win/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", urljoin(OLLAMA_HOST, "api/generate"))

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

LOG_FILE = os.path.join(os.path.expanduser("~"), "renamer_error.log")
DISTRIBUTION_LOG_FILE = os.path.join(os.path.expanduser("~"), "renamer_distribution.log")

ACCENT_COLOR = "#00FF66"
BACKGROUND_COLOR = "#000000"
PANEL_COLOR = "#000000"
TEXT_PRIMARY = "#00FF66"
TEXT_SECONDARY = "#00CC55"
BORDER_COLOR = "#00FF66"


def load_stylesheet() -> str:
    retro_path = os.path.join(BASE_DIR, "retro.qss")
    try:
        with open(retro_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return f"""
* {{
    font-family: 'Segoe UI', sans-serif;
    color: {TEXT_PRIMARY};
}}
QWidget {{
    background-color: {BACKGROUND_COLOR};
}}
"""


GLOBAL_STYLESHEET = load_stylesheet()

POPPLER_PATH = os.path.join(BASE_DIR, "poppler", "Library", "bin")
PDFTOPPM_EXE = os.path.join(POPPLER_PATH, "pdftoppm.exe")
os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

if not os.path.exists(PDFTOPPM_EXE):
    raise RuntimeError(f"pdftoppm.exe not found: {PDFTOPPM_EXE}")
