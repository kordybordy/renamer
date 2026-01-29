import os
from urllib.parse import urljoin


AI_BACKEND = os.environ.get("AI_BACKEND", "openai")  # openai | ollama | auto
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


BASE_SYSTEM_PROMPT = """
Return JSON only, no markdown, no commentary.
Return strict JSON in this exact shape (include every listed field):

{
  "plaintiff": ["Given Surname", ...],
  "defendant": ["Given Surname", ...],
  "case_numbers": ["I C 1234/25", ...],
  "letter_type": "Pozew" | "Pozew + Postanowienie" |
                 "Postanowienie" | "Portal" | "Korespondencja" |
                 "Unknown" | "Zawiadomienie" |
                 "Odpowiedź na pozew" | "Wniosek" | "Replika",
  "custom": {}
}

Rules:
- letter_type MUST be one of the allowed values above.
- Ignore DWF Poland Jamka and Raiffeisen Bank (do not include them in any party list).
- Each list item MUST be EXACTLY TWO WORDS: "Given Surname".
  If the person has multiple given names, KEEP ONLY THE FIRST given name.
  Examples:
    "Szymon Hubert Marciniak" -> "Szymon Marciniak"
    "Katarzyna Magdalena Obałek" -> "Katarzyna Obałek"
- Never include PESEL, addresses, or IDs.
- Extract ALL case numbers.
- Do NOT invent placeholders like "Defendant S Surname Name".
- Preserve Polish letters.
- If custom fields are provided, populate them under "custom".
- If no custom fields are provided, return "custom": {}.
"""


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
