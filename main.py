# ==========================================================
# PyQt5 GUI PDF Renamer — OCR + AI (GPT-5-nano / GPT-4.1-mini)
# Human-format filenames (comma-separated parties, Polish letters preserved)
# ==========================================================

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
import requests
from dataclasses import dataclass

from PIL import Image
import pytesseract

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox,
    QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QRadioButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

AI_BACKEND = os.environ.get("AI_BACKEND", "openai")  # openai | ollama | auto

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

# ===============================
# FILENAME POLICY
# ===============================

FILENAME_RULES = {
    "remove_raiffeisen": True,        # always remove Raiffeisen from parties
    "max_parties": 3,                 # limit number of names in filename
    "surname_first": True,            # SURNAME Name
    "use_commas": True,               # comma-separated parties
    "replace_slash_only": True,       # only replace "/" → "_"
    "force_letter_type": True,        # GUI overrides AI letter type
    "default_letter_type": "pozew",   # fallback if AI unsure
}


# ===============================
# Logging
# ===============================
import traceback
from datetime import datetime

LOG_FILE = os.path.join(os.path.expanduser("~"), "ai_pdf_renamer_error.log")

def log_exception(e: Exception):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(datetime.now().isoformat() + "\n")
        f.write(str(e) + "\n")
        f.write(traceback.format_exc())
        f.flush()
        os.fsync(f.fileno())

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
    tessdata_dir  = os.path.join(tesseract_dir, "tessdata")

    tesseract_exe = os.path.join(tesseract_dir, "tesseract.exe")

    if not os.path.exists(tesseract_exe):
        raise RuntimeError(f"Tesseract EXE not found: {tesseract_exe}")

    if not os.path.exists(tessdata_dir):
        raise RuntimeError(f"Tessdata folder not found: {tessdata_dir}")

    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    os.environ["TESSDATA_PREFIX"] = tessdata_dir

    return tessdata_dir

# Call ONCE
TESSDATA_DIR = configure_tesseract()

print("[DEBUG] Using tessdata:", TESSDATA_DIR)

# ==========================================================
# AI PROMPTING
# ==========================================================

OUTPUT_SCHEMA = """
{
  "plaintiff": ["Name Surname", "..."],
  "defendant": ["Name Surname", "..."],
  "case_numbers": ["I C 1234/25", "..."],
  "letter_type": "pozew | postanowienie | wezwanie | odpowiedz_na_pozew | wniosek | unknown"
}
"""

DEFAULT_PROMPT = """
Extract the requested legal metadata from the following OCR text.
Return strict JSON exactly matching this schema:
{SCHEMA}

OCR text:
{TEXT}
"""

PROMPT_FLAGS_DEFAULT = {
    "plaintiff": True,
    "defendant": True,
    "case_numbers": True,
    "letter_type": True,
    "raiffeisen_rule": True,
}

PROMPT_CONTEXT = {
    "template": DEFAULT_PROMPT,
    "flags": PROMPT_FLAGS_DEFAULT.copy(),
}


@dataclass
class NamingOptions:
    include_parties: bool = True
    include_case_numbers: bool = True
    include_letter_type: bool = True
    party_mode: str = "opposing"  # opposing | plaintiff | defendant
    ocr_enabled: bool = True
    ocr_char_limit: int = 1500
    ocr_dpi: int = 300
    prompt_template: str = DEFAULT_PROMPT


def set_prompt_context(template: str, flags: dict):
    PROMPT_CONTEXT["template"] = template or DEFAULT_PROMPT
    PROMPT_CONTEXT["flags"] = {**PROMPT_FLAGS_DEFAULT, **(flags or {})}


def build_prompt(text: str) -> str:
    template = PROMPT_CONTEXT.get("template", DEFAULT_PROMPT)
    flags = PROMPT_CONTEXT.get("flags", PROMPT_FLAGS_DEFAULT)

    directives = []
    if flags.get("plaintiff"):
        directives.append("- Extract plaintiffs.")
    if flags.get("defendant"):
        directives.append("- Extract defendants.")
    if flags.get("case_numbers"):
        directives.append("- Extract case numbers.")
    if flags.get("letter_type"):
        directives.append("- Determine letter type.")
    if flags.get("raiffeisen_rule"):
        directives.append("- Apply Raiffeisen exclusion (opposing side is relevant).")

    prefix = "\n".join(directives)
    prompt_body = template.replace("{SCHEMA}", OUTPUT_SCHEMA).replace("{TEXT}", text)
    if prefix:
        return prefix + "\n\n" + prompt_body
    return prompt_body

# ==========================================================
# Helpers
# ==========================================================

def sanitize_case_number(case: str) -> str:
    """Replace only '/' with '_' in case numbers."""
    return case.replace("/", "_")

def extract_text_ocr(pdf_path: str, max_chars: int = 1500, dpi: int = 300) -> str:
    pdf_path = os.path.normpath(pdf_path)  # <<< CRITICAL FIX
    """
    OCR first page at 300 DPI using pdftoppm + Tesseract.
    No pdf2image dependency.
    """
    tmp_dir = tempfile.mkdtemp(prefix="ocr_")
    try:
        out_prefix = os.path.join(tmp_dir, "page")

        # --- render first page to PNG (300 DPI) ---
        cmd = [
            PDFTOPPM_EXE,
            "-f", "1",
            "-l", "1",
            "-r", str(dpi),
            "-png",
            pdf_path,
            out_prefix
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Find generated PNG (usually page-1.png)
        pngs = glob.glob(os.path.join(tmp_dir, "page-*.png"))
        if not pngs:
            return ""

        img_path = pngs[0]

        # --- OCR ---
        img = Image.open(img_path)

        text = pytesseract.image_to_string(
            img,                              # ← FIXED (was undefined "image")
            lang="pol+eng",
        )

        return text[:max_chars]

    except Exception as e:
        log_exception(e)
        return ""

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def choose_party(meta: dict):
    """Return opposing party if Raiffeisen is involved; otherwise plaintiff or defendant."""
    blacklist = ["raiffeisen", "raiffeisen bank", "raiffeisen bank international"]

    pl = meta.get("plaintiff", []) or []
    df = meta.get("defendant", []) or []

    def has_raiff(lst):
        return any(any(b in n.lower() for b in blacklist) for n in lst)

    if has_raiff(pl):
        return df if df else ["Unknown"]
    if has_raiff(df):
        return pl if pl else ["Unknown"]
    if pl:
        return pl
    if df:
        return df
    return ["Unknown"]

def normalize_parties(parties: list[str]) -> list[str]:
    """Surname first, preserve spaces, Polish letters."""
    out = []
    for p in parties:
        parts = p.strip().split()
        if len(parts) >= 2:
            surname = parts[-1]
            given = " ".join(parts[:-1])
            reordered = f"{surname} {given}" if FILENAME_RULES["surname_first"] else p.strip()
            out.append(reordered)
        else:
            out.append(p.strip())

    if FILENAME_RULES["max_parties"]:
        out = out[:FILENAME_RULES["max_parties"]]
    return out

def join_parties(parties: list[str]) -> str:
    """
    Join parties in EXACT human format:
    - comma-separated
    - spaces preserved
    """
    return ", ".join(parties) if parties else "UNKNOWN"

def format_case_numbers(cases: list[str]) -> str:
    """
    First case normally, others in parentheses.
    """
    if not cases:
        return "UNKNOWN"

    if len(cases) == 1:
        return cases[0]

    first = cases[0]
    rest = ", ".join(cases[1:])
    return f"{first} ({rest})"


def sanitize_filename_human(name: str) -> str:
    """
    Human rules:
    - keep spaces
    - replace ONLY forbidden Windows chars
    - replace '/' with '_'
    """
    name = name.replace("/", "_")
    return re.sub(r'[<>:"\\|?*]', "", name)


def normalize_target_filename(name: str) -> str:
    """Normalize a user-provided filename before saving.

    - Strip whitespace
    - Remove Windows-forbidden characters
    - Ensure the name ends with .pdf (case-insensitive)
    """
    cleaned = sanitize_filename_human(name.strip())
    if cleaned and not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned


def parse_json_content(content: str, source: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        snippet = (content or "").strip()[:120]
        raise ValueError(
            f"{source} did not return valid JSON. Received: '{snippet or 'empty response'}'"
        ) from e


def extract_metadata_openai(text: str, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content or ""
    return parse_json_content(content, "OpenAI chat response")


def extract_metadata_ollama(text: str, prompt: str) -> dict:
    payload = {
        "model": "llama3.1",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            "Unable to reach Ollama at http://127.0.0.1:11434. Start the Ollama server or switch AI backend."
        ) from e

    data = resp.json()
    content = data.get("message", {}).get("content", "{}")
    return parse_json_content(content, "Ollama chat response")


def extract_metadata(text: str) -> dict:
    prompt = build_prompt(text)
    backend = AI_BACKEND
    last_error = None
    if backend in ("ollama", "auto"):
        try:
            return extract_metadata_ollama(text, prompt)
        except Exception as e:
            last_error = e
            if backend == "ollama":
                raise
    if backend in ("openai", "auto"):
        try:
            return extract_metadata_openai(text, prompt)
        except Exception as e:
            last_error = e
    if last_error:
        raise last_error
    return {}

def select_parties(meta: dict, options: NamingOptions) -> list[str]:
    if options.party_mode == "plaintiff":
        return meta.get("plaintiff", []) or ["Unknown"]
    if options.party_mode == "defendant":
        return meta.get("defendant", []) or ["Unknown"]
    return choose_party(meta)


def build_filename(meta: dict, options: NamingOptions) -> str:
    parties = select_parties(meta, options) if options.include_parties else []
    cases = meta.get("case_numbers", []) if options.include_case_numbers else []
    letter_type = meta.get("letter_type", "unknown") if options.include_letter_type else ""

    segments = []

    if parties:
        normalized = normalize_parties(parties)
        segments.append(join_parties(normalized))

    if cases:
        segments.append(format_case_numbers(cases))

    if letter_type:
        lt = (letter_type or "unknown").strip()
        segments.append(lt)

    if not segments:
        segments.append("UNNAMED")

    filename = " - ".join(segments) + ".pdf"
    return sanitize_filename_human(filename)

# ==========================================================
# GUI
# ==========================================================

class FileProcessWorker(QThread):
    finished = pyqtSignal(int, dict)
    failed = pyqtSignal(int, Exception)

    def __init__(
        self,
        index: int,
        pdf_path: str,
        options: NamingOptions,
        prompt_flags: dict,
        selected_letter_type: str,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.index = index
        self.pdf_path = pdf_path
        self.options = options
        self.prompt_flags = prompt_flags or {}
        self.selected_letter_type = selected_letter_type
        self.stop_event = stop_event

    def run(self):
        try:
            if self.stop_event.is_set():
                return
            ocr_text = extract_text_ocr(
                self.pdf_path,
                self.options.ocr_char_limit,
                self.options.ocr_dpi,
            ) if self.options.ocr_enabled else ""

            if self.stop_event.is_set():
                return

            set_prompt_context(self.options.prompt_template, self.prompt_flags)
            meta = extract_metadata(ocr_text) if ocr_text else {
                "plaintiff": [],
                "defendant": [],
                "case_numbers": [],
                "letter_type": "unknown",
            }

            if self.selected_letter_type:
                meta["letter_type"] = self.selected_letter_type

            party = select_parties(meta, self.options)
            cases = meta.get("case_numbers", [])
            lt = meta.get("letter_type", "unknown")

            if self.stop_event.is_set():
                return

            filename = build_filename(meta, self.options)

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

class RenamerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI PDF Renamer")
        self.setGeometry(200, 200, 900, 700)

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

        # ---------- Layout ----------
        root_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.settings_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.settings_tab, "AI & Filename Settings")
        root_layout.addWidget(self.tabs)

        self.main_layout = QVBoxLayout()
        self.main_tab.setLayout(self.main_layout)
        self.settings_layout = QVBoxLayout()
        self.settings_tab.setLayout(self.settings_layout)

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

        # Type selector
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("PDF type:"))
        self.type_box = QComboBox()
        self.type_box.addItems([
            "Pozew",
            "Kontrpozew",
            "Pozew + Postanowienie",
            "Postanowienie",
            "Portal",
            "Korespondencja"
        ])
        h3.addWidget(self.type_box)
        self.main_layout.addLayout(h3)

        # OCR options
        h3b = QHBoxLayout()
        self.run_ocr_checkbox = QCheckBox("Run OCR")
        self.run_ocr_checkbox.setChecked(True)
        h3b.addWidget(self.run_ocr_checkbox)

        h3b.addWidget(QLabel("Max characters:"))
        self.char_limit_spin = QSpinBox()
        self.char_limit_spin.setRange(100, 10000)
        self.char_limit_spin.setSingleStep(100)
        self.char_limit_spin.setValue(1500)
        h3b.addWidget(self.char_limit_spin)

        h3b.addWidget(QLabel("OCR DPI:"))
        self.ocr_dpi_spin = QSpinBox()
        self.ocr_dpi_spin.setRange(72, 600)
        self.ocr_dpi_spin.setValue(300)
        h3b.addWidget(self.ocr_dpi_spin)

        self.char_count_label = QLabel("Characters retrieved: 0")
        h3b.addWidget(self.char_count_label)

        self.show_ocr_cb = QCheckBox("Show OCR preview")
        self.show_ocr_cb.setChecked(False)
        self.show_ocr_cb.toggled.connect(self.update_ocr_visibility)
        h3b.addWidget(self.show_ocr_cb)
        self.settings_layout.addLayout(h3b)

        # Party preference
        h3d = QHBoxLayout()
        h3d.addWidget(QLabel("Party to use:"))
        self.party_opp_rb = QRadioButton("Opposing side only")
        self.party_plaintiff_rb = QRadioButton("Plaintiff only")
        self.party_defendant_rb = QRadioButton("Defendant only")
        self.party_opp_rb.setChecked(True)
        h3d.addWidget(self.party_opp_rb)
        h3d.addWidget(self.party_plaintiff_rb)
        h3d.addWidget(self.party_defendant_rb)
        h3d.addStretch()
        self.settings_layout.addLayout(h3d)

        # Filename components
        h3c = QHBoxLayout()
        h3c.addWidget(QLabel("Include in filename:"))
        self.include_parties_cb = QCheckBox("Plaintiff/Defendant")
        self.include_parties_cb.setChecked(True)
        self.include_cases_cb = QCheckBox("Case numbers")
        self.include_cases_cb.setChecked(True)
        self.include_letter_cb = QCheckBox("Letter type")
        self.include_letter_cb.setChecked(True)

        h3c.addWidget(self.include_parties_cb)
        h3c.addWidget(self.include_cases_cb)
        h3c.addWidget(self.include_letter_cb)
        self.settings_layout.addLayout(h3c)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("AI Engine:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "OpenAI (cloud)",
            "Local AI (Ollama)",
            "Auto (Local → Cloud)",
        ])
        backend_row.addWidget(self.backend_combo)
        backend_row.addStretch()
        self.settings_layout.addLayout(backend_row)

        self.settings_layout.addWidget(QLabel("AI Prompt Template"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(DEFAULT_PROMPT)
        self.settings_layout.addWidget(self.prompt_edit)

        flags_row = QHBoxLayout()
        self.flag_plaintiff_cb = QCheckBox("Extract plaintiffs")
        self.flag_defendant_cb = QCheckBox("Extract defendants")
        self.flag_cases_cb = QCheckBox("Extract case numbers")
        self.flag_letter_cb = QCheckBox("Extract letter type")
        self.flag_raiff_cb = QCheckBox("Apply Raiffeisen exclusion rule")
        for cb in [
            self.flag_plaintiff_cb,
            self.flag_defendant_cb,
            self.flag_cases_cb,
            self.flag_letter_cb,
            self.flag_raiff_cb,
        ]:
            cb.setChecked(True)
            flags_row.addWidget(cb)
        self.settings_layout.addLayout(flags_row)

        reset_row = QHBoxLayout()
        reset_btn = QPushButton("Reset prompt to default")
        reset_btn.clicked.connect(self.reset_prompt)
        reset_row.addWidget(reset_btn)
        reset_row.addStretch()
        self.settings_layout.addLayout(reset_row)
        
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

        # OCR preview
        self.ocr_label = QLabel("OCR Preview:")
        self.ocr_label.setVisible(False)
        self.main_layout.addWidget(self.ocr_label)
        self.ocr_view = QTextEdit()
        self.ocr_view.setReadOnly(True)
        self.ocr_view.setVisible(False)
        self.main_layout.addWidget(self.ocr_view)

        # Filename editing
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Proposed filename:"))
        self.filename_edit = QLineEdit()
        self.filename_edit.editingFinished.connect(self.update_filename_for_current_row)
        h4.addWidget(self.filename_edit)
        self.main_layout.addLayout(h4)

        play_row = QHBoxLayout()
        self.play_button = QPushButton("▶ Play (Generate Proposals)")
        self.play_button.setStyleSheet("font-size: 16px; padding: 12px; font-weight: bold;")
        self.play_button.clicked.connect(self.start_processing_clicked)
        play_row.addStretch()
        play_row.addWidget(self.play_button)
        play_row.addStretch()
        self.main_layout.addLayout(play_row)

        # Buttons row
        h5 = QHBoxLayout()
        btn_next = QPushButton("Next File")
        btn_next.clicked.connect(self.next_file)
        self.btn_next = btn_next

        btn_process = QPushButton("Process This File")
        btn_process.clicked.connect(self.process_this_file)
        self.btn_process = btn_process

        btn_all = QPushButton("Process All")
        btn_all.clicked.connect(self.process_all_files_safe)
        self.btn_all = btn_all

        btn_stop = QPushButton("Stop && Reprocess")
        btn_stop.clicked.connect(self.stop_and_reprocess)
        self.btn_stop = btn_stop

        btn_quit = QPushButton("Quit")
        btn_quit.clicked.connect(self.close)

        h5.addWidget(btn_next)
        h5.addWidget(btn_process)
        h5.addWidget(btn_all)
        h5.addWidget(btn_stop)
        h5.addWidget(btn_quit)
        self.main_layout.addLayout(h5)

        self.setLayout(root_layout)

        self.processing_enabled = False

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
            QMessageBox.critical(
                self,
                "Unhandled error",
                f"An error occurred while selecting input folder.\n\n"
                f"Details were written to:\n{LOG_FILE}"
            )

    def choose_output(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if not folder:
                return
            self.output_edit.setText(folder)
        except Exception as e:
            log_exception(e)
            QMessageBox.critical(
                self,
                "Unhandled error",
                f"An error occurred while selecting output folder.\n\n"
                f"Details were written to:\n{LOG_FILE}"
            )

    # ------------------------------------------------------
    # Load PDFs
    # ------------------------------------------------------

    def load_pdfs(self):
        folder = self.input_edit.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Error", "Input folder does not exist.")
            return

        self.stop_event.clear()
        self.pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
        self.current_index = 0
        self.file_results.clear()
        self.active_workers.clear()
        self.failed_indices.clear()

        self.processing_enabled = False

        self.file_table.setRowCount(0)
        for idx, filename in enumerate(self.pdf_files):
            self.file_table.insertRow(idx)
            self.file_table.setItem(idx, 0, QTableWidgetItem(filename))
            self.file_table.setItem(idx, 1, QTableWidgetItem(filename))

        if not self.pdf_files:
            QMessageBox.information(self, "Info", "No PDFs found in folder.")
            return

        self.file_table.selectRow(0)

    def reset_prompt(self):
        self.prompt_edit.setPlainText(DEFAULT_PROMPT)
        for cb in [
            self.flag_plaintiff_cb,
            self.flag_defendant_cb,
            self.flag_cases_cb,
            self.flag_letter_cb,
            self.flag_raiff_cb,
        ]:
            cb.setChecked(True)

    def get_prompt_flags(self) -> dict:
        return {
            "plaintiff": self.flag_plaintiff_cb.isChecked(),
            "defendant": self.flag_defendant_cb.isChecked(),
            "case_numbers": self.flag_cases_cb.isChecked(),
            "letter_type": self.flag_letter_cb.isChecked(),
            "raiffeisen_rule": self.flag_raiff_cb.isChecked(),
        }

    def get_party_mode(self) -> str:
        if self.party_plaintiff_rb.isChecked():
            return "plaintiff"
        if self.party_defendant_rb.isChecked():
            return "defendant"
        return "opposing"

    def get_ai_backend(self) -> str:
        idx = self.backend_combo.currentIndex()
        if idx == 1:
            return "ollama"
        if idx == 2:
            return "auto"
        return "openai"

    def build_options(self) -> NamingOptions:
        return NamingOptions(
            include_parties=self.include_parties_cb.isChecked(),
            include_case_numbers=self.include_cases_cb.isChecked(),
            include_letter_type=self.include_letter_cb.isChecked(),
            party_mode=self.get_party_mode(),
            ocr_enabled=self.run_ocr_checkbox.isChecked(),
            ocr_char_limit=self.char_limit_spin.value(),
            ocr_dpi=self.ocr_dpi_spin.value(),
            prompt_template=self.prompt_edit.toPlainText(),
        )

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
        if not self.pdf_files:
            QMessageBox.information(self, "Info", "Select an input folder with PDFs first.")
            return

        self.stop_event.clear()
        self.failed_indices.clear()
        self.processing_enabled = True
        self.start_parallel_processing()

    def next_file(self):
        if not self.pdf_files or not self.processing_enabled:
            return

        next_index = (self.current_index + 1) % len(self.pdf_files)
        self.current_index = next_index
        self.process_current_file()

    def stop_and_reprocess(self):
        self.stop_event.set()

        for worker in list(self.active_workers.values()):
            worker.requestInterruption()

        self.active_workers.clear()
        self.file_results.clear()
        self.failed_indices.clear()

        self.ocr_text = ""
        self.meta = {}
        self.ocr_view.clear()
        self.filename_edit.clear()
        self.char_count_label.setText("Characters retrieved: 0")
        self.update_ocr_visibility()

        for row in range(self.file_table.rowCount()):
            current_name = self.file_table.item(row, 0).text()
            self.file_table.setItem(row, 1, QTableWidgetItem(current_name))

        self.stop_event.clear()
        self.current_index = 0
        self.processing_enabled = False
        if self.pdf_files:
            self.file_table.selectRow(0)

    def process_this_file(self):
        out_folder = self.output_edit.text()
        if not os.path.isdir(out_folder):
            QMessageBox.warning(self, "Error", "Output folder does not exist.")
            return

        if not self.pdf_files or (self.current_index in self.active_workers):
            return

        self.stop_event.clear()

        self.update_filename_for_current_row()

        pdf_name = self.pdf_files[self.current_index]
        inp = os.path.join(self.input_edit.text(), pdf_name)

        proposed = self.file_table.item(self.current_index, 1)
        target_name_raw = proposed.text() if proposed else self.filename_edit.text()
        target_name = normalize_target_filename(target_name_raw)
        if not target_name:
            QMessageBox.warning(self, "Error", "Proposed filename is empty.")
            return

        out = os.path.join(out_folder, target_name)

        try:
            shutil.move(inp, out)
            if self.current_index in self.file_results:
                self.file_results[self.current_index]["filename"] = target_name
            QMessageBox.information(self, "Done", f"Renamed:\n{out}")
        except Exception as e:
            log_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to rename file:\n{e}")

    def process_all_files_safe(self):
        out_folder = self.output_edit.text()
        if not os.path.isdir(out_folder):
            QMessageBox.warning(self, "Error", "Output folder does not exist.")
            return

        if not self.pdf_files:
            return

        self.stop_event.clear()
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
            except Exception as e:
                log_exception(e)
                QMessageBox.warning(
                    self,
                    "File error",
                    f"Error processing:\n{pdf_name}\n\nContinuing with next file."
                )
                continue

        QMessageBox.information(self, "Done", "All files processed.")

    def process_all(self):
        out_folder = self.output_edit.text()
        if not os.path.isdir(out_folder):
            QMessageBox.warning(self, "Error", "Output folder does not exist.")
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
        self.start_parallel_processing()

    def handle_worker_failed(self, index: int, error: Exception):
        self.active_workers.pop(index, None)
        if self.stop_event.is_set():
            return
        self.failed_indices.add(index)
        QMessageBox.critical(self, "Error", f"Failed processing file at index {index}: {error}")
        self.start_parallel_processing()

    def on_row_selected(self, row: int, _col: int):
        self.current_index = row

        if row in self.file_results:
            self.apply_cached_result(row, self.file_results[row])
        else:
            item = self.file_table.item(row, 1)
            current_name = item.text() if item else ""
            self.filename_edit.setText(current_name)
            if not self.processing_enabled:
                self.ocr_text = ""
                self.meta = {}
                self.ocr_view.clear()
                self.char_count_label.setText("Characters retrieved: 0")
                self.update_ocr_visibility()
            if self.processing_enabled:
                self.process_current_file()

    def pick_party_from_meta(self, meta: dict) -> list[str]:
        if self.party_plaintiff_rb.isChecked():
            return meta.get("plaintiff", []) or ["Unknown"]
        if self.party_defendant_rb.isChecked():
            return meta.get("defendant", []) or ["Unknown"]
        return choose_party(meta)

    def update_ocr_visibility(self):
        visible = self.show_ocr_cb.isChecked() and bool(self.ocr_text)
        self.ocr_label.setVisible(visible)
        self.ocr_view.setVisible(visible)

    def generate_result_for_index(self, index: int) -> dict:
        pdf = self.pdf_files[index]
        pdf_path = os.path.join(self.input_edit.text(), pdf)
        global AI_BACKEND
        AI_BACKEND = self.get_ai_backend()
        options = self.build_options()
        set_prompt_context(options.prompt_template, self.get_prompt_flags())
        ocr_text = extract_text_ocr(
            pdf_path,
            options.ocr_char_limit,
            options.ocr_dpi,
        ) if options.ocr_enabled else ""

        meta = extract_metadata(ocr_text) if ocr_text else {
            "plaintiff": [],
            "defendant": [],
            "case_numbers": [],
            "letter_type": "unknown",
        }

        if self.type_box.currentText():
            meta["letter_type"] = self.type_box.currentText()

        filename = build_filename(meta, options)

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

            self.ocr_view.setText(self.ocr_text)
            self.update_ocr_visibility()
            self.ocr_view.setVisible(bool(self.ocr_text))
            self.filename_edit.setText(cached.get("filename", ""))
            self.char_count_label.setText(f"Characters retrieved: {cached.get('char_count', 0)}")

        self.file_table.setItem(index, 1, QTableWidgetItem(cached.get("filename", "")))

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
        flags = self.get_prompt_flags()
        worker = FileProcessWorker(
            index=index,
            pdf_path=pdf_path,
            options=options,
            prompt_flags=flags,
            selected_letter_type=self.type_box.currentText(),
            stop_event=self.stop_event,
        )
        worker.finished.connect(self.handle_worker_finished)
        worker.failed.connect(self.handle_worker_failed)
        self.active_workers[index] = worker
        worker.start()

    def start_parallel_processing(self):
        if not self.processing_enabled:
            return
        if self.stop_event.is_set():
            return
        if not self.pdf_files:
            return

        for idx in range(len(self.pdf_files)):
            if len(self.active_workers) >= self.max_parallel_workers:
                break
            if idx in self.file_results or idx in self.active_workers or idx in self.failed_indices:
                continue
            pdf_path = os.path.join(self.input_edit.text(), self.pdf_files[idx])
            self.start_worker_for_index(idx, pdf_path)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RenamerGUI()
    gui.show()
    sys.exit(app.exec())
