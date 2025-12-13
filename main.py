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

from PIL import Image
import pytesseract

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox,
    QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

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
# AI SYSTEM PROMPT — returns structured JSON
# ==========================================================

SYSTEM_PROMPT = """
Return strict JSON in this exact shape:

{
  "plaintiff": ["Name Surname", ...],
  "defendant": ["Name Surname", ...],
  "case_numbers": ["I C 1234/25", ...],
  "letter_type": "Pozew" | "Kontrpozew" | "Pozew + Postanowienie" |
                  "Postanowienie" | "Portal" | "Korespondencja" |
                  "Unknown"
}

Rules:
- Identify all parties. If Raiffeisen appears, the opposing side is the relevant one.
- Extract ALL case numbers.
- Preserve Polish letters.
- Infer letter type according to content.
- No commentary. Output JSON only.
"""

# ==========================================================
# Helpers
# ==========================================================

def sanitize_case_number(case: str) -> str:
    """Replace only '/' with '_' in case numbers."""
    return case.replace("/", "_")

def extract_text_ocr(pdf_path: str, max_chars: int = 1500) -> str:
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
            "-r", "300",
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

def call_model(text: str) -> str:
    """Try GPT-5-nano → fallback GPT-4.1-mini."""
    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
        )
        return resp.choices[0].message.content
    except Exception:
        pass

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content

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

def build_filename(
    parties: list[str],
    cases: list[str],
    letter_type: str,
    include_parties: bool = True,
    include_cases: bool = True,
    include_letter_type: bool = True,
    **_ignored,
) -> str:
    segments = []

    if include_parties:
        normalized = normalize_parties(parties)
        segments.append(join_parties(normalized))

    if include_cases:
        segments.append(format_case_numbers(cases))

    if include_letter_type:
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
        run_ocr: bool,
        char_limit: int,
        include_parties: bool,
        include_cases: bool,
        include_letter_type: bool,
        selected_letter_type: str,
        prefer_plaintiff: bool,
        prefer_defendant: bool,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.index = index
        self.pdf_path = pdf_path
        self.run_ocr = run_ocr
        self.char_limit = char_limit
        self.include_parties = include_parties
        self.include_cases = include_cases
        self.include_letter_type = include_letter_type
        self.selected_letter_type = selected_letter_type
        self.prefer_plaintiff = prefer_plaintiff
        self.prefer_defendant = prefer_defendant
        self.stop_event = stop_event

    def run(self):
        try:
            if self.stop_event.is_set():
                return
            ocr_text = extract_text_ocr(self.pdf_path, self.char_limit) if self.run_ocr else ""

            if self.stop_event.is_set():
                return

            raw = call_model(ocr_text) if ocr_text else "{}"
            try:
                meta = json.loads(raw)
            except Exception:
                meta = {
                    "plaintiff": [],
                    "defendant": [],
                    "case_numbers": [],
                    "letter_type": "Unknown",
                }

            if self.selected_letter_type:
                meta["letter_type"] = self.selected_letter_type

            if self.prefer_plaintiff:
                party = meta.get("plaintiff", []) or ["Unknown"]
            elif self.prefer_defendant:
                party = meta.get("defendant", []) or ["Unknown"]
            else:
                party = choose_party(meta)
            cases = meta.get("case_numbers", [])
            lt = meta.get("letter_type", "Unknown")

            if self.stop_event.is_set():
                return

            filename = build_filename(
                party,
                cases,
                lt,
                include_parties=self.include_parties,
                include_cases=self.include_cases,
                include_letter_type=self.include_letter_type,
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
        self.max_parallel_workers = 3
        self.stop_event = threading.Event()

        # ---------- Layout ----------
        layout = QVBoxLayout()

        # Input folder
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Input folder:"))
        self.input_edit = QLineEdit()
        h1.addWidget(self.input_edit)
        btn_input = QPushButton("Browse")
        btn_input.clicked.connect(self.choose_input)
        h1.addWidget(btn_input)
        layout.addLayout(h1)

        # Output folder
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Output folder:"))
        self.output_edit = QLineEdit()
        h2.addWidget(self.output_edit)
        btn_output = QPushButton("Browse")
        btn_output.clicked.connect(self.choose_output)
        h2.addWidget(btn_output)
        layout.addLayout(h2)

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
        layout.addLayout(h3)

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

        self.char_count_label = QLabel("Characters retrieved: 0")
        h3b.addWidget(self.char_count_label)

        self.show_ocr_cb = QCheckBox("Show OCR preview")
        self.show_ocr_cb.setChecked(False)
        self.show_ocr_cb.toggled.connect(self.update_ocr_visibility)
        h3b.addWidget(self.show_ocr_cb)
        layout.addLayout(h3b)

        # Party preference
        h3d = QHBoxLayout()
        h3d.addWidget(QLabel("Party to use:"))
        self.use_plaintiff_cb = QCheckBox("Plaintiff")
        self.use_defendant_cb = QCheckBox("Defendant")
        self.use_plaintiff_cb.toggled.connect(self.handle_party_toggle)
        self.use_defendant_cb.toggled.connect(self.handle_party_toggle)
        h3d.addWidget(self.use_plaintiff_cb)
        h3d.addWidget(self.use_defendant_cb)
        h3d.addStretch()
        layout.addLayout(h3d)

        # Filename components
        h3c = QHBoxLayout()
        h3c.addWidget(QLabel("Include in filename:"))
        self.include_parties_cb = QCheckBox("Parties")
        self.include_parties_cb.setChecked(True)
        self.include_cases_cb = QCheckBox("Case numbers")
        self.include_cases_cb.setChecked(True)
        self.include_letter_cb = QCheckBox("Letter type")
        self.include_letter_cb.setChecked(True)

        h3c.addWidget(self.include_parties_cb)
        h3c.addWidget(self.include_cases_cb)
        h3c.addWidget(self.include_letter_cb)
        layout.addLayout(h3c)
        
        layout.addWidget(QLabel("Files and proposed names:"))
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["PDF file", "Proposed filename"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.setSelectionBehavior(self.file_table.SelectionBehavior.SelectRows)
        self.file_table.setEditTriggers(self.file_table.EditTrigger.NoEditTriggers)
        self.file_table.cellClicked.connect(self.on_row_selected)
        layout.addWidget(self.file_table)

        # OCR preview
        self.ocr_label = QLabel("OCR Preview:")
        self.ocr_label.setVisible(False)
        layout.addWidget(self.ocr_label)
        self.ocr_view = QTextEdit()
        self.ocr_view.setReadOnly(True)
        self.ocr_view.setVisible(False)
        layout.addWidget(self.ocr_view)

        # Filename editing
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Proposed filename:"))
        self.filename_edit = QLineEdit()
        h4.addWidget(self.filename_edit)
        layout.addLayout(h4)

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
        layout.addLayout(h5)

        self.setLayout(layout)

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

        self.file_table.setRowCount(0)
        for idx, filename in enumerate(self.pdf_files):
            self.file_table.insertRow(idx)
            self.file_table.setItem(idx, 0, QTableWidgetItem(filename))
            self.file_table.setItem(idx, 1, QTableWidgetItem("Pending"))

        if not self.pdf_files:
            QMessageBox.information(self, "Info", "No PDFs found in folder.")
            return

        self.file_table.selectRow(0)
        self.start_parallel_processing()

    # ------------------------------------------------------
    # Process One File
    # ------------------------------------------------------

    def process_current_file(self):
        if not self.pdf_files:
            return

        folder = self.input_edit.text()
        pdf = self.pdf_files[self.current_index]
        pdf_path = os.path.join(folder, pdf)
        self.start_worker_for_index(self.current_index, pdf_path)

    # ------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------

    def next_file(self):
        if not self.pdf_files:
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

        self.ocr_text = ""
        self.meta = {}
        self.ocr_view.clear()
        self.filename_edit.clear()
        self.char_count_label.setText("Characters retrieved: 0")
        self.update_ocr_visibility()

        for row in range(self.file_table.rowCount()):
            self.file_table.setItem(row, 1, QTableWidgetItem("Pending"))

        self.stop_event.clear()
        self.current_index = 0
        if self.pdf_files:
            self.file_table.selectRow(0)
            self.start_parallel_processing()

    def process_this_file(self):
        out_folder = self.output_edit.text()
        if not os.path.isdir(out_folder):
            QMessageBox.warning(self, "Error", "Output folder does not exist.")
            return

        if not self.pdf_files or (self.current_index in self.active_workers):
            return

        self.stop_event.clear()

        pdf_name = self.pdf_files[self.current_index]
        inp = os.path.join(self.input_edit.text(), pdf_name)
        out = os.path.join(out_folder, self.filename_edit.text())

        shutil.move(inp, out)
        QMessageBox.information(self, "Done", f"Renamed:\n{out}")

    def process_all_files_safe(self):
        self.stop_event.clear()
        for idx, pdf_path in enumerate(self.pdf_files):
            try:
                result = self.file_results.get(idx)
                if result is None:
                    result = self.generate_result_for_index(idx)
                    self.file_results[idx] = result
                self.apply_cached_result(idx, result)
            except Exception as e:
                log_exception(e)
                QMessageBox.warning(
                    self,
                    "File error",
                    f"Error processing:\n{pdf_path}\n\nContinuing with next file."
                )
                continue

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
        QMessageBox.critical(self, "Error", f"Failed processing file at index {index}: {error}")
        self.start_parallel_processing()

    def on_row_selected(self, row: int, _col: int):
        if row in self.file_results:
            self.apply_cached_result(row, self.file_results[row])
        else:
            self.current_index = row
            self.process_current_file()

    def handle_party_toggle(self, _checked: bool):
        # keep preference mutually exclusive
        if self.sender() is self.use_plaintiff_cb and self.use_plaintiff_cb.isChecked():
            self.use_defendant_cb.setChecked(False)
        elif self.sender() is self.use_defendant_cb and self.use_defendant_cb.isChecked():
            self.use_plaintiff_cb.setChecked(False)

    def pick_party_from_meta(self, meta: dict) -> list[str]:
        if self.use_plaintiff_cb.isChecked():
            return meta.get("plaintiff", []) or ["Unknown"]
        if self.use_defendant_cb.isChecked():
            return meta.get("defendant", []) or ["Unknown"]
        return choose_party(meta)

    def update_ocr_visibility(self):
        visible = self.show_ocr_cb.isChecked() and bool(self.ocr_text)
        self.ocr_label.setVisible(visible)
        self.ocr_view.setVisible(visible)

    def generate_result_for_index(self, index: int) -> dict:
        pdf = self.pdf_files[index]
        pdf_path = os.path.join(self.input_edit.text(), pdf)
        ocr_text = extract_text_ocr(pdf_path, self.char_limit_spin.value()) if self.run_ocr_checkbox.isChecked() else ""

        raw = call_model(ocr_text) if ocr_text else "{}"
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {
                "plaintiff": [],
                "defendant": [],
                "case_numbers": [],
                "letter_type": "Unknown",
            }

        if self.type_box.currentText():
            meta["letter_type"] = self.type_box.currentText()

        party = self.pick_party_from_meta(meta)
        party = choose_party(meta)
        cases = meta.get("case_numbers", [])
        lt = meta.get("letter_type", "Unknown")

        filename = build_filename(
            party,
            cases,
            lt,
            include_parties=self.include_parties_cb.isChecked(),
            include_cases=self.include_cases_cb.isChecked(),
            include_letter_type=self.include_letter_cb.isChecked(),
        )

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

    # Background processing
    def start_worker_for_index(self, index: int, pdf_path: str):
        if self.stop_event.is_set():
            return
        if index in self.active_workers or index in self.file_results:
            return

        worker = FileProcessWorker(
            index=index,
            pdf_path=pdf_path,
            run_ocr=self.run_ocr_checkbox.isChecked(),
            char_limit=self.char_limit_spin.value(),
            include_parties=self.include_parties_cb.isChecked(),
            include_cases=self.include_cases_cb.isChecked(),
            include_letter_type=self.include_letter_cb.isChecked(),
            selected_letter_type=self.type_box.currentText(),
            prefer_plaintiff=self.use_plaintiff_cb.isChecked(),
            prefer_defendant=self.use_defendant_cb.isChecked(),
            stop_event=self.stop_event,
        )
        worker.finished.connect(self.handle_worker_finished)
        worker.failed.connect(self.handle_worker_failed)
        self.active_workers[index] = worker
        worker.start()

    def start_parallel_processing(self):
        if self.stop_event.is_set():
            return
        if not self.pdf_files:
            return

        for idx in range(len(self.pdf_files)):
            if len(self.active_workers) >= self.max_parallel_workers:
                break
            if idx in self.file_results or idx in self.active_workers:
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
