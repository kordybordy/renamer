 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/main.py b/main.py
index fa9daad5b8dca10510817f58291a604c18819c8d..23d1b350a3276ae4d869f7da9ab99ac60c657d29 100644
--- a/main.py
+++ b/main.py
@@ -5,92 +5,191 @@
 
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
 from urllib.parse import urljoin
 from dataclasses import dataclass
 
 from PIL import Image
 import pytesseract
 
 from PyQt6.QtWidgets import (
     QApplication, QWidget, QPushButton, QLabel, QLineEdit,
     QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox,
     QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
-    QTabWidget, QListWidget, QListWidgetItem
+    QTabWidget, QListWidget, QListWidgetItem, QTextEdit, QProgressBar,
+    QStatusBar, QAbstractItemView
 )
-from PyQt6.QtCore import Qt, QThread, pyqtSignal
+from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
+from PyQt6.QtGui import QPixmap, QIcon
 
 AI_BACKEND = os.environ.get("AI_BACKEND", "openai")  # openai | ollama | auto
 OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434/")
 
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
 
+ACCENT_COLOR = "#4F7CFF"
+BACKGROUND_COLOR = "#1E1E1E"
+PANEL_COLOR = "#252526"
+TEXT_PRIMARY = "#FFFFFF"
+TEXT_SECONDARY = "#B0B0B0"
+BORDER_COLOR = "#333333"
+
+GLOBAL_STYLESHEET = f"""
+* {{
+    font-family: 'Segoe UI', sans-serif;
+    color: {TEXT_PRIMARY};
+}}
+
+QWidget {{
+    background-color: {BACKGROUND_COLOR};
+}}
+
+QLineEdit, QComboBox, QListWidget, QTableWidget, QTextEdit, QSpinBox {{
+    background-color: {PANEL_COLOR};
+    border: 1px solid {BORDER_COLOR};
+    border-radius: 6px;
+    padding: 6px;
+    color: {TEXT_PRIMARY};
+}}
+
+QLabel {{
+    color: {TEXT_PRIMARY};
+}}
+
+QTabWidget::pane {{
+    border: 1px solid {BORDER_COLOR};
+    background: {PANEL_COLOR};
+    border-radius: 10px;
+    padding: 6px;
+}}
+
+QTabBar::tab {{
+    background: {PANEL_COLOR};
+    border: 1px solid {BORDER_COLOR};
+    border-bottom: none;
+    padding: 8px 16px;
+    border-top-left-radius: 10px;
+    border-top-right-radius: 10px;
+    margin-right: 4px;
+}}
+
+QTabBar::tab:selected {{
+    background: {ACCENT_COLOR};
+    color: {TEXT_PRIMARY};
+}}
+
+QTabBar::tab:hover {{
+    border-color: {ACCENT_COLOR};
+}}
+
+QPushButton {{
+    background-color: {ACCENT_COLOR};
+    border: 1px solid {ACCENT_COLOR};
+    color: {TEXT_PRIMARY};
+    padding: 10px 14px;
+    border-radius: 8px;
+    font-weight: 600;
+}}
+
+QPushButton:hover {{
+    box-shadow: 0 0 8px {ACCENT_COLOR};
+}}
+
+QPushButton:disabled {{
+    background-color: {BORDER_COLOR};
+    border-color: {BORDER_COLOR};
+    color: {TEXT_SECONDARY};
+}}
+
+QProgressBar {{
+    background: {PANEL_COLOR};
+    border: 1px solid {BORDER_COLOR};
+    border-radius: 6px;
+    text-align: center;
+}}
+
+QProgressBar::chunk {{
+    background-color: {ACCENT_COLOR};
+    border-radius: 6px;
+}}
+"""
+
 def log_exception(e: Exception):
     with open(LOG_FILE, "a", encoding="utf-8") as f:
         f.write("\n" + "=" * 60 + "\n")
         f.write(datetime.now().isoformat() + "\n")
         f.write(str(e) + "\n")
         f.write(traceback.format_exc())
         f.flush()
         os.fsync(f.fileno())
 
+
+def show_friendly_error(parent: QWidget, title: str, friendly: str, details: str):
+    box = QMessageBox(parent)
+    box.setWindowTitle(title)
+    box.setText(friendly)
+    box.setInformativeText("Details were written to the log file.")
+    box.setDetailedText(details)
+    box.setIcon(QMessageBox.Icon.Critical)
+    box.exec()
+
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
 
@@ -522,632 +621,835 @@ class FileProcessWorker(QThread):
 
             meta = extract_metadata(ocr_text, self.requirements) if ocr_text else {}
             meta = apply_meta_defaults(meta, self.requirements)
 
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
-        self.setWindowTitle("AI PDF Renamer")
-        self.setGeometry(200, 200, 900, 700)
+        self.setWindowTitle("Renamer")
+        self.setGeometry(200, 200, 1000, 760)
+
+        self.settings = QSettings("Renamer", "Renamer")
 
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
+
+        header = QHBoxLayout()
+        logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
+        pixmap = QPixmap(logo_path)
+        if not pixmap.isNull():
+            logo_label = QLabel()
+            logo_label.setPixmap(pixmap.scaled(QSize(40, 40), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
+            header.addWidget(logo_label)
+        title_col = QVBoxLayout()
+        title_label = QLabel("Renamer")
+        title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
+        subtitle_label = QLabel("Smart document naming")
+        subtitle_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
+        title_col.addWidget(title_label)
+        title_col.addWidget(subtitle_label)
+        header.addLayout(title_col)
+        header.addStretch()
+        root_layout.addLayout(header)
+
         self.tabs = QTabWidget()
         self.main_tab = QWidget()
         self.settings_tab = QWidget()
         self.tabs.addTab(self.main_tab, "Main")
-        self.tabs.addTab(self.settings_tab, "AI & Filename Settings")
+        self.tabs.addTab(self.settings_tab, "AI Filename Settings")
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
 
         # OCR options
         h3b = QHBoxLayout()
         self.run_ocr_checkbox = QCheckBox("Run OCR")
         self.run_ocr_checkbox.setChecked(True)
+        self.run_ocr_checkbox.toggled.connect(self.update_preview)
         h3b.addWidget(self.run_ocr_checkbox)
 
         h3b.addWidget(QLabel("Max characters:"))
         self.char_limit_spin = QSpinBox()
         self.char_limit_spin.setRange(100, 10000)
         self.char_limit_spin.setSingleStep(100)
         self.char_limit_spin.setValue(1500)
+        self.char_limit_spin.valueChanged.connect(self.update_preview)
         h3b.addWidget(self.char_limit_spin)
 
         h3b.addWidget(QLabel("OCR DPI:"))
         self.ocr_dpi_spin = QSpinBox()
         self.ocr_dpi_spin.setRange(72, 600)
         self.ocr_dpi_spin.setValue(300)
+        self.ocr_dpi_spin.valueChanged.connect(self.update_preview)
         h3b.addWidget(self.ocr_dpi_spin)
 
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
+        self.template_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
+        self.template_list.setDefaultDropAction(Qt.DropAction.MoveAction)
+        self.template_list.model().rowsMoved.connect(lambda *_: self.update_preview())
         for element in DEFAULT_TEMPLATE_ELEMENTS:
-            self.add_template_item(element)
+            self.add_template_item(element, refresh=False)
+        self.update_preview()
         list_col.addWidget(self.template_list)
 
         buttons_col = QVBoxLayout()
-        move_up_btn = QPushButton("Move up")
-        move_up_btn.clicked.connect(self.move_template_element_up)
-        move_down_btn = QPushButton("Move down")
-        move_down_btn.clicked.connect(self.move_template_element_down)
         remove_btn = QPushButton("Remove")
         remove_btn.clicked.connect(self.remove_selected_template_element)
 
-        for btn in (move_up_btn, move_down_btn, remove_btn):
-            buttons_col.addWidget(btn)
+        buttons_col.addWidget(remove_btn)
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
+        self.backend_combo.setToolTip("OpenAI = cloud (cost). Ollama = local (free). Auto tries local then cloud.")
+        self.backend_combo.currentIndexChanged.connect(self.check_ollama_status)
         backend_row.addWidget(self.backend_combo)
+        self.ollama_badge = QLabel("")
+        backend_row.addWidget(self.ollama_badge)
         backend_row.addStretch()
         self.settings_layout.addLayout(backend_row)
         
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
 
+        preview_row = QHBoxLayout()
+        preview_label = QLabel("Live preview:")
+        preview_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
+        preview_row.addWidget(preview_label)
+        self.preview_value = QLabel("—")
+        self.preview_value.setStyleSheet("font-weight: 600;")
+        preview_row.addWidget(self.preview_value)
+        preview_row.addStretch()
+        self.main_layout.addLayout(preview_row)
+
         play_row = QHBoxLayout()
-        self.play_button = QPushButton("▶ Play (Generate Proposals)")
+        self.play_button = QPushButton("▶ Generate")
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
 
-        btn_process = QPushButton("Process This File")
+        btn_process = QPushButton("✎ Rename File")
         btn_process.clicked.connect(self.process_this_file)
         self.btn_process = btn_process
 
-        btn_all = QPushButton("Process All")
+        btn_all = QPushButton("⏩ Rename All")
         btn_all.clicked.connect(self.process_all_files_safe)
         self.btn_all = btn_all
 
         btn_quit = QPushButton("Quit")
         btn_quit.clicked.connect(self.close)
 
         h5.addWidget(btn_next)
         h5.addWidget(btn_process)
         h5.addWidget(btn_all)
         h5.addWidget(btn_quit)
         self.main_layout.addLayout(h5)
 
+        # Activity log panel
+        log_header = QHBoxLayout()
+        self.log_toggle = QPushButton("▼ What is happening now")
+        self.log_toggle.setCheckable(True)
+        self.log_toggle.setChecked(True)
+        self.log_toggle.clicked.connect(self.toggle_log)
+        self.log_toggle.setStyleSheet("text-align: left;")
+        log_header.addWidget(self.log_toggle)
+        log_header.addStretch()
+        self.main_layout.addLayout(log_header)
+
+        self.activity_log = QTextEdit()
+        self.activity_log.setReadOnly(True)
+        self.activity_log.setMaximumHeight(120)
+        self.main_layout.addWidget(self.activity_log)
+
+        # Status bar
+        self.status_bar = QStatusBar()
+        self.status_label = QLabel("Waiting for input…")
+        self.spinner_label = QLabel("")
+        self.progress_bar = QProgressBar()
+        self.progress_bar.setRange(0, 1)
+        self.progress_bar.setValue(0)
+        self.progress_bar.setTextVisible(False)
+        status_widget = QWidget()
+        status_layout = QHBoxLayout()
+        status_layout.setContentsMargins(0, 0, 0, 0)
+        status_layout.addWidget(self.spinner_label)
+        status_layout.addWidget(self.status_label)
+        status_layout.addStretch()
+        status_layout.addWidget(self.progress_bar)
+        status_widget.setLayout(status_layout)
+        self.status_bar.addPermanentWidget(status_widget, 1)
+        root_layout.addWidget(self.status_bar)
+
         self.setLayout(root_layout)
 
         self.processing_enabled = False
+        self.spinner_timer = QTimer(self)
+        self.spinner_timer.timeout.connect(self.animate_spinner)
+        self.spinner_state = 0
+        self.activity_entries: list[str] = []
+
+        self.load_settings()
+        self.update_preview()
+        self.check_ollama_status()
+
+    # ------------------------------------------------------
+    # UI helpers
+    # ------------------------------------------------------
+
+    def load_settings(self):
+        self.input_edit.setText(self.settings.value("input_folder", ""))
+        self.output_edit.setText(self.settings.value("output_folder", ""))
+        saved_template = self.settings.value("template", [])
+        if isinstance(saved_template, str):
+            saved_template = json.loads(saved_template) if saved_template else []
+        if saved_template:
+            self.template_list.clear()
+            for element in saved_template:
+                self.add_template_item(element)
+
+    def save_settings(self):
+        self.settings.setValue("input_folder", self.input_edit.text())
+        self.settings.setValue("output_folder", self.output_edit.text())
+        self.settings.setValue("template", self.get_template_elements())
+
+    def closeEvent(self, event):
+        self.save_settings()
+        super().closeEvent(event)
+
+    def toggle_log(self):
+        visible = not self.activity_log.isVisible()
+        self.activity_log.setVisible(visible)
+        self.log_toggle.setText(("▼" if visible else "▶") + " What is happening now")
+
+    def log_activity(self, message: str):
+        self.activity_entries.append(message)
+        self.activity_entries = self.activity_entries[-10:]
+        self.activity_log.setText("\n".join(self.activity_entries))
+
+    def set_status(self, text: str):
+        if not text:
+            text = "Working…"
+        self.status_label.setText(text)
+
+    def animate_spinner(self):
+        dots = "." * (self.spinner_state % 4)
+        self.spinner_label.setText(f"⏳{dots}")
+        self.spinner_state += 1
+
+    def start_processing_ui(self, status: str = "Processing…"):
+        self.set_status(status)
+        self.progress_bar.setRange(0, 0)
+        self.spinner_timer.start(300)
+        for btn in (self.play_button, self.btn_process, self.btn_all, self.btn_next):
+            btn.setDisabled(True)
+
+    def stop_processing_ui(self, status: str = "Idle"):
+        self.set_status(status)
+        self.progress_bar.setRange(0, 1)
+        self.progress_bar.setValue(0)
+        self.spinner_timer.stop()
+        self.spinner_label.setText("")
+        for btn in (self.play_button, self.btn_process, self.btn_all, self.btn_next):
+            btn.setDisabled(False)
+
+    def check_ollama_status(self):
+        if self.backend_combo.currentIndex() != 1:
+            self.ollama_badge.setText("")
+            return
+        try:
+            resp = requests.get(urljoin(OLLAMA_HOST, "api/tags"), timeout=2)
+            ok = resp.status_code == 200
+        except Exception:
+            ok = False
+        if ok:
+            self.ollama_badge.setText("🟢 Connected")
+            self.ollama_badge.setStyleSheet("color: #7CFC00;")
+        else:
+            self.ollama_badge.setText("🔴 Offline")
+            self.ollama_badge.setStyleSheet("color: #FF6B6B;")
 
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
-            QMessageBox.critical(
+            show_friendly_error(
                 self,
-                "Unhandled error",
-                f"An error occurred while selecting input folder.\n\n"
-                f"Details were written to:\n{LOG_FILE}"
+                "Folder error",
+                "Renamer could not open the selected input folder.",
+                traceback.format_exc(),
             )
 
     def choose_output(self):
         try:
             folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
             if not folder:
                 return
             self.output_edit.setText(folder)
         except Exception as e:
             log_exception(e)
-            QMessageBox.critical(
+            show_friendly_error(
                 self,
-                "Unhandled error",
-                f"An error occurred while selecting output folder.\n\n"
-                f"Details were written to:\n{LOG_FILE}"
+                "Folder error",
+                "Renamer could not open the selected output folder.",
+                traceback.format_exc(),
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
+        self.log_activity(f"Loaded {len(self.pdf_files)} PDF files")
+        self.set_status("Waiting for generate…")
 
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
         )
 
     def display_name_for_element(self, element: str) -> str:
         mapping = {
             "date": "Date (today)",
             "plaintiff": "Plaintiff",
             "defendant": "Defendant",
             "letter_type": "Letter type",
         }
         return mapping.get(element, element)
 
-    def add_template_item(self, element: str):
+    def add_template_item(self, element: str, refresh: bool = True):
         item = QListWidgetItem(self.display_name_for_element(element))
         item.setData(Qt.ItemDataRole.UserRole, element)
         self.template_list.addItem(item)
+        if refresh:
+            self.update_preview()
 
     def add_template_element(self):
         element = self.template_selector.currentData()
         if not element:
             return
         self.add_template_item(element)
 
     def remove_selected_template_element(self):
         row = self.template_list.currentRow()
         if row >= 0:
             self.template_list.takeItem(row)
+            self.update_preview()
 
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
-                self.add_template_item(element)
+                self.add_template_item(element, refresh=False)
 
         return elements
 
+    def update_preview(self):
+        options = self.build_options()
+        meta = self.meta or {}
+        if self.current_index in self.file_results:
+            meta = self.file_results[self.current_index].get("meta", meta)
+        requirements = requirements_from_template(options.template_elements)
+        meta = apply_meta_defaults(meta, requirements)
+        filename = build_filename(meta, options)
+        display_name = filename or "—"
+        self.preview_value.setText(display_name)
+        if filename:
+            self.filename_edit.blockSignals(True)
+            self.filename_edit.setText(filename)
+            self.filename_edit.blockSignals(False)
+            if self.current_index < self.file_table.rowCount():
+                self.file_table.setItem(self.current_index, 1, QTableWidgetItem(filename))
+
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
             QMessageBox.information(self, "Info", "Select an input folder with PDFs first.")
             return
 
         self.stop_event.clear()
         self.failed_indices.clear()
         self.processing_enabled = True
+        self.start_processing_ui("Generating proposals…")
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
         self.filename_edit.clear()
         self.char_count_label.setText("Characters retrieved: 0")
 
         for row in range(self.file_table.rowCount()):
             current_name = self.file_table.item(row, 0).text()
             self.file_table.setItem(row, 1, QTableWidgetItem(current_name))
 
         self.stop_event.clear()
         self.current_index = 0
         self.processing_enabled = False
         if self.pdf_files:
             self.file_table.selectRow(0)
+        self.stop_processing_ui("Reset")
 
     def process_this_file(self):
         out_folder = self.output_edit.text()
         if not os.path.isdir(out_folder):
             QMessageBox.warning(self, "Error", "Output folder does not exist.")
             return
 
         if not self.pdf_files or (self.current_index in self.active_workers):
             return
 
         self.stop_event.clear()
 
         self.update_filename_for_current_row()
+        self.start_processing_ui("Renaming current file…")
 
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
+        finally:
+            self.stop_processing_ui("Idle")
 
     def process_all_files_safe(self):
         out_folder = self.output_edit.text()
         if not os.path.isdir(out_folder):
             QMessageBox.warning(self, "Error", "Output folder does not exist.")
             return
 
         if not self.pdf_files:
             return
 
         self.stop_event.clear()
+        self.start_processing_ui("Renaming all files…")
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
+        self.stop_processing_ui("Idle")
 
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
+        self.log_activity(f"✓ Processed file {index + 1} of {len(self.pdf_files)}")
+        if not self.active_workers:
+            self.stop_processing_ui("Idle")
 
     def handle_worker_failed(self, index: int, error: Exception):
         self.active_workers.pop(index, None)
         if self.stop_event.is_set():
             return
         self.failed_indices.add(index)
         QMessageBox.critical(self, "Error", f"Failed processing file at index {index}: {error}")
         self.start_parallel_processing()
+        if not self.active_workers:
+            self.stop_processing_ui("Idle")
 
     def on_row_selected(self, row: int, _col: int):
         self.current_index = row
+        item = self.file_table.item(row, 0)
+        if item:
+            self.file_table.scrollToItem(item)
 
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
             if self.processing_enabled:
                 self.process_current_file()
+        self.update_preview()
 
     def generate_result_for_index(self, index: int) -> dict:
         pdf = self.pdf_files[index]
         pdf_path = os.path.join(self.input_edit.text(), pdf)
         global AI_BACKEND
         AI_BACKEND = self.get_ai_backend()
         options = self.build_options()
         requirements = requirements_from_template(options.template_elements)
         ocr_text = extract_text_ocr(
             pdf_path,
             options.ocr_char_limit,
             options.ocr_dpi,
         ) if options.ocr_enabled else ""
 
         meta = extract_metadata(ocr_text, requirements) if ocr_text else {}
         meta = apply_meta_defaults(meta, requirements)
 
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
             self.filename_edit.setText(cached.get("filename", ""))
             self.char_count_label.setText(f"Characters retrieved: {cached.get('char_count', 0)}")
 
         self.file_table.setItem(index, 1, QTableWidgetItem(cached.get("filename", "")))
+        self.update_preview()
 
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
         )
         worker.finished.connect(self.handle_worker_finished)
         worker.failed.connect(self.handle_worker_failed)
         self.active_workers[index] = worker
+        self.set_status(f"Running OCR (DPI {options.ocr_dpi}) for file {index + 1}…")
+        self.log_activity(f"→ Processing file {index + 1}")
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
+    app.setStyleSheet(GLOBAL_STYLESHEET)
+    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
+    if os.path.exists(logo_path):
+        app.setWindowIcon(QIcon(logo_path))
     gui = RenamerGUI()
+    if os.path.exists(logo_path):
+        gui.setWindowIcon(QIcon(logo_path))
     gui.show()
     sys.exit(app.exec())
 
EOF
)
