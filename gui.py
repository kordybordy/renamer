import json
import os
import shutil
import threading
import traceback
from dataclasses import dataclass
from typing import Dict, List

import requests
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox,
    QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QListWidget, QListWidgetItem, QTextEdit, QProgressBar,
    QStatusBar, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings
from PyQt6.QtGui import QPixmap, QIcon

from ai_utils import extract_metadata_ai
from config import (
    ACCENT_COLOR,
    AI_BACKEND_DEFAULT,
    BACKGROUND_COLOR,
    BASE_DIR,
    BORDER_COLOR,
    DEFAULT_TEMPLATE_ELEMENTS,
    FILENAME_RULES,
    OLLAMA_HOST,
    PANEL_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from distribution_utils import CaseFolderInfo, DistributionManager
from logging_utils import append_distribution_log, log_exception, log_info
from ocr_utils import get_ocr_text
from settings_utils import load_settings, save_settings
from text_utils import (
    apply_meta_defaults,
    apply_party_order,
    build_filename,
    normalize_target_filename,
    normalize_polish,
    requirements_from_template,
)


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
            meta = apply_party_order(
                meta,
                plaintiff_surname_first=self.options.plaintiff_surname_first,
                defendant_surname_first=self.options.defendant_surname_first,
            )

            if defaults_applied:
                log_info(
                    f"[Worker {self.index + 1}] Applied defaults for missing fields: {', '.join(defaults_applied)}"
                )
            log_info(
                f"[Worker {self.index + 1}] Extracted meta: {json.dumps(meta, ensure_ascii=False)}"
            )

            if self.stop_event.is_set():
                return

            filename = build_filename(meta, self.options.template_elements)

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

                raw_meta = result.get("raw_meta") or result.get("meta") or {}
                log_lines.append(f"Raw defendant field value: {raw_meta.get('defendant')!r}")
                defendants = self.gui_ref.get_defendants_from_result(result)
                if defendants:
                    log_lines.append(f"Defendants: {', '.join(defendants)}")
                    primary_defendant = defendants[0]
                    for defendant in defendants:
                        tokens, surname = self.gui_ref.distribution_manager._defendant_tokens(defendant)
                        log_lines.append(
                            f"Normalized defendant '{defendant}': tokens={tokens}, surname={surname}"
                        )
                else:
                    log_lines.append("Defendants: none detected (parsed list is empty)")
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
                    candidate_surnames = [
                        self.gui_ref.distribution_manager._defendant_tokens(name)[1]
                        for name in defendants
                        if name
                    ]
                    log_lines.append(
                        f"Status: no matching case folder found (checked {len(self.case_index)} folders; surnames tried={candidate_surnames})"
                    )

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

        self.preview_value = QLabel("—")
        self.preview_value.setStyleSheet("font-weight: 600;")

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
        title_col.addStretch()
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

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Input folder:"))
        self.input_edit = QLineEdit()
        h1.addWidget(self.input_edit)
        btn_input = QPushButton("Browse")
        btn_input.clicked.connect(self.choose_input)
        h1.addWidget(btn_input)
        self.main_layout.addLayout(h1)

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
        backend_default = AI_BACKEND_DEFAULT.lower()
        if backend_default == "openai":
            self.backend_combo.setCurrentIndex(0)
        elif backend_default == "auto":
            self.backend_combo.setCurrentIndex(2)
        else:
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

        load_settings(self)
        self.ui_ready = True
        self.update_preview()
        self.check_ollama_status()

    def closeEvent(self, event):
        save_settings(self)
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
            resp = requests.get(f"{OLLAMA_HOST}api/tags", timeout=2)
            ok = resp.status_code == 200
        except Exception:
            ok = False
        if ok:
            self.ollama_badge.setText("🟢 Connected")
            self.ollama_badge.setStyleSheet("color: #7CFC00;")
        else:
            self.ollama_badge.setText("🔴 Offline")
            self.ollama_badge.setStyleSheet("color: #FF6B6B;")

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
        meta = apply_party_order(
            meta,
            plaintiff_surname_first=options.plaintiff_surname_first,
            defendant_surname_first=options.defendant_surname_first,
        )
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
        meta = apply_party_order(
            meta,
            plaintiff_surname_first=options.plaintiff_surname_first,
            defendant_surname_first=options.defendant_surname_first,
        )
        filename = build_filename(meta, options.template_elements)
        display_name = filename or "—"
        self.preview_value.setText(display_name)
        if filename:
            self.filename_edit.blockSignals(True)
            self.filename_edit.setText(filename)
            self.filename_edit.blockSignals(False)
            if self.current_index < self.file_table.rowCount():
                self.file_table.setItem(self.current_index, 1, QTableWidgetItem(filename))

    def process_current_file(self):
        if not self.pdf_files or not self.processing_enabled:
            return

        folder = self.input_edit.text()
        pdf = self.pdf_files[self.current_index]
        pdf_path = os.path.join(folder, pdf)
        self.start_worker_for_index(self.current_index, pdf_path)

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
        self.start_processing_ui("Copying current file…", total=1)

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
            shutil.copy2(inp, out)
            if self.current_index in self.file_results:
                self.file_results[self.current_index]["filename"] = target_name
            self.update_processing_progress(total=1, processed_override=1)
            QMessageBox.information(self, "Done", f"Copied to:\n{out}")
        except Exception as e:
            log_exception(e)
            show_friendly_error(
                self,
                "Copy failed",
                "Renamer could not copy the file to the output folder.",
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
        self.start_processing_ui("Copying all files…", total=len(self.pdf_files))
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
                shutil.copy2(inp_path, out_path)
                self.update_processing_progress(
                    total=len(self.pdf_files), processed_override=idx + 1
                )
            except Exception as e:
                log_exception(e)
                show_friendly_error(
                    self,
                    "File error",
                    "Renamer hit a problem while copying one of the files.",
                    traceback.format_exc(),
                    icon=QMessageBox.Icon.Warning,
                )
                continue

        QMessageBox.information(self, "Done", "All files copied.")
        self.stop_processing_ui("Idle")

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
        options = self.build_options()
        requirements = requirements_from_template(options.template_elements)
        self.log_activity(
            f"[UI] Starting OCR for '{pdf}' (pages={options.ocr_pages}, dpi={options.ocr_dpi}, "
            f"char_limit={options.ocr_char_limit}, backend={self.get_ai_backend()})"
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
        meta = apply_party_order(
            meta,
            plaintiff_surname_first=options.plaintiff_surname_first,
            defendant_surname_first=options.defendant_surname_first,
        )

        if defaults_applied:
            self.log_activity(
                f"[UI] Applied defaults for missing fields: {', '.join(defaults_applied)}"
            )
        self.log_activity(f"[UI] Extracted meta: {json.dumps(meta, ensure_ascii=False)}")

        filename = build_filename(meta, options.template_elements)

        self.log_activity(f"[UI] Proposed filename: {filename} (backend={self.get_ai_backend()})")

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

    def start_worker_for_index(self, index: int, pdf_path: str):
        if self.stop_event.is_set():
            return
        if index in self.failed_indices:
            return
        if index in self.active_workers or index in self.file_results:
            return

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
