import json
import os
import re
import sys
import threading
from datetime import datetime
from typing import Dict, List

from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem, QTextEdit, QProgressBar, QSizePolicy,
    QStatusBar, QAbstractItemView, QStackedWidget, QFrame, QGroupBox,
    QButtonGroup, QPlainTextEdit, QToolButton, QScrollArea
)
from PyQt6.QtCore import Qt, QSize, QSettings
from PyQt6.QtGui import QPixmap, QIcon, QPalette, QColor

from distribution.engine import DistributionConfig
from distribution.models import DistributionPlanItem
from distribution.scorer import DEFAULT_STOPWORDS
from distribution.safety import resolve_destination_path, validate_distribution_paths

from ai_service import check_ollama_health, get_ollama_base_url
from app_ai import extract_metadata_ai
from app_constants import AI_BACKEND, DEFAULT_TEMPLATE_ELEMENTS, FILENAME_RULES
from app_logging import append_distribution_log, log_exception, log_info
from app_ocr import get_ocr_text
from app_path_utils import validate_path_set
from app_runtime import BASE_DIR
from app_style import load_stylesheet
from app_text_utils import (
    apply_meta_defaults,
    apply_party_order,
    build_filename,
    defendant_from_filename,
    normalize_target_filename,
    requirements_from_template,
)
from app_ui import show_friendly_error
from app_workers import (
    DistributionApplyWorker,
    DistributionPlanWorker,
    FileProcessWorker,
    NamingOptions,
)
from ui_components import Card, SidebarButton, TopBarIconButton


def log_filesystem_action(operation: str, source: str, destination: str, status: str) -> None:
    log_info(
        f"[FS] {operation} | status={status} | source={os.path.abspath(source)} | destination={os.path.abspath(destination)}"
    )


class RenamerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Renamer")
        self.settings = QSettings("Renamer", "Renamer")

        # State
        self.pdf_files = []
        self.current_index = 0
        self.ocr_text = ""
        self.meta = {}
        self.file_results: dict[int, dict] = {}
        self.active_workers: dict[int, FileProcessWorker] = {}
        self.failed_indices: set[int] = set()
        cpu_count = os.cpu_count() or 2
        self.max_parallel_workers = max(2, min(4, cpu_count))
        self.stop_event = threading.Event()
        self.ui_ready = False
        self.distribution_plan: list[DistributionPlanItem] = []
        self.distribution_plan_view: list[DistributionPlanItem] = []
        self.distribution_plan_worker: DistributionPlanWorker | None = None
        self.distribution_apply_worker: DistributionApplyWorker | None = None
        self.distribution_audit_log_path: str | None = None
        self.distribution_pause_event = threading.Event()
        self.distribution_pause_event.set()
        self.custom_elements: dict[str, str] = {}

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)
        central_widget.setLayout(root_layout)

        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("Sidebar")
        self.sidebar_frame.setFixedWidth(220)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(8)

        brand_frame = QFrame()
        brand_frame.setObjectName("SidebarBrand")
        brand_layout = QHBoxLayout()
        brand_layout.setContentsMargins(8, 8, 8, 8)
        brand_layout.setSpacing(8)
        logo_path = None
        for candidate in ("logo-32.png", "logo-square.png", "logo.png"):
            candidate_path = os.path.join(BASE_DIR, "assets", candidate)
            if os.path.exists(candidate_path):
                logo_path = candidate_path
                break
        pixmap = QPixmap(logo_path) if logo_path else QPixmap()
        if not pixmap.isNull():
            logo_label = QLabel()
            logo_label.setPixmap(
                pixmap.scaled(QSize(32, 32), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            brand_layout.addWidget(logo_label)
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_label = QLabel("Renamer")
        title_label.setObjectName("TitleLabel")
        subtitle_label = QLabel("Smart document naming")
        subtitle_label.setObjectName("Subtitle")
        title_col.addWidget(title_label)
        title_col.addWidget(subtitle_label)
        brand_layout.addLayout(title_col)
        brand_layout.addStretch()
        brand_frame.setLayout(brand_layout)
        sidebar_layout.addWidget(brand_frame)

        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.setExclusive(True)
        self.mode_buttons.idClicked.connect(self.on_mode_changed)

        icon_dir = os.path.join(BASE_DIR, "assets", "icons")
        self.rename_mode_button = SidebarButton("Rename", os.path.join(icon_dir, "rename.svg"))
        self.filename_mode_button = SidebarButton("Template", os.path.join(icon_dir, "template.svg"))
        self.distribute_mode_button = SidebarButton("Distribution", os.path.join(icon_dir, "distribution.svg"))
        self.logs_mode_button = SidebarButton("Logs", os.path.join(icon_dir, "logs.svg"))
        self.settings_mode_button = SidebarButton("Settings", os.path.join(icon_dir, "settings.svg"))
        self.settings_mode_button.setEnabled(False)

        for idx, btn in enumerate(
            (
                self.rename_mode_button,
                self.filename_mode_button,
                self.distribute_mode_button,
                self.logs_mode_button,
            )
        ):
            self.mode_buttons.addButton(btn, idx)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addWidget(self.settings_mode_button)
        sidebar_layout.addStretch()
        self.sidebar_frame.setLayout(sidebar_layout)
        root_layout.addWidget(self.sidebar_frame)

        right_container = QFrame()
        right_container.setObjectName("ContentRoot")
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(12, 10, 12, 10)
        top_layout.setSpacing(8)
        self.top_search_edit = QLineEdit()
        self.top_search_edit.setObjectName("TopSearch")
        self.top_search_edit.setPlaceholderText("Search files, cases, or logs")
        top_layout.addWidget(self.top_search_edit, 1)
        self.top_new_button = QPushButton("+ New")
        self.top_new_button.setObjectName("PrimaryButton")
        self.top_new_button.clicked.connect(self.choose_input)
        top_layout.addWidget(self.top_new_button)
        self.top_bell_button = TopBarIconButton("•")
        self.top_user_button = TopBarIconButton("◦")
        top_layout.addWidget(self.top_bell_button)
        top_layout.addWidget(self.top_user_button)
        top_bar.setLayout(top_layout)
        right_layout.addWidget(top_bar)

        self.content_stack = QStackedWidget()
        right_layout.addWidget(self.content_stack, 1)
        right_container.setLayout(right_layout)
        root_layout.addWidget(right_container, 1)

        self.main_page = QWidget()
        self.settings_page = QWidget()
        self.distribution_page = QWidget()
        self.logs_page = QWidget()
        self.content_stack.addWidget(self.main_page)
        self.content_stack.addWidget(self.settings_page)
        self.content_stack.addWidget(self.distribution_page)
        self.content_stack.addWidget(self.logs_page)

        logs_page_layout = QVBoxLayout()
        logs_page_layout.setContentsMargins(0, 0, 0, 0)
        logs_page_layout.setSpacing(0)
        logs_card = Card("STATUS LOG")
        logs_layout = QVBoxLayout()
        logs_layout.setSpacing(8)
        self.logs_page_view = QPlainTextEdit()
        self.logs_page_view.setReadOnly(True)
        self.logs_page_view.setPlaceholderText("Status messages appear here")
        self.logs_page_view.setObjectName("StatusLog")
        logs_layout.addWidget(self.logs_page_view)
        logs_card.setLayout(logs_layout)
        logs_page_layout.addWidget(logs_card)
        self.logs_page.setLayout(logs_page_layout)

        def build_collapsible_section(title: str, body: QWidget, collapsed: bool = True) -> QWidget:
            container = QFrame()
            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)
            toggle = QToolButton()
            toggle.setText(title)
            toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            toggle.setArrowType(Qt.ArrowType.RightArrow if collapsed else Qt.ArrowType.DownArrow)
            toggle.setCheckable(True)
            toggle.setChecked(not collapsed)
            toggle.setObjectName("CollapsibleToggle")
            body.setVisible(not collapsed)

            def on_toggled(checked: bool):
                body.setVisible(checked)
                toggle.setArrowType(
                    Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
                )

            toggle.toggled.connect(on_toggled)
            container_layout.addWidget(toggle)
            container_layout.addWidget(body)
            container.setLayout(container_layout)
            return container

        # Main page (Page 0)
        self.main_scroll_area = QScrollArea()
        self.main_scroll_area.setWidgetResizable(True)
        self.main_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.main_scroll_container = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(12)
        self.main_scroll_container.setLayout(self.main_layout)
        self.main_scroll_area.setWidget(self.main_scroll_container)
        main_page_layout = QVBoxLayout()
        main_page_layout.setContentsMargins(0, 0, 0, 0)
        main_page_layout.addWidget(self.main_scroll_area)
        self.main_page.setLayout(main_page_layout)

        io_group = Card("PATHS")
        io_layout = QVBoxLayout()
        io_layout.setSpacing(6)
        io_layout.setContentsMargins(12, 12, 12, 12)
        input_col = QVBoxLayout()
        input_col.setSpacing(6)
        input_label = QLabel("Input folder:")
        input_col.addWidget(input_label)
        self.input_edit = QLineEdit()
        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        input_row.addWidget(self.input_edit)
        btn_input = QPushButton("BROWSE")
        btn_input.clicked.connect(self.choose_input)
        input_row.addWidget(btn_input)
        input_col.addLayout(input_row)
        io_layout.addLayout(input_col)

        output_col = QVBoxLayout()
        output_col.setSpacing(6)
        output_col.addWidget(QLabel("Output folder:"))
        self.output_edit = QLineEdit()
        output_row = QHBoxLayout()
        output_row.setSpacing(6)
        output_row.addWidget(self.output_edit)
        btn_output = QPushButton("BROWSE")
        btn_output.clicked.connect(self.choose_output)
        output_row.addWidget(btn_output)
        output_col.addLayout(output_row)
        io_layout.addLayout(output_col)
        safety_row = QHBoxLayout()
        safety_row.setSpacing(6)
        self.rename_dry_run_checkbox = QCheckBox("Dry run (preview copy destinations)")
        self.rename_dry_run_checkbox.setToolTip("Plan copy operations without writing any files.")
        safety_row.addWidget(self.rename_dry_run_checkbox)
        safety_row.addStretch()
        io_layout.addLayout(safety_row)
        io_group.setLayout(io_layout)
        self.main_layout.addWidget(io_group)

        action_group = Card("COMMAND")
        action_layout = QHBoxLayout()
        action_layout.setSpacing(6)
        action_layout.setContentsMargins(12, 12, 12, 12)
        action_layout.addStretch()
        self.play_button = QPushButton("SCAN")
        self.play_button.setObjectName("PrimaryButton")
        self.play_button.clicked.connect(self.start_processing_clicked)
        action_layout.addWidget(self.play_button)
        self.stop_button = QPushButton("STOP")
        self.stop_button.clicked.connect(self.stop_generation)
        action_layout.addWidget(self.stop_button)
        self.reset_button = QPushButton("RESET")
        self.reset_button.clicked.connect(self.stop_and_reprocess)
        action_layout.addWidget(self.reset_button)
        action_layout.addStretch()
        action_group.setLayout(action_layout)
        self.main_layout.addWidget(action_group)

        table_group = Card("FILES")
        table_layout = QVBoxLayout()
        table_layout.setSpacing(6)
        table_layout.setContentsMargins(12, 12, 12, 12)
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["PDF file", "Proposed filename"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.setSelectionBehavior(self.file_table.SelectionBehavior.SelectRows)
        self.file_table.setEditTriggers(self.file_table.EditTrigger.NoEditTriggers)
        self.file_table.cellClicked.connect(self.on_row_selected)
        table_layout.addWidget(self.file_table)
        table_group.setLayout(table_layout)
        self.main_layout.addWidget(table_group)

        preview_group = Card("FILENAME")
        preview_layout = QVBoxLayout()
        preview_layout.setSpacing(6)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        name_col = QVBoxLayout()
        name_col.setSpacing(6)
        name_col.addWidget(QLabel("Proposed filename:"))
        self.filename_edit = QLineEdit()
        self.filename_edit.editingFinished.connect(self.update_filename_for_current_row)
        name_col.addWidget(self.filename_edit)
        preview_layout.addLayout(name_col)
        preview_group.setLayout(preview_layout)
        self.main_layout.addWidget(preview_group)

        ocr_group = Card("")
        ocr_layout = QVBoxLayout()
        ocr_layout.setSpacing(6)
        ocr_layout.setContentsMargins(12, 12, 12, 12)
        self.ocr_preview_label = QLabel("OCR text sent to AI:")
        self.ocr_preview = QTextEdit()
        self.ocr_preview.setReadOnly(True)
        self.ocr_preview.setPlaceholderText(
            "The OCR excerpt forwarded to the AI/backend will appear here."
        )
        self.ocr_preview.setMinimumHeight(110)
        ocr_layout.addWidget(self.ocr_preview_label)
        ocr_layout.addWidget(self.ocr_preview)
        ocr_group.setLayout(ocr_layout)
        self.main_layout.addWidget(build_collapsible_section("OCR EXCERPT", ocr_group, collapsed=True))

        log_group = Card("")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(6)
        log_layout.setContentsMargins(12, 12, 12, 12)
        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setPlaceholderText("[hh:mm:ss] status messages appear here")
        self.status_log.setMinimumHeight(110)
        self.status_log.setObjectName("StatusLog")
        log_layout.addWidget(self.status_log)
        log_group.setLayout(log_layout)
        self.main_layout.addWidget(build_collapsible_section("STATUS LOG", log_group, collapsed=True))

        bottom_group = Card("ACTIONS")
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(6)
        bottom_layout.setContentsMargins(12, 12, 12, 12)
        btn_process = QPushButton("EXECUTE ONE")
        btn_process.clicked.connect(self.process_this_file)
        self.btn_process = btn_process

        btn_all = QPushButton("EXECUTE ALL")
        btn_all.clicked.connect(self.process_all_files_safe)
        self.btn_all = btn_all

        bottom_layout.addWidget(btn_process)
        bottom_layout.addWidget(btn_all)
        bottom_layout.addStretch()
        bottom_group.setLayout(bottom_layout)
        self.main_layout.addWidget(bottom_group)

        copyright_label = QLabel("────────────────────")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(copyright_label)

        # Settings page (Page 1)
        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setWidgetResizable(True)
        self.settings_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll_container = QWidget()
        self.settings_layout = QHBoxLayout()
        self.settings_layout.setSpacing(12)
        self.settings_scroll_container.setLayout(self.settings_layout)
        self.settings_scroll_area.setWidget(self.settings_scroll_container)
        settings_page_layout = QVBoxLayout()
        settings_page_layout.setContentsMargins(0, 0, 0, 0)
        settings_page_layout.addWidget(self.settings_scroll_area)
        self.settings_page.setLayout(settings_page_layout)

        controls_group = Card("AI + OCR")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        ocr_group = Card("OCR SETTINGS")
        ocr_layout = QVBoxLayout()
        ocr_layout.setSpacing(6)
        self.run_ocr_checkbox = QCheckBox("Run OCR")
        self.run_ocr_checkbox.setChecked(True)
        self.run_ocr_checkbox.toggled.connect(self.update_preview)
        ocr_layout.addWidget(self.run_ocr_checkbox)

        char_col = QVBoxLayout()
        char_col.setSpacing(6)
        char_col.addWidget(QLabel("Max characters:"))
        self.char_limit_spin = QSpinBox()
        self.char_limit_spin.setRange(100, 10000)
        self.char_limit_spin.setSingleStep(100)
        self.char_limit_spin.setValue(1500)
        self.char_limit_spin.valueChanged.connect(self.update_preview)
        char_col.addWidget(self.char_limit_spin)
        ocr_layout.addLayout(char_col)

        dpi_col = QVBoxLayout()
        dpi_col.setSpacing(6)
        dpi_col.addWidget(QLabel("OCR DPI:"))
        self.ocr_dpi_spin = QSpinBox()
        self.ocr_dpi_spin.setRange(72, 600)
        self.ocr_dpi_spin.setValue(300)
        self.ocr_dpi_spin.valueChanged.connect(self.update_preview)
        dpi_col.addWidget(self.ocr_dpi_spin)
        ocr_layout.addLayout(dpi_col)

        pages_col = QVBoxLayout()
        pages_col.setSpacing(6)
        pages_col.addWidget(QLabel("Pages to scan:"))
        self.ocr_pages_spin = QSpinBox()
        self.ocr_pages_spin.setRange(1, 50)
        self.ocr_pages_spin.setValue(1)
        self.ocr_pages_spin.valueChanged.connect(self.update_preview)
        pages_col.addWidget(self.ocr_pages_spin)
        ocr_layout.addLayout(pages_col)

        self.char_count_label = QLabel("Characters retrieved: 0")
        ocr_layout.addWidget(self.char_count_label)
        ocr_group.setLayout(ocr_layout)
        controls_layout.addWidget(ocr_group)

        ai_group = Card("AI ENGINE")
        ai_layout = QVBoxLayout()
        ai_layout.setSpacing(6)
        backend_col = QVBoxLayout()
        backend_col.setSpacing(6)
        backend_col.addWidget(QLabel("Engine:"))
        backend_row = QHBoxLayout()
        backend_row.setSpacing(6)
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
        backend_col.addLayout(backend_row)
        ai_layout.addLayout(backend_col)

        turbo_col = QVBoxLayout()
        turbo_col.setSpacing(6)
        turbo_col.addWidget(QLabel("Parallelism:"))
        self.turbo_mode_checkbox = QCheckBox("Turbo mode (parallel AI queries)")
        self.turbo_mode_checkbox.setToolTip("Send a couple of requests to each backend and keep the first valid answer.")
        turbo_col.addWidget(self.turbo_mode_checkbox)
        ai_layout.addLayout(turbo_col)
        ai_group.setLayout(ai_layout)
        controls_layout.addWidget(ai_group)

        order_group = Card("NAME ORDER")
        order_layout = QVBoxLayout()
        order_layout.setSpacing(6)
        plaintiff_col = QVBoxLayout()
        plaintiff_col.setSpacing(6)
        plaintiff_col.addWidget(QLabel("Plaintiff order:"))
        self.plaintiff_order_combo = QComboBox()
        self.plaintiff_order_combo.addItem("Surname Name", True)
        self.plaintiff_order_combo.addItem("Name Surname", False)
        self.plaintiff_order_combo.currentIndexChanged.connect(self.update_preview)
        plaintiff_col.addWidget(self.plaintiff_order_combo)
        order_layout.addLayout(plaintiff_col)

        defendant_col = QVBoxLayout()
        defendant_col.setSpacing(6)
        defendant_col.addWidget(QLabel("Defendant order:"))
        self.defendant_order_combo = QComboBox()
        self.defendant_order_combo.addItem("Surname Name", True)
        self.defendant_order_combo.addItem("Name Surname", False)
        self.defendant_order_combo.currentIndexChanged.connect(self.update_preview)
        defendant_col.addWidget(self.defendant_order_combo)
        order_layout.addLayout(defendant_col)
        order_group.setLayout(order_layout)
        controls_layout.addWidget(order_group)

        distribution_group = Card("DISTRIBUTION SETTINGS")
        distribution_layout = QVBoxLayout()
        distribution_layout.setSpacing(6)

        auto_threshold_col = QVBoxLayout()
        auto_threshold_col.setSpacing(6)
        auto_threshold_col.addWidget(QLabel("Auto threshold (score):"))
        self.distribution_auto_threshold_spin = QSpinBox()
        self.distribution_auto_threshold_spin.setRange(0, 200)
        # Conservative defaults: favor surname+given-name alignment while keeping auto-accept cautious.
        self.distribution_auto_threshold_spin.setValue(60)
        auto_threshold_col.addWidget(self.distribution_auto_threshold_spin)
        distribution_layout.addLayout(auto_threshold_col)

        gap_threshold_col = QVBoxLayout()
        gap_threshold_col.setSpacing(6)
        gap_threshold_col.addWidget(QLabel("Gap threshold (score):"))
        self.distribution_gap_threshold_spin = QSpinBox()
        self.distribution_gap_threshold_spin.setRange(0, 200)
        self.distribution_gap_threshold_spin.setValue(12)
        gap_threshold_col.addWidget(self.distribution_gap_threshold_spin)
        distribution_layout.addLayout(gap_threshold_col)

        ai_threshold_col = QVBoxLayout()
        ai_threshold_col.setSpacing(6)
        ai_threshold_col.addWidget(QLabel("AI confidence threshold:"))
        self.distribution_ai_threshold_spin = QDoubleSpinBox()
        self.distribution_ai_threshold_spin.setRange(0.0, 1.0)
        self.distribution_ai_threshold_spin.setSingleStep(0.05)
        self.distribution_ai_threshold_spin.setValue(0.7)
        ai_threshold_col.addWidget(self.distribution_ai_threshold_spin)
        distribution_layout.addLayout(ai_threshold_col)

        topk_col = QVBoxLayout()
        topk_col.setSpacing(6)
        topk_col.addWidget(QLabel("Top-K candidates:"))
        self.distribution_topk_spin = QSpinBox()
        self.distribution_topk_spin.setRange(3, 50)
        self.distribution_topk_spin.setValue(15)
        topk_col.addWidget(self.distribution_topk_spin)
        distribution_layout.addLayout(topk_col)

        unmatched_col = QVBoxLayout()
        unmatched_col.setSpacing(6)
        unmatched_col.addWidget(QLabel("Handle UNMATCHED files:"))
        self.distribution_unmatched_combo = QComboBox()
        self.distribution_unmatched_combo.addItem("Leave in place", "leave")
        self.distribution_unmatched_combo.addItem("Copy to UNASSIGNED subfolder", "copy")
        self.distribution_unmatched_combo.addItem("Move to UNASSIGNED subfolder (destructive)", "move")
        self.distribution_unmatched_combo.setToolTip(
            "COPY keeps originals. MOVE removes originals from the input folder."
        )
        unmatched_col.addWidget(self.distribution_unmatched_combo)
        self.distribution_unassigned_ask_checkbox = QCheckBox("Send ASK to UNASSIGNED too")
        self.distribution_unassigned_ask_checkbox.setToolTip(
            "When enabled, ASK files without manual selection are sent to UNASSIGNED."
        )
        unmatched_col.addWidget(self.distribution_unassigned_ask_checkbox)
        distribution_layout.addLayout(unmatched_col)

        dest_exists_col = QVBoxLayout()
        dest_exists_col.setSpacing(6)
        dest_exists_col.addWidget(QLabel("If destination exists:"))
        self.distribution_dest_exists_combo = QComboBox()
        self.distribution_dest_exists_combo.addItem("Skip (default)", "skip")
        self.distribution_dest_exists_combo.addItem("Rename copy", "rename")
        self.distribution_dest_exists_combo.addItem("Overwrite (danger)", "overwrite")
        self.distribution_dest_exists_combo.setToolTip(
            "Skip leaves existing files in place. Rename copy adds a counter. Overwrite replaces files."
        )
        dest_exists_col.addWidget(self.distribution_dest_exists_combo)
        distribution_layout.addLayout(dest_exists_col)

        stopwords_col = QVBoxLayout()
        stopwords_col.setSpacing(6)
        stopwords_col.addWidget(QLabel("Stopwords (comma-separated):"))
        self.distribution_stopwords_edit = QLineEdit()
        self.distribution_stopwords_edit.setPlaceholderText(
            "sp, spółka, sa, s.a, ag, z.o.o, oddzial, bank, international"
        )
        stopwords_col.addWidget(self.distribution_stopwords_edit)
        distribution_layout.addLayout(stopwords_col)

        distribution_group.setLayout(distribution_layout)
        controls_layout.addWidget(distribution_group)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        self.settings_layout.addWidget(controls_group, 1)

        template_group = Card("FILENAME TEMPLATE")
        template_layout = QVBoxLayout()
        template_layout.setSpacing(6)
        template_layout.setContentsMargins(12, 12, 12, 12)

        selector_col = QVBoxLayout()
        selector_col.setSpacing(6)
        selector_col.addWidget(QLabel("Add element:"))
        selector_row = QHBoxLayout()
        selector_row.setSpacing(6)
        self.template_selector = QComboBox()
        self.template_selector.addItem("Date (today)", "date")
        self.template_selector.addItem("Plaintiff", "plaintiff")
        self.template_selector.addItem("Defendant", "defendant")
        self.template_selector.addItem("Letter type", "letter_type")
        self.template_selector.addItem("Case number", "case_number")
        selector_row.addWidget(self.template_selector)
        add_template_btn = QPushButton("ADD ELEMENT")
        add_template_btn.clicked.connect(self.add_template_element)
        selector_row.addWidget(add_template_btn)
        selector_col.addLayout(selector_row)
        custom_row = QHBoxLayout()
        custom_row.setSpacing(6)
        self.custom_name_edit = QLineEdit()
        self.custom_name_edit.setPlaceholderText("Custom element key (e.g., reference)")
        self.custom_desc_edit = QLineEdit()
        self.custom_desc_edit.setPlaceholderText("AI guidance for this element")
        custom_add_btn = QPushButton("ADD CUSTOM ELEMENT")
        custom_add_btn.clicked.connect(self.add_custom_element_from_inputs)
        custom_row.addWidget(self.custom_name_edit)
        custom_row.addWidget(self.custom_desc_edit)
        custom_row.addWidget(custom_add_btn)
        selector_col.addLayout(custom_row)
        template_layout.addLayout(selector_col)

        template_builder = QHBoxLayout()
        template_builder.setSpacing(6)
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
        template_builder.addWidget(self.template_list)

        buttons_col = QVBoxLayout()
        buttons_col.setSpacing(6)
        remove_btn = QPushButton("REMOVE")
        remove_btn.clicked.connect(self.remove_selected_template_element)
        buttons_col.addWidget(remove_btn)
        buttons_col.addStretch()
        template_builder.addLayout(buttons_col)

        template_layout.addLayout(template_builder)
        template_group.setLayout(template_layout)
        self.settings_layout.addWidget(template_group, 1)

        # Distribution page (Page 2)
        self.distribution_scroll_area = QScrollArea()
        self.distribution_scroll_area.setWidgetResizable(True)
        self.distribution_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.distribution_scroll_container = QWidget()
        self.distribution_layout = QVBoxLayout()
        self.distribution_layout.setSpacing(12)
        self.distribution_scroll_container.setLayout(self.distribution_layout)
        self.distribution_scroll_area.setWidget(self.distribution_scroll_container)
        distribution_page_layout = QVBoxLayout()
        distribution_page_layout.setContentsMargins(0, 0, 0, 0)
        distribution_page_layout.addWidget(self.distribution_scroll_area)
        self.distribution_page.setLayout(distribution_page_layout)

        dist_paths_group = Card("DISTRIBUTION PATHS")
        dist_frame_layout = QVBoxLayout()
        dist_frame_layout.setSpacing(6)
        dist_frame_layout.setContentsMargins(12, 12, 12, 12)
        dist_input_col = QVBoxLayout()
        dist_input_col.setSpacing(6)
        dist_input_col.addWidget(QLabel("Folder containing PDFs to distribute:"))
        dist_input_row = QHBoxLayout()
        dist_input_row.setSpacing(6)
        self.distribution_input_edit = QLineEdit()
        dist_input_row.addWidget(self.distribution_input_edit)
        self.distribution_input_button = QPushButton("BROWSE")
        self.distribution_input_button.clicked.connect(self.choose_distribution_input)
        dist_input_row.addWidget(self.distribution_input_button)
        dist_input_col.addLayout(dist_input_row)
        dist_frame_layout.addLayout(dist_input_col)

        dist_cases_col = QVBoxLayout()
        dist_cases_col.setSpacing(6)
        dist_cases_col.addWidget(QLabel("Case folders root:"))
        dist_cases_row = QHBoxLayout()
        dist_cases_row.setSpacing(6)
        self.case_root_edit = QLineEdit()
        dist_cases_row.addWidget(self.case_root_edit)
        self.case_root_button = QPushButton("BROWSE")
        self.case_root_button.clicked.connect(self.choose_case_root)
        dist_cases_row.addWidget(self.case_root_button)
        dist_cases_col.addLayout(dist_cases_row)
        dist_frame_layout.addLayout(dist_cases_col)
        dist_paths_group.setLayout(dist_frame_layout)
        self.distribution_layout.addWidget(dist_paths_group)

        mode_group = Card("MODE")
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.copy_mode_checkbox = QCheckBox("Copy files (default, mandatory)")
        self.copy_mode_checkbox.setChecked(True)
        self.copy_mode_checkbox.setEnabled(False)
        mode_row.addWidget(self.copy_mode_checkbox)
        self.allow_home_case_root_checkbox = QCheckBox("Allow case root under home folder")
        self.allow_home_case_root_checkbox.setToolTip(
            "Unchecked = extra safety guard. Check only if you want to use a case root inside your home folder."
        )
        mode_row.addWidget(self.allow_home_case_root_checkbox)
        self.auto_apply_checkbox = QCheckBox("Auto apply high-confidence only")
        self.auto_apply_checkbox.setToolTip(
            "Apply only AUTO/AI items that meet confidence thresholds."
        )
        mode_row.addWidget(self.auto_apply_checkbox)
        self.distribution_apply_during_planning_checkbox = QCheckBox(
            "Apply AUTO matches during planning (fast workflow)"
        )
        self.distribution_apply_during_planning_checkbox.setToolTip(
            "When enabled, AUTO matches are copied immediately while the plan is built."
        )
        mode_row.addWidget(self.distribution_apply_during_planning_checkbox)
        self.distribution_use_ai_tiebreaker_checkbox = QCheckBox(
            "Use AI tie-breaker for ambiguous matches"
        )
        self.distribution_use_ai_tiebreaker_checkbox.setToolTip(
            "When enabled, AI can pick between close folder matches."
        )
        mode_row.addWidget(self.distribution_use_ai_tiebreaker_checkbox)
        self.distribution_fast_mode_checkbox = QCheckBox("Fast mode (recommended)")
        self.distribution_fast_mode_checkbox.setToolTip(
            "Uses smaller candidate pools and skips expensive similarity work when no pair match."
        )
        self.distribution_fast_mode_checkbox.setChecked(True)
        mode_row.addWidget(self.distribution_fast_mode_checkbox)
        mode_row.addStretch()
        mode_group.setLayout(mode_row)
        self.distribution_layout.addWidget(mode_group)

        unresolved_group = Card("UNRESOLVED HANDLING")
        unresolved_layout = QVBoxLayout()
        unresolved_layout.setSpacing(6)
        self.distribution_create_unresolved_checkbox = QCheckBox(
            "Create new folder for unresolved files"
        )
        self.distribution_create_unresolved_checkbox.setToolTip(
            "When enabled, unresolved files can be routed into newly created case folders."
        )
        unresolved_layout.addWidget(self.distribution_create_unresolved_checkbox)
        unresolved_scope_row = QHBoxLayout()
        unresolved_scope_row.setSpacing(6)
        unresolved_scope_row.addWidget(QLabel("Create folder for:"))
        self.distribution_create_unresolved_scope_combo = QComboBox()
        self.distribution_create_unresolved_scope_combo.addItem(
            "UNMATCHED only", "unmatched_only"
        )
        self.distribution_create_unresolved_scope_combo.addItem(
            "UNMATCHED + unresolved ASK", "unmatched_and_ask"
        )
        unresolved_scope_row.addWidget(self.distribution_create_unresolved_scope_combo)
        unresolved_layout.addLayout(unresolved_scope_row)
        self.distribution_create_unresolved_case_checkbox = QCheckBox(
            "Include case number/tag in created folder name"
        )
        unresolved_layout.addWidget(self.distribution_create_unresolved_case_checkbox)
        self.distribution_create_unresolved_dry_run_checkbox = QCheckBox(
            "Dry-run folder creation (log only)"
        )
        unresolved_layout.addWidget(self.distribution_create_unresolved_dry_run_checkbox)
        unresolved_group.setLayout(unresolved_layout)
        self.distribution_layout.addWidget(unresolved_group)

        dist_controls_group = Card("EXECUTION")
        dist_controls = QHBoxLayout()
        dist_controls.setSpacing(6)
        self.distribution_status_label = QLabel("Idle")
        dist_controls.addWidget(self.distribution_status_label)
        dist_controls.addStretch()
        self.distribution_progress = QProgressBar()
        self.distribution_progress.setRange(0, 1)
        self.distribution_progress.setValue(0)
        self.distribution_progress.setTextVisible(True)
        dist_controls.addWidget(self.distribution_progress)
        self.distribute_button = QPushButton("RUN DRY RUN")
        self.distribute_button.setObjectName("PrimaryButton")
        self.distribute_button.clicked.connect(self.on_distribute_clicked)
        dist_controls.addWidget(self.distribute_button)
        self.apply_distribution_button = QPushButton("APPLY PLAN")
        self.apply_distribution_button.setEnabled(False)
        self.apply_distribution_button.clicked.connect(self.on_apply_distribution_clicked)
        dist_controls.addWidget(self.apply_distribution_button)
        self.pause_distribution_button = QPushButton("PAUSE")
        self.pause_distribution_button.setEnabled(False)
        self.pause_distribution_button.clicked.connect(self.on_pause_distribution_clicked)
        dist_controls.addWidget(self.pause_distribution_button)
        dist_controls_group.setLayout(dist_controls)
        self.distribution_layout.addWidget(dist_controls_group)

        plan_group = Card("DISTRIBUTION PLAN")
        plan_layout = QVBoxLayout()
        plan_layout.setSpacing(6)
        plan_layout.setContentsMargins(12, 12, 12, 12)
        plan_controls = QHBoxLayout()
        plan_controls.setSpacing(6)
        plan_controls.addWidget(QLabel("Show:"))
        self.distribution_filter_combo = QComboBox()
        self.distribution_filter_combo.addItem("All", "all")
        self.distribution_filter_combo.addItem("ASK only", "ASK")
        self.distribution_filter_combo.addItem("AUTO only", "AUTO")
        self.distribution_filter_combo.addItem("AI only", "AI")
        self.distribution_filter_combo.addItem("UNMATCHED only", "UNMATCHED")
        self.distribution_filter_combo.addItem("Unresolved (ASK+UNMATCHED)", "UNRESOLVED")
        self.distribution_filter_combo.currentIndexChanged.connect(
            self.on_distribution_filter_changed
        )
        plan_controls.addWidget(self.distribution_filter_combo)
        plan_controls.addSpacing(12)
        plan_controls.addWidget(QLabel("Sort:"))
        self.distribution_sort_combo = QComboBox()
        self.distribution_sort_combo.addItem("Decision (ASK first)", "decision")
        self.distribution_sort_combo.addItem("Confidence (low→high)", "confidence_asc")
        self.distribution_sort_combo.addItem("Confidence (high→low)", "confidence_desc")
        self.distribution_sort_combo.currentIndexChanged.connect(
            self.on_distribution_sort_changed
        )
        plan_controls.addWidget(self.distribution_sort_combo)
        self.distribution_unresolved_workset_button = QPushButton("Create work set from unresolved")
        self.distribution_unresolved_workset_button.setToolTip(
            "Copies unresolved files into _unresolved_workset and switches input to that folder."
        )
        self.distribution_unresolved_workset_button.clicked.connect(
            self.on_create_unresolved_workset
        )
        plan_controls.addWidget(self.distribution_unresolved_workset_button)
        self.distribution_show_unresolved_button = QPushButton("Show unresolved only")
        self.distribution_show_unresolved_button.clicked.connect(
            self.on_show_unresolved_only
        )
        plan_controls.addWidget(self.distribution_show_unresolved_button)
        plan_controls.addStretch()
        plan_layout.addLayout(plan_controls)
        self.distribution_plan_table = QTableWidget(0, 6)
        self.distribution_plan_table.setHorizontalHeaderLabels(
            [
                "PDF",
                "Decision",
                "Chosen folder",
                "Confidence",
                "Result",
                "Select folder (ASK)",
            ]
        )
        self.distribution_plan_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.distribution_plan_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.distribution_plan_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.distribution_plan_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.distribution_plan_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.distribution_plan_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.ResizeToContents
        )
        self.distribution_plan_table.setEditTriggers(
            self.distribution_plan_table.EditTrigger.NoEditTriggers
        )
        self.distribution_plan_table.setMaximumHeight(300)
        self.distribution_plan_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        plan_layout.addWidget(self.distribution_plan_table)
        plan_group.setLayout(plan_layout)
        self.distribution_layout.addWidget(plan_group)

        log_group = Card("DISTRIBUTION LOG")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(6)
        log_layout.setContentsMargins(12, 12, 12, 12)
        self.distribution_log_view = QTextEdit()
        self.distribution_log_view.setReadOnly(True)
        self.distribution_log_view.setPlaceholderText(
            "Processing details will appear here. Copies are logged to disk as well."
        )
        self.distribution_log_view.setMaximumHeight(170)
        self.distribution_log_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        log_layout.addWidget(self.distribution_log_view)
        log_group.setLayout(log_layout)
        self.distribution_layout.addWidget(log_group)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Waiting for input…")
        self.backend_status_label = QLabel("")
        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.backend_status_label)
        status_layout.addStretch()
        status_widget.setLayout(status_layout)
        self.status_bar.addPermanentWidget(status_widget, 1)
        self.setStatusBar(self.status_bar)

        self.rename_mode_button.setChecked(True)
        self.on_mode_changed(0)
        self.update_backend_status_label()

        self.processing_enabled = False

        self.apply_window_geometry()
        self.load_settings()
        self.ui_ready = True
        self.update_preview()
        self.check_ollama_status()

    # ------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------

    def apply_window_geometry(self):
        default_size = QSize(1320, 720)
        self.setMinimumWidth(1100)
        self.setMinimumHeight(650)
        self.resize(default_size)
        saved_geometry = self.settings.value("window_geometry")
        if not saved_geometry:
            return
        if not self.restoreGeometry(saved_geometry):
            self.resize(default_size)
            return
        screen = self.screen() or QApplication.primaryScreen()
        if not screen:
            return
        available_height = screen.availableGeometry().height()
        if self.height() > available_height * 0.9:
            self.resize(default_size)

    def on_mode_changed(self, index: int):
        if index < 0 or index >= self.content_stack.count():
            return
        self.content_stack.setCurrentIndex(index)
        mode_names = ["Rename", "Filename Rules", "Distribute", "Logs"]
        selected_mode = mode_names[index] if index < len(mode_names) else f"Mode {index}"
        self.append_status_message(f"[MODE] Switched to {selected_mode}")

    def update_backend_status_label(self):
        if not hasattr(self, "backend_combo"):
            return
        backend_text = self.backend_combo.currentText()
        backend_name = backend_text.split("(")[0].strip()
        self.backend_status_label.setText(f"AI: {backend_name}")

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
        auto_threshold = self.settings.value("distribution_auto_threshold", 60)
        gap_threshold = self.settings.value("distribution_gap_threshold", 12)
        ai_threshold = self.settings.value("distribution_ai_threshold", 0.7)
        top_k = self.settings.value("distribution_top_k", 15)
        unmatched_policy = self.settings.value("distribution_unmatched_policy", "leave")
        if unmatched_policy == "unmatched_folder":
            unmatched_policy = "copy"
        dest_exists_policy = self.settings.value("distribution_dest_exists_policy", "skip")
        stopwords = self.settings.value(
            "distribution_stopwords", ", ".join(DEFAULT_STOPWORDS)
        )
        allow_home = self.settings.value("distribution_allow_home_case_root", False)
        auto_apply = self.settings.value("distribution_auto_apply", False)
        apply_during_planning = self.settings.value(
            "distribution_apply_during_planning", False
        )
        use_ai_tiebreaker = self.settings.value(
            "distribution_use_ai_tiebreaker", False
        )
        fast_mode = self.settings.value("distribution_fast_mode", True)
        unassigned_ask = self.settings.value("distribution_unassigned_include_ask", False)
        create_unresolved = self.settings.value("distribution_create_unresolved", False)
        create_unresolved_scope = self.settings.value(
            "distribution_create_unresolved_scope", "unmatched_only"
        )
        create_unresolved_case = self.settings.value(
            "distribution_create_unresolved_case", False
        )
        create_unresolved_dry_run = self.settings.value(
            "distribution_create_unresolved_dry_run", False
        )
        self.distribution_auto_threshold_spin.setValue(int(auto_threshold))
        self.distribution_gap_threshold_spin.setValue(int(gap_threshold))
        self.distribution_ai_threshold_spin.setValue(float(ai_threshold))
        self.distribution_topk_spin.setValue(int(top_k))
        unmatched_index = self.distribution_unmatched_combo.findData(str(unmatched_policy))
        if unmatched_index >= 0:
            self.distribution_unmatched_combo.setCurrentIndex(unmatched_index)
        dest_exists_index = self.distribution_dest_exists_combo.findData(str(dest_exists_policy))
        if dest_exists_index >= 0:
            self.distribution_dest_exists_combo.setCurrentIndex(dest_exists_index)
        self.distribution_stopwords_edit.setText(str(stopwords))
        self.allow_home_case_root_checkbox.setChecked(str(allow_home).lower() == "true")
        self.auto_apply_checkbox.setChecked(str(auto_apply).lower() == "true")
        self.distribution_apply_during_planning_checkbox.setChecked(
            str(apply_during_planning).lower() == "true"
        )
        self.distribution_use_ai_tiebreaker_checkbox.setChecked(
            str(use_ai_tiebreaker).lower() == "true"
        )
        self.distribution_fast_mode_checkbox.setChecked(str(fast_mode).lower() != "false")
        self.distribution_unassigned_ask_checkbox.setChecked(str(unassigned_ask).lower() == "true")
        self.distribution_create_unresolved_checkbox.setChecked(
            str(create_unresolved).lower() == "true"
        )
        scope_index = self.distribution_create_unresolved_scope_combo.findData(
            str(create_unresolved_scope)
        )
        if scope_index >= 0:
            self.distribution_create_unresolved_scope_combo.setCurrentIndex(scope_index)
        self.distribution_create_unresolved_case_checkbox.setChecked(
            str(create_unresolved_case).lower() == "true"
        )
        self.distribution_create_unresolved_dry_run_checkbox.setChecked(
            str(create_unresolved_dry_run).lower() == "true"
        )
        saved_custom = self.settings.value("custom_elements", "{}")
        try:
            self.custom_elements = json.loads(saved_custom) if isinstance(saved_custom, str) else (saved_custom or {})
        except Exception:
            self.custom_elements = {}
        self.ensure_custom_selector_items()
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
        self.settings.setValue("custom_elements", json.dumps(self.custom_elements))
        self.settings.setValue(
            "plaintiff_surname_first", bool(self.plaintiff_order_combo.currentData())
        )
        self.settings.setValue(
            "defendant_surname_first", bool(self.defendant_order_combo.currentData())
        )
        self.settings.setValue(
            "distribution_auto_threshold", self.distribution_auto_threshold_spin.value()
        )
        self.settings.setValue(
            "distribution_gap_threshold", self.distribution_gap_threshold_spin.value()
        )
        self.settings.setValue(
            "distribution_ai_threshold", self.distribution_ai_threshold_spin.value()
        )
        self.settings.setValue("distribution_top_k", self.distribution_topk_spin.value())
        self.settings.setValue(
            "distribution_unmatched_policy", self.distribution_unmatched_combo.currentData()
        )
        self.settings.setValue(
            "distribution_unassigned_include_ask",
            self.distribution_unassigned_ask_checkbox.isChecked(),
        )
        self.settings.setValue(
            "distribution_dest_exists_policy",
            self.distribution_dest_exists_combo.currentData(),
        )
        self.settings.setValue(
            "distribution_stopwords", self.distribution_stopwords_edit.text()
        )
        self.settings.setValue(
            "distribution_allow_home_case_root", self.allow_home_case_root_checkbox.isChecked()
        )
        self.settings.setValue(
            "distribution_auto_apply", self.auto_apply_checkbox.isChecked()
        )
        self.settings.setValue(
            "distribution_apply_during_planning",
            self.distribution_apply_during_planning_checkbox.isChecked(),
        )
        self.settings.setValue(
            "distribution_use_ai_tiebreaker",
            self.distribution_use_ai_tiebreaker_checkbox.isChecked(),
        )
        self.settings.setValue(
            "distribution_fast_mode", self.distribution_fast_mode_checkbox.isChecked()
        )
        self.settings.setValue(
            "distribution_create_unresolved",
            self.distribution_create_unresolved_checkbox.isChecked(),
        )
        self.settings.setValue(
            "distribution_create_unresolved_scope",
            self.distribution_create_unresolved_scope_combo.currentData(),
        )
        self.settings.setValue(
            "distribution_create_unresolved_case",
            self.distribution_create_unresolved_case_checkbox.isChecked(),
        )
        self.settings.setValue(
            "distribution_create_unresolved_dry_run",
            self.distribution_create_unresolved_dry_run_checkbox.isChecked(),
        )

    def closeEvent(self, event):
        self.save_settings()
        self.settings.setValue("window_geometry", self.saveGeometry())
        super().closeEvent(event)

    def log_activity(self, message: str):
        log_info(message)
        self.append_status_message(message)

    def append_status_message(self, message: str):
        if not hasattr(self, "status_log"):
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{stamp}] {message}"
        self.status_log.appendPlainText(formatted)
        if hasattr(self, "logs_page_view"):
            self.logs_page_view.appendPlainText(formatted)
        self.status_log.verticalScrollBar().setValue(self.status_log.verticalScrollBar().maximum())
        if hasattr(self, "logs_page_view"):
            self.logs_page_view.verticalScrollBar().setValue(
                self.logs_page_view.verticalScrollBar().maximum()
            )

    def set_status(self, text: str):
        if not text:
            text = "Working…"
        self.status_label.setText(text)
        self.append_status_message(f"[STATUS] {text}")

    def update_processing_progress(self, total: int = None, processed_override: int = None):
        if self.stop_event.is_set() and not self.processing_enabled:
            return
        total_files = total if total is not None else len(self.pdf_files)
        if total_files <= 0:
            total_files = 1
        processed = (
            processed_override
            if processed_override is not None
            else len(self.file_results) + len(self.failed_indices)
        )
        processed = min(processed, total_files)
        self.status_label.setText(f"{processed}/{total_files} processed")
        self.append_status_message(f"[RUN] {processed}/{total_files} processed")

    def start_processing_ui(self, status: str = "Processing…", total: int = None):
        self.set_status(status)
        self.update_processing_progress(total)
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(True)

    def stop_processing_ui(self, status: str = "Idle"):
        self.set_status(status)
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(False)

    def start_distribution_ui(self, total: int):
        self.distribute_button.setDisabled(True)
        self.apply_distribution_button.setDisabled(True)
        self.distribution_input_button.setDisabled(True)
        self.case_root_button.setDisabled(True)
        if hasattr(self, "pause_distribution_button"):
            self.pause_distribution_button.setDisabled(False)
            self.pause_distribution_button.setText("PAUSE")
        self.distribution_pause_event.set()
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(True)
        self.distribution_status_label.setText("Processing…")
        self.distribution_progress.setRange(0, max(1, total))
        self.distribution_progress.setValue(0)

    def stop_distribution_ui(self, status: str = "Idle"):
        self.distribute_button.setDisabled(False)
        self.distribution_input_button.setDisabled(False)
        self.case_root_button.setDisabled(False)
        self.apply_distribution_button.setEnabled(bool(self.distribution_plan))
        if hasattr(self, "pause_distribution_button"):
            self.pause_distribution_button.setDisabled(True)
            self.pause_distribution_button.setText("PAUSE")
        self.distribution_pause_event.set()
        for btn in (self.play_button, self.btn_process, self.btn_all):
            btn.setDisabled(False)
        self.distribution_status_label.setText(status)
        self.distribution_progress.setRange(0, 1)
        self.distribution_progress.setValue(0)

    def check_ollama_status(self):
        self.update_backend_status_label()
        idx = self.backend_combo.currentIndex()
        if idx != 1:
            if idx == 0:
                self.ollama_badge.setText("🌐 Cloud")
            else:
                self.ollama_badge.setText("⚙️ Auto")
            self.ollama_badge.setStyleSheet("")
            return
        ok = check_ollama_health()
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
            self.append_status_message(message)

    def handle_distribution_progress(self, processed: int, total: int, status_text: str):
        total = max(1, total)
        processed = min(processed, total)
        self.distribution_status_label.setText(status_text or "Processing…")
        self.distribution_progress.setRange(0, total)
        self.distribution_progress.setValue(processed)
        self.distribution_progress.setFormat(f"{processed}/{total}")
        self.distribution_progress.setTextVisible(True)
        self.append_status_message(f"[DISTRIBUTE] {processed}/{total} • {status_text}")

    def handle_distribution_log(self, message: str):
        self.append_distribution_log_message(message)

    def handle_distribution_plan_ready(self, plan: list[DistributionPlanItem]):
        self.distribution_plan = plan
        sorted_plan = self.get_sorted_plan(plan, self.distribution_sort_combo.currentData())
        self.update_distribution_plan_table(sorted_plan)
        self.apply_distribution_filter()
        self.apply_distribution_button.setEnabled(bool(plan))
        if plan:
            copied_during_planning = sum(
                1 for item in plan if item.result and item.result.startswith("COPIED")
            )
            unresolved = sum(
                1
                for item in plan
                if item.decision in ("ASK", "UNMATCHED")
            )
            self.append_distribution_log_message(
                f"Plan ready: {len(plan)} files. "
                f"Copied during planning: {copied_during_planning}. "
                f"Remaining ASK/UNMATCHED: {unresolved}."
            )

    def handle_distribution_finished(self):
        self.distribution_status_label.setText("Finished")
        if self.distribution_progress.maximum() > 0:
            self.distribution_progress.setValue(self.distribution_progress.maximum())
        self.stop_distribution_ui("Finished")
        if self.distribution_apply_worker:
            QMessageBox.information(self, "Distribution complete", "Finished applying the plan.")
            if self.distribution_audit_log_path:
                self.append_distribution_log_message(
                    f"Audit log saved to: {self.distribution_audit_log_path}"
                )
            self.distribution_apply_worker = None
        else:
            QMessageBox.information(self, "Distribution complete", "Finished building the plan.")
            if self.distribution_audit_log_path:
                self.append_distribution_log_message(
                    f"Audit log saved to: {self.distribution_audit_log_path}"
                )
        self.distribution_plan_worker = None

    def on_pause_distribution_clicked(self):
        if not hasattr(self, "pause_distribution_button"):
            return
        worker_running = (
            self.distribution_plan_worker
            and self.distribution_plan_worker.isRunning()
        ) or (
            self.distribution_apply_worker
            and self.distribution_apply_worker.isRunning()
        )
        if not worker_running:
            return
        if self.distribution_pause_event.is_set():
            self.distribution_pause_event.clear()
            self.distribution_status_label.setText("Paused")
            self.pause_distribution_button.setText("RESUME")
            self.append_distribution_log_message("[DISTRIBUTE] Paused by user")
        else:
            self.distribution_pause_event.set()
            self.distribution_status_label.setText("Processing…")
            self.pause_distribution_button.setText("PAUSE")
            self.append_distribution_log_message("[DISTRIBUTE] Resumed by user")

    # ------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------

    def paths_are_safe(self, paths: list[tuple[str, str]]) -> bool:
        ok, reason = validate_path_set(paths)
        if not ok:
            log_info(f"[SAFETY] Blocking operation: {reason}")
            show_friendly_error(
                self,
                "Unsafe path selection",
                "The operation was stopped to protect your files.",
                reason,
                icon=QMessageBox.Icon.Warning,
            )
            return False
        return True

    def confirm_batch_summary(
        self, title: str, details: list[str], *, dry_run: bool = False, file_count: int | None = None
    ) -> bool:
        summary_lines = details[:]
        if file_count is not None:
            summary_lines.append(f"Files queued: {file_count}")
        summary_lines.append(
            "Mode: DRY RUN (no files will be copied)" if dry_run else "Mode: COPY (originals stay untouched)"
        )
        summary_lines.append("NO FILES WILL BE DELETED.")
        summary_lines.append("Do you want to continue?")
        message = "\n".join(summary_lines)
        response = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def plan_or_copy_file(
        self, source_path: str, destination_dir: str, target_name: str, *, dry_run: bool = False
    ) -> str:
        if not os.path.isfile(source_path):
            log_filesystem_action("COPY", source_path, destination_dir, status="failed_missing_source")
            raise FileNotFoundError(f"Source file not found: {source_path}")
        base, ext = os.path.splitext(target_name)
        candidate = target_name
        counter = 1
        destination_dir = os.path.abspath(destination_dir)
        while os.path.exists(os.path.join(destination_dir, candidate)):
            candidate = f"{base} ({counter}){ext}"
            counter += 1
        final_path = os.path.join(destination_dir, candidate)
        log_filesystem_action(
            "COPY", source_path, final_path, status="planned (dry-run)" if dry_run else "pending"
        )
        if dry_run:
            return final_path
        os.makedirs(destination_dir, exist_ok=True)
        try:
            shutil.copy2(source_path, final_path)
        except Exception as exc:  # noqa: PERF203 - explicit logging required
            log_filesystem_action("COPY", source_path, final_path, status=f"failed: {exc}")
            raise
        log_filesystem_action("COPY", source_path, final_path, status="success")
        return final_path

    def should_skip_pozew(self, destination_dir: str, target_name: str) -> tuple[bool, str | None]:
        if "pozew" not in target_name.lower():
            return False, None
        if not os.path.isdir(destination_dir):
            return False, None
        try:
            for filename in os.listdir(destination_dir):
                if not filename.lower().endswith(".pdf"):
                    continue
                if "pozew" in filename.lower():
                    return True, os.path.join(destination_dir, filename)
        except OSError as exc:
            log_info(f"[SKIP] Failed to scan destination for pozew: {exc}")
        return False, None

    # ------------------------------------------------------
    # Load PDFs
    # ------------------------------------------------------

    def load_pdfs(self):
        folder = self.input_edit.text()
        if not self.paths_are_safe([("Input folder", folder)]):
            return
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

    def describe_unassigned_action(self, action: str, include_ask: bool) -> str:
        if action == "copy":
            summary = "Copy UNMATCHED to UNASSIGNED"
        elif action == "move":
            summary = "Move UNMATCHED to UNASSIGNED (destructive)"
        else:
            summary = "Leave UNMATCHED in place"
        if include_ask:
            summary += "; ASK included"
        return summary

    def describe_unresolved_folder_action(self, config: DistributionConfig) -> str:
        if not config.create_unresolved_folders:
            return "No auto-created folders for unresolved"
        scope = "UNMATCHED only"
        if config.create_unresolved_scope == "unmatched_and_ask":
            scope = "UNMATCHED + unresolved ASK"
        case_tag = "with case tag" if config.create_unresolved_include_case_number else "no case tag"
        dry_run = "dry-run" if config.create_unresolved_dry_run else "create"
        return f"{dry_run} folders for {scope} ({case_tag})"

    def get_sorted_plan(
        self, plan: list[DistributionPlanItem], sort_mode: str
    ) -> list[DistributionPlanItem]:
        if not plan:
            return []
        if sort_mode == "confidence_asc":
            return sorted(plan, key=lambda item: item.confidence)
        if sort_mode == "confidence_desc":
            return sorted(plan, key=lambda item: item.confidence, reverse=True)
        if sort_mode == "decision":
            priority = {
                "ASK": 0,
                "AI": 1,
                "AUTO": 2,
                "UNMATCHED": 3,
                "SKIP": 3,
            }
            return sorted(
                plan,
                key=lambda item: (priority.get(item.decision, 99), -item.confidence),
            )
        return plan[:]

    def apply_distribution_filter(self):
        if not hasattr(self, "distribution_plan_table"):
            return
        if self.distribution_plan_table.rowCount() == 0:
            return
        filter_mode = self.distribution_filter_combo.currentData()
        for row in range(self.distribution_plan_table.rowCount()):
            decision_item = self.distribution_plan_table.item(row, 1)
            decision = decision_item.text() if decision_item else ""
            if filter_mode == "UNMATCHED":
                matches = decision in ("UNMATCHED", "SKIP")
            elif filter_mode == "UNRESOLVED":
                matches = decision in ("ASK", "UNMATCHED")
            else:
                matches = filter_mode in ("all", decision)
            self.distribution_plan_table.setRowHidden(row, not matches)

    def on_distribution_filter_changed(self):
        self.apply_distribution_filter()

    def on_distribution_sort_changed(self):
        if not self.distribution_plan:
            return
        sorted_plan = self.get_sorted_plan(
            self.distribution_plan, self.distribution_sort_combo.currentData()
        )
        self.update_distribution_plan_table(sorted_plan)
        self.apply_distribution_filter()

    def _get_unresolved_distribution_items(self) -> list[DistributionPlanItem]:
        return [
            item
            for item in self.distribution_plan
            if item.decision == "UNMATCHED"
            or (item.decision == "ASK" and not item.chosen_folder)
        ]

    def on_create_unresolved_workset(self):
        if not self.distribution_plan:
            QMessageBox.information(
                self,
                "No distribution plan",
                "Run a dry run first to generate a plan before creating a work set.",
            )
            return
        input_dir = self.distribution_input_edit.text() or self.input_edit.text()
        if not input_dir or not os.path.isdir(input_dir):
            show_friendly_error(
                self,
                "Invalid input folder",
                "Renamer could not find the distribution input folder.",
                input_dir or "No input folder configured.",
            )
            return
        unresolved = self._get_unresolved_distribution_items()
        if not unresolved:
            QMessageBox.information(
                self,
                "No unresolved files",
                "All files are resolved. Nothing to add to the work set.",
            )
            return
        workset_dir = os.path.join(input_dir, "_unresolved_workset")
        try:
            if os.path.exists(workset_dir):
                shutil.rmtree(workset_dir)
            os.makedirs(workset_dir, exist_ok=True)
        except Exception as exc:
            show_friendly_error(
                self,
                "Work set error",
                "Renamer could not create the unresolved work set folder.",
                str(exc),
            )
            return

        copied = 0
        for item in unresolved:
            try:
                dest_path, _ = resolve_destination_path(
                    item.source_pdf, workset_dir, exists_policy="rename"
                )
                try:
                    os.link(item.source_pdf, dest_path)
                    action = "LINKED"
                except OSError:
                    shutil.copy2(item.source_pdf, dest_path)
                    action = "COPIED"
                copied += 1
                self.append_distribution_log_message(
                    f"{action} -> {os.path.basename(dest_path)}"
                )
            except Exception as exc:
                self.append_distribution_log_message(
                    f"FAILED -> {os.path.basename(item.source_pdf)} ({exc})"
                )

        self.distribution_input_edit.setText(workset_dir)
        unresolved_index = self.distribution_filter_combo.findData("UNRESOLVED")
        if unresolved_index >= 0:
            self.distribution_filter_combo.setCurrentIndex(unresolved_index)
        self.apply_distribution_filter()
        self.append_distribution_log_message(
            f"Work set created: {copied}/{len(unresolved)} files in {workset_dir}"
        )

    def on_show_unresolved_only(self):
        unresolved_index = self.distribution_filter_combo.findData("UNRESOLVED")
        if unresolved_index < 0:
            return
        self.distribution_filter_combo.setCurrentIndex(unresolved_index)
        self.apply_distribution_filter()

    def build_distribution_config(self) -> DistributionConfig:
        raw_stopwords = self.distribution_stopwords_edit.text()
        stopwords = [word.strip() for word in raw_stopwords.split(",") if word.strip()]
        if not stopwords:
            stopwords = DEFAULT_STOPWORDS[:]
        fast_mode = self.distribution_fast_mode_checkbox.isChecked()
        stage2_k = 60 if fast_mode else 80
        candidate_pool_limit = 120 if fast_mode else 200
        return DistributionConfig(
            auto_threshold=float(self.distribution_auto_threshold_spin.value()),
            gap_threshold=float(self.distribution_gap_threshold_spin.value()),
            ai_threshold=float(self.distribution_ai_threshold_spin.value()),
            enable_ai_tiebreaker=self.distribution_use_ai_tiebreaker_checkbox.isChecked(),
            ai_provider=self.get_ai_backend(),
            top_k=int(self.distribution_topk_spin.value()),
            stage2_k=stage2_k,
            candidate_pool_limit=candidate_pool_limit,
            fast_mode=fast_mode,
            stopwords=stopwords,
            unassigned_action=str(self.distribution_unmatched_combo.currentData()),
            unassigned_include_ask=self.distribution_unassigned_ask_checkbox.isChecked(),
            dest_exists_policy=str(self.distribution_dest_exists_combo.currentData()),
            apply_during_planning=self.distribution_apply_during_planning_checkbox.isChecked(),
            ollama_base_url=get_ollama_base_url(),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
            create_unresolved_folders=self.distribution_create_unresolved_checkbox.isChecked(),
            create_unresolved_scope=str(
                self.distribution_create_unresolved_scope_combo.currentData()
            ),
            create_unresolved_include_case_number=(
                self.distribution_create_unresolved_case_checkbox.isChecked()
            ),
            create_unresolved_dry_run=(
                self.distribution_create_unresolved_dry_run_checkbox.isChecked()
            ),
        )

    def update_distribution_plan_table(self, plan: list[DistributionPlanItem]):
        self.distribution_plan_view = plan[:]
        self.distribution_plan_table.setRowCount(0)
        self.distribution_plan_table.setRowCount(len(plan))
        for row, item in enumerate(plan):
            filename = os.path.basename(item.source_pdf)
            chosen_name = os.path.basename(item.chosen_folder) if item.chosen_folder else "—"
            name_item = QTableWidgetItem(filename)
            name_item.setData(Qt.ItemDataRole.UserRole, item.source_pdf)
            decision_item = QTableWidgetItem(item.decision)
            decision_item.setData(Qt.ItemDataRole.UserRole, item.source_pdf)
            chosen_item = QTableWidgetItem(chosen_name)
            chosen_item.setData(Qt.ItemDataRole.UserRole, item.source_pdf)
            confidence_item = QTableWidgetItem(f"{item.confidence:.2f}")
            confidence_item.setData(Qt.ItemDataRole.UserRole, item.source_pdf)
            result_text = item.result or "—"
            result_item = QTableWidgetItem(result_text)
            result_item.setData(Qt.ItemDataRole.UserRole, item.source_pdf)
            self.distribution_plan_table.setItem(row, 0, name_item)
            self.distribution_plan_table.setItem(row, 1, decision_item)
            self.distribution_plan_table.setItem(row, 2, chosen_item)
            self.distribution_plan_table.setItem(row, 3, confidence_item)
            self.distribution_plan_table.setItem(row, 4, result_item)

            chooser = QComboBox()
            top_candidates = item.candidates[:5]
            if item.decision in ("ASK", "AI") and top_candidates:
                chooser.addItem("Skip", "")
                for candidate in top_candidates:
                    chooser.addItem(
                        f"{candidate.folder.folder_name} ({candidate.score:.1f})",
                        candidate.folder.folder_path,
                    )
                if item.chosen_folder and item.chosen_folder not in [
                    cand.folder.folder_path for cand in top_candidates
                ]:
                    chooser.addItem(os.path.basename(item.chosen_folder), item.chosen_folder)
                if item.chosen_folder:
                    chosen_index = chooser.findData(item.chosen_folder)
                    if chosen_index >= 0:
                        chooser.setCurrentIndex(chosen_index)
                chooser.setEnabled(True)
            elif item.chosen_folder:
                chooser.addItem(chosen_name, item.chosen_folder)
                chooser.setEnabled(False)
            else:
                chooser.addItem("—", "")
                chooser.setEnabled(False)

            chooser.setProperty("row_id", item.source_pdf)
            chooser.currentIndexChanged.connect(self.on_distribution_choice_changed)
            if item.result and item.result.startswith("COPIED"):
                chooser.setEnabled(False)
            self.distribution_plan_table.setCellWidget(row, 5, chooser)

        self.distribution_plan_table.resizeRowsToContents()

    def find_distribution_plan_item(self, row_id: str) -> DistributionPlanItem | None:
        if not row_id:
            return None
        for item in self.distribution_plan:
            if item.source_pdf == row_id:
                return item
        return None

    def update_distribution_plan_row_display(self, row_id: str):
        if not row_id:
            return
        for row in range(self.distribution_plan_table.rowCount()):
            name_item = self.distribution_plan_table.item(row, 0)
            if not name_item or name_item.data(Qt.ItemDataRole.UserRole) != row_id:
                continue
            plan_item = self.find_distribution_plan_item(row_id)
            if not plan_item:
                return
            chosen_name = os.path.basename(plan_item.chosen_folder) if plan_item.chosen_folder else "—"
            decision_item = self.distribution_plan_table.item(row, 1)
            chosen_item = self.distribution_plan_table.item(row, 2)
            confidence_item = self.distribution_plan_table.item(row, 3)
            result_item = self.distribution_plan_table.item(row, 4)
            if decision_item:
                decision_item.setText(plan_item.decision)
            if chosen_item:
                chosen_item.setText(chosen_name)
            if confidence_item:
                confidence_item.setText(f"{plan_item.confidence:.2f}")
            if result_item:
                result_item.setText(plan_item.result or "—")
            break

    def on_distribution_choice_changed(self, _index: int = 0):
        chooser = self.sender()
        if not isinstance(chooser, QComboBox):
            return
        row_id = chooser.property("row_id")
        if not row_id:
            return
        plan_item = self.find_distribution_plan_item(str(row_id))
        if not plan_item:
            return
        selected_path = str(chooser.currentData() or "")
        if selected_path:
            plan_item.chosen_folder = selected_path
            if plan_item.decision in ("ASK", "AI"):
                plan_item.confidence = 1.0
                plan_item.reason = "User confirmed target folder"
                if plan_item.decision == "AI":
                    plan_item.decision = "ASK"
        else:
            plan_item.chosen_folder = None
            if plan_item.decision in ("ASK", "AI"):
                plan_item.confidence = 0.0
                if not plan_item.reason:
                    plan_item.reason = "Awaiting user selection"
                if plan_item.decision == "AI":
                    plan_item.decision = "ASK"
        self.update_distribution_plan_row_display(str(row_id))

    def collect_plan_for_apply(self) -> list[DistributionPlanItem]:
        return self.distribution_plan

    def on_distribute_clicked(self):
        input_dir = self.distribution_input_edit.text() or self.input_edit.text()
        case_root = self.case_root_edit.text()
        allow_home = self.allow_home_case_root_checkbox.isChecked()
        ok, reason = validate_distribution_paths(
            input_dir, case_root, allow_home_case_root=allow_home
        )
        if not ok:
            show_friendly_error(
                self,
                "Unsafe distribution paths",
                "Renamer blocked this distribution plan for safety.",
                reason,
                icon=QMessageBox.Icon.Warning,
            )
            return

        if (
            self.distribution_plan_worker
            and self.distribution_plan_worker.isRunning()
        ) or (
            self.distribution_apply_worker
            and self.distribution_apply_worker.isRunning()
        ):
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
        if not any(
            os.path.isdir(os.path.join(case_root, name)) for name in os.listdir(case_root)
        ):
            show_friendly_error(
                self,
                "No case folders",
                "Renamer did not find any case folders inside the selected root.",
                f"Checked path: {case_root}",
                icon=QMessageBox.Icon.Information,
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

        config = self.build_distribution_config()
        unassigned_summary = self.describe_unassigned_action(
            config.unassigned_action, config.unassigned_include_ask
        )
        unresolved_summary = self.describe_unresolved_folder_action(config)
        apply_during_planning = config.apply_during_planning
        audit_log_path = None
        if apply_during_planning:
            response = QMessageBox.warning(
                self,
                "Confirm streaming apply",
                "AUTO matches will be copied immediately while the plan is built.\n"
                "This is not a dry run. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                self.append_distribution_log_message("[SAFETY] Distribution cancelled by user")
                return
            audit_log_path = os.path.join(
                os.path.expanduser("~"), "Renamer", "distribution_audit.log"
            )
            self.distribution_audit_log_path = audit_log_path
        if not self.confirm_batch_summary(
            "Confirm distribution plan",
            [
                f"Input folder: {input_dir}",
                f"Case root: {case_root}",
                f"Unassigned handling: {unassigned_summary}",
                f"Unresolved handling: {unresolved_summary}",
                f"Fast mode: {'On' if config.fast_mode else 'Off'}",
                f"Streaming apply: {'On' if apply_during_planning else 'Off'}",
            ],
            dry_run=not apply_during_planning,
            file_count=len(pdf_files),
        ):
            self.append_distribution_log_message("[SAFETY] Distribution cancelled by user")
            return

        self.distribution_log_view.clear()
        self.distribution_plan_table.setRowCount(0)
        self.apply_distribution_button.setEnabled(False)
        self.distribution_plan = []
        self.distribution_plan_view = []
        if not apply_during_planning:
            self.distribution_audit_log_path = None
        backend = self.get_ai_backend()
        self.start_distribution_ui(len(pdf_files))
        self.distribution_plan_worker = DistributionPlanWorker(
            input_dir=input_dir,
            pdf_files=pdf_files,
            case_root=case_root,
            config=config,
            ai_provider=backend,
            audit_log_path=audit_log_path,
            pause_event=self.distribution_pause_event,
        )
        self.distribution_plan_worker.progress.connect(self.handle_distribution_progress)
        self.distribution_plan_worker.log_ready.connect(self.handle_distribution_log)
        self.distribution_plan_worker.plan_ready.connect(self.handle_distribution_plan_ready)
        self.distribution_plan_worker.finished.connect(self.handle_distribution_finished)
        self.distribution_plan_worker.start()

    def on_apply_distribution_clicked(self):
        if not self.distribution_plan:
            show_friendly_error(
                self,
                "No plan available",
                "Run a dry run first to generate a distribution plan.",
                "No plan found.",
                icon=QMessageBox.Icon.Information,
            )
            return

        input_dir = self.distribution_input_edit.text() or self.input_edit.text()
        case_root = self.case_root_edit.text()
        allow_home = self.allow_home_case_root_checkbox.isChecked()
        ok, reason = validate_distribution_paths(
            input_dir, case_root, allow_home_case_root=allow_home
        )
        if not ok:
            show_friendly_error(
                self,
                "Unsafe distribution paths",
                "Renamer blocked this distribution apply for safety.",
                reason,
                icon=QMessageBox.Icon.Warning,
            )
            return

        if (
            self.distribution_plan_worker
            and self.distribution_plan_worker.isRunning()
        ) or (
            self.distribution_apply_worker
            and self.distribution_apply_worker.isRunning()
        ):
            show_friendly_error(
                self,
                "Distribution in progress",
                "Please wait for the current distribution run to finish.",
                "Distribution worker already running.",
                icon=QMessageBox.Icon.Information,
            )
            return

        config = self.build_distribution_config()
        if config.unassigned_action == "move":
            response = QMessageBox.warning(
                self,
                "Confirm destructive move",
                "MOVE will remove originals from the input folder and place them in UNASSIGNED.\n"
                "This cannot be undone. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                self.append_distribution_log_message("[SAFETY] Apply cancelled by user")
                return
        unassigned_summary = self.describe_unassigned_action(
            config.unassigned_action, config.unassigned_include_ask
        )
        unresolved_summary = self.describe_unresolved_folder_action(config)
        if not self.confirm_batch_summary(
            "Confirm distribution apply",
            [
                f"Input folder: {input_dir}",
                f"Case root: {case_root}",
                f"Unassigned handling: {unassigned_summary}",
                f"Unresolved handling: {unresolved_summary}",
            ],
            dry_run=False,
            file_count=len(self.distribution_plan),
        ):
            self.append_distribution_log_message("[SAFETY] Apply cancelled by user")
            return

        backend = self.get_ai_backend()
        audit_log_path = os.path.join(
            os.path.expanduser("~"), "Renamer", "distribution_audit.log"
        )
        self.distribution_audit_log_path = audit_log_path
        plan_to_apply = self.distribution_plan
        auto_only = self.auto_apply_checkbox.isChecked()

        self.start_distribution_ui(len(plan_to_apply))
        self.distribution_apply_worker = DistributionApplyWorker(
            input_dir=input_dir,
            case_root=case_root,
            config=config,
            ai_provider=backend,
            plan=plan_to_apply,
            auto_only=auto_only,
            audit_log_path=audit_log_path,
            pause_event=self.distribution_pause_event,
        )
        self.distribution_apply_worker.progress.connect(self.handle_distribution_progress)
        self.distribution_apply_worker.log_ready.connect(self.handle_distribution_log)
        self.distribution_apply_worker.finished.connect(self.handle_distribution_finished)
        self.distribution_apply_worker.start()

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
            custom_elements=self.custom_elements,
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
            "case_number": "Case number",
        }
        return mapping.get(element, element)

    def add_template_item(self, element: str, refresh: bool = True):
        item = QListWidgetItem(self.display_name_for_element(element))
        item.setData(Qt.ItemDataRole.UserRole, element)
        self.template_list.addItem(item)
        if refresh:
            self.update_preview()

    def ensure_custom_selector_items(self):
        existing = {self.template_selector.itemData(i) for i in range(self.template_selector.count())}
        for key in self.custom_elements:
            if key not in existing:
                self.template_selector.addItem(f"Custom: {key}", key)

    def add_template_element(self):
        element = self.template_selector.currentData()
        if not element:
            return
        self.add_template_item(element)

    def add_custom_element_from_inputs(self):
        raw_name = (self.custom_name_edit.text() or "").strip()
        desc = (self.custom_desc_edit.text() or "").strip()
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name).strip("_").lower()
        if not key:
            self.append_status_message("[Template] Custom element name is required")
            return
        self.custom_elements[key] = desc
        self.ensure_custom_selector_items()
        self.add_template_item(key)
        self.save_settings()

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
        requirements = requirements_from_template(options.template_elements, options.custom_elements)
        meta = apply_meta_defaults(meta, requirements)
        meta = apply_party_order(
            meta,
            plaintiff_surname_first=options.plaintiff_surname_first,
            defendant_surname_first=options.defendant_surname_first,
        )
        filename = build_filename(meta, options.template_elements)
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

    def stop_generation(self):
        if not self.processing_enabled and not self.active_workers:
            self.append_status_message("[STOP] No active generation to stop.")
            return
        self.stop_event.set()
        self.processing_enabled = False
        for worker in list(self.active_workers.values()):
            worker.requestInterruption()
        self.set_status("Stopped • results kept")
        self.append_status_message("[STOP] Halted new OCR/AI tasks; existing results preserved")
        self.stop_processing_ui("Stopped")

    def stop_and_reprocess(self):
        self.stop_event.set()
        log_info("Stopping current processing and resetting state")
        self.append_status_message("[RESET] Clearing OCR cache and filenames")

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
        input_folder = self.input_edit.text()
        if not self.paths_are_safe(
            [("Input folder", input_folder), ("Output folder", out_folder)]
        ):
            return
        if not os.path.isdir(out_folder):
            show_friendly_error(
                self,
                "Output folder missing",
                "Please choose where copied files should be saved.",
                f"Checked path: {out_folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not os.path.isdir(input_folder):
            show_friendly_error(
                self,
                "Input folder missing",
                "Please choose an input folder containing your PDFs.",
                f"Checked path: {input_folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not self.pdf_files or (self.current_index in self.active_workers):
            return

        self.stop_event.clear()

        self.update_filename_for_current_row()
        self.start_processing_ui("Copying current file…", total=1)

        pdf_name = self.pdf_files[self.current_index]
        inp = os.path.join(input_folder, pdf_name)

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

        dry_run = self.rename_dry_run_checkbox.isChecked()
        out = os.path.join(out_folder, target_name)

        try:
            should_skip, existing_path = self.should_skip_pozew(out_folder, target_name)
            if should_skip:
                message = f"Pozew already exists in output folder:\n{existing_path}"
                self.append_status_message(f"[SKIP] {pdf_name} → {message}")
                QMessageBox.information(self, "Skipped", message)
                self.update_processing_progress(total=1, processed_override=1)
                return
            final_path = self.plan_or_copy_file(inp, out_folder, target_name, dry_run=dry_run)
            if self.current_index in self.file_results:
                self.file_results[self.current_index]["filename"] = os.path.basename(final_path)
            self.update_processing_progress(total=1, processed_override=1)
            if dry_run:
                QMessageBox.information(
                    self,
                    "Dry run complete",
                    f"Planned copy path:\n{final_path}\n\nNO FILES WERE COPIED.",
                )
            else:
                QMessageBox.information(self, "Done", f"Copied to:\n{final_path}")
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
        input_folder = self.input_edit.text()
        if not self.paths_are_safe(
            [("Input folder", input_folder), ("Output folder", out_folder)]
        ):
            return
        if not os.path.isdir(out_folder):
            show_friendly_error(
                self,
                "Output folder missing",
                "Please choose where copied files should be saved.",
                f"Checked path: {out_folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not os.path.isdir(input_folder):
            show_friendly_error(
                self,
                "Input folder missing",
                "Please choose an input folder containing your PDFs.",
                f"Checked path: {input_folder}",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if not self.pdf_files:
            return

        dry_run = self.rename_dry_run_checkbox.isChecked()
        if not self.confirm_batch_summary(
            "Confirm batch copy",
            [f"Input folder: {input_folder}", f"Output folder: {out_folder}"],
            dry_run=dry_run,
            file_count=len(self.pdf_files),
        ):
            self.append_status_message("[SAFETY] Batch copy cancelled by user")
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

                inp_path = os.path.join(input_folder, pdf_name)
                should_skip, existing_path = self.should_skip_pozew(out_folder, target_name)
                if should_skip:
                    self.append_status_message(
                        f"[SKIP] {pdf_name} → Pozew already exists ({existing_path})"
                    )
                    self.update_processing_progress(
                        total=len(self.pdf_files), processed_override=idx + 1
                    )
                    continue
                planned_path = self.plan_or_copy_file(
                    inp_path, out_folder, target_name, dry_run=dry_run
                )
                prefix = "[DRY-RUN]" if dry_run else "[COPY]"
                self.append_status_message(f"{prefix} {pdf_name} → {planned_path}")
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

        if dry_run:
            QMessageBox.information(
                self,
                "Dry run complete",
                "Planned copy destinations are listed in the status panel.\n\nNO FILES WERE COPIED.",
            )
        else:
            QMessageBox.information(self, "Done", "All files copied.")
        self.stop_processing_ui("Idle")

    def process_all(self):
        self.process_all_files_safe()

    # Helpers
    def handle_worker_finished(self, index: int, result: dict):
        self.active_workers.pop(index, None)
        self.file_results[index] = result
        self.apply_cached_result(index, result)
        self.update_processing_progress()
        if not self.stop_event.is_set() and self.processing_enabled:
            self.start_parallel_processing()
        self.log_activity(
            f"✓ Processed file {index + 1} of {len(self.pdf_files)} (chars: {result.get('char_count', 0)})"
        )
        if not self.active_workers:
            final_status = "Stopped" if self.stop_event.is_set() else "Idle"
            self.stop_processing_ui(final_status)
            log_info(f"All queued workers completed ({final_status})")

    def handle_worker_failed(self, index: int, error: Exception):
        self.active_workers.pop(index, None)
        self.failed_indices.add(index)
        log_exception(error)
        log_info(f"Worker {index} failed: {error}")
        if not self.stop_event.is_set():
            show_friendly_error(
                self,
                "Processing failed",
                "Renamer could not finish processing one of the files.",
                traceback.format_exc(),
            )
        self.update_processing_progress()
        if not self.stop_event.is_set() and self.processing_enabled:
            self.start_parallel_processing()
        if not self.active_workers:
            final_status = "Stopped" if self.stop_event.is_set() else "Idle"
            self.stop_processing_ui(final_status)

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
        requirements = requirements_from_template(options.template_elements, options.custom_elements)
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

        raw_meta = extract_metadata_ai(ocr_text, self.get_ai_backend(), options.custom_elements, options.turbo_mode) or {}
        self.log_activity(
            "[AI] JSON parsed: "
            f"defendant={raw_meta.get('defendant')}, "
            f"plaintiff={raw_meta.get('plaintiff')}, "
            f"letter_type={raw_meta.get('letter_type')}"
        )
        if not raw_meta.get("defendant"):
            fallback_defendant = defendant_from_filename(pdf)
            if fallback_defendant:
                raw_meta["defendant"] = fallback_defendant
        defaults_applied = [key for key in requirements if key not in raw_meta or not raw_meta.get(key)]
        meta = apply_meta_defaults(raw_meta, requirements)
        meta = apply_party_order(
            meta,
            plaintiff_surname_first=options.plaintiff_surname_first,
            defendant_surname_first=options.defendant_surname_first,
        )

        if defaults_applied:
            self.log_activity(
                f"[UI] Applied defaults for missing fields: {', '.join(defaults_applied)}"
            )
            missing_custom = [
                key for key in defaults_applied
                if key not in ("plaintiff", "defendant", "letter_type", "date", "case_number")
            ]
            if missing_custom:
                self.log_activity(
                    f"[WARN] Missing custom element values: {', '.join(missing_custom)}"
                )
        self.log_activity(f"[UI] Extracted meta: {json.dumps(meta, ensure_ascii=False)}")

        filename = build_filename(meta, options.template_elements)

        self.log_activity(f"[UI] Proposed filename: {filename} (backend={AI_BACKEND})")

        return {
            "ocr_text": ocr_text,
            "meta": meta,
            "raw_meta": raw_meta,
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
    app.setStyleSheet(load_stylesheet())
    retro_palette = app.palette()
    retro_palette.setColor(QPalette.ColorRole.Window, QColor("#000000"))
    retro_palette.setColor(QPalette.ColorRole.Base, QColor("#000000"))
    retro_palette.setColor(QPalette.ColorRole.Text, QColor("#00FF66"))
    retro_palette.setColor(QPalette.ColorRole.ButtonText, QColor("#00FF66"))
    retro_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#00AA44"))
    retro_palette.setColor(QPalette.ColorRole.Highlight, QColor("#00FF66"))
    retro_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    app.setPalette(retro_palette)

    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
    icon_path = os.path.join(BASE_DIR, "assets", "logo.ico")
    icon_file = icon_path if os.path.exists(icon_path) else logo_path
    if os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))

    gui = RenamerGUI()
    if icon_file and os.path.exists(icon_file):
        gui.setWindowIcon(QIcon(icon_file))
    gui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
