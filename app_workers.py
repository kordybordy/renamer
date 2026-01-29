import json
import os
import threading
from dataclasses import dataclass
from typing import List

from PyQt6.QtCore import QThread, pyqtSignal

from distribution.engine import DistributionConfig, DistributionEngine
from distribution.models import DistributionPlanItem

from app_ai import extract_metadata_ai
from app_logging import log_exception, log_info
from app_ocr import get_ocr_text
from app_text_utils import (
    apply_meta_defaults,
    apply_party_order,
    build_filename,
    defendant_from_filename,
    requirements_from_template,
)


@dataclass
class NamingOptions:
    template_elements: list[str]
    custom_elements: dict[str, str]
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
        self.requirements = requirements_from_template(options.template_elements, options.custom_elements)

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
            raw_meta = extract_metadata_ai(ocr_text, self.backend, self.options.custom_elements, self.options.turbo_mode) or {}
            if not raw_meta.get("defendant"):
                fallback_defendant = defendant_from_filename(self.pdf_path)
                if fallback_defendant:
                    raw_meta["defendant"] = fallback_defendant
            defaults_applied = [
                key for key in self.requirements if key not in raw_meta or not raw_meta.get(key)
            ]
            meta = apply_meta_defaults(raw_meta, self.requirements)
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

            filename = build_filename(meta, self.options.template_elements)

            log_info(
                f"[Worker {self.index + 1}] Proposed filename: {filename} (backend={self.backend})"
            )

            self.finished.emit(
                self.index,
                {
                    "ocr_text": ocr_text,
                    "meta": meta,
                    "raw_meta": raw_meta,
                    "filename": filename,
                    "char_count": len(ocr_text),
                },
            )
        except Exception as e:
            log_exception(e)
            self.failed.emit(self.index, e)


class DistributionPlanWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_ready = pyqtSignal(str)
    plan_ready = pyqtSignal(list)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        input_dir: str,
        pdf_files: List[str],
        case_root: str,
        config: DistributionConfig,
        ai_provider: str,
        audit_log_path: str | None,
        pause_event: threading.Event | None = None,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.pdf_files = pdf_files
        self.case_root = case_root
        self.config = config
        self.ai_provider = ai_provider
        self.audit_log_path = audit_log_path
        self.pause_event = pause_event

    def run(self):
        try:
            engine = DistributionEngine(
                input_folder=self.input_dir,
                case_root=self.case_root,
                config=self.config,
                ai_provider=self.ai_provider,
                logger=self.log_ready.emit,
            )
            plan = engine.plan_distribution(
                self.pdf_files,
                progress_cb=lambda processed, total, status: self.progress.emit(
                    processed, total, status
                ),
                pause_event=self.pause_event,
                audit_log_path=self.audit_log_path,
            )
            self.plan_ready.emit(plan)
        except Exception as e:
            log_exception(e)
            self.log_ready.emit(f"Unexpected error during planning: {e}")
        finally:
            self.finished.emit()


class DistributionApplyWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_ready = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        input_dir: str,
        case_root: str,
        config: DistributionConfig,
        ai_provider: str,
        plan: List[DistributionPlanItem],
        auto_only: bool,
        audit_log_path: str,
        pause_event: threading.Event | None = None,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.case_root = case_root
        self.config = config
        self.ai_provider = ai_provider
        self.plan = plan
        self.auto_only = auto_only
        self.audit_log_path = audit_log_path
        self.pause_event = pause_event

    def run(self):
        try:
            engine = DistributionEngine(
                input_folder=self.input_dir,
                case_root=self.case_root,
                config=self.config,
                ai_provider=self.ai_provider,
                logger=self.log_ready.emit,
            )
            engine.apply_plan(
                self.plan,
                auto_only=self.auto_only,
                audit_log_path=self.audit_log_path,
                progress_cb=lambda processed, total, status: self.progress.emit(
                    processed, total, status
                ),
                pause_event=self.pause_event,
            )
        except Exception as e:
            log_exception(e)
            self.log_ready.emit(f"Unexpected error during apply: {e}")
        finally:
            self.finished.emit()
