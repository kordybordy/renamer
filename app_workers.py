import json
import os
import threading
from dataclasses import dataclass
from typing import List

from PyQt6.QtCore import QThread, pyqtSignal

from distribution.engine import DistributionConfig
from distribution.models import DistributionPlanItem

from app_logging import log_exception, log_info
from core_service import (
    AiMetadataRequest,
    CoreApplicationService,
    DistributionApplyRequest,
    DistributionPlanRequest,
    OcrRequest,
    RenameDocumentRequest,
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
        self.core_service = CoreApplicationService()

    def run(self):
        try:
            if self.stop_event.is_set():
                return
            log_info(
                f"[Worker {self.index + 1}] Starting OCR for '{os.path.basename(self.pdf_path)}' "
                f"(pages={self.options.ocr_pages}, dpi={self.options.ocr_dpi}, "
                f"char_limit={self.options.ocr_char_limit}, backend={self.backend})"
            )
            result = self.core_service.process_document(
                RenameDocumentRequest(
                    pdf_path=self.pdf_path,
                    source_name=self.pdf_path,
                    ocr=OcrRequest(
                        pdf_path=self.pdf_path,
                        enabled=self.options.ocr_enabled,
                        char_limit=self.options.ocr_char_limit,
                        dpi=self.options.ocr_dpi,
                        pages=self.options.ocr_pages,
                    ),
                    ai=AiMetadataRequest(
                        ocr_text="",
                        backend=self.backend,
                        custom_elements=self.options.custom_elements,
                        turbo_mode=self.options.turbo_mode,
                    ),
                    template_elements=self.options.template_elements,
                    plaintiff_surname_first=self.options.plaintiff_surname_first,
                    defendant_surname_first=self.options.defendant_surname_first,
                )
            )
            ocr_text = result.ocr_text
            char_count = result.char_count
            if char_count:
                log_info(f"[Worker {self.index + 1}] OCR extracted {char_count} characters")
            else:
                log_info(
                    f"[Worker {self.index + 1}] OCR returned no text; placeholders likely in output"
                )
            raw_meta = result.raw_meta
            defaults_applied = result.defaults_applied
            meta = result.meta

            if defaults_applied:
                log_info(
                    f"[Worker {self.index + 1}] Applied defaults for missing fields: {', '.join(defaults_applied)}"
                )
            log_info(
                f"[Worker {self.index + 1}] Extracted meta: {json.dumps(meta, ensure_ascii=False)}"
            )

            filename = result.filename

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
                    "source_pdf": self.pdf_path,
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
        self.core_service = CoreApplicationService()

    def run(self):
        try:
            response = self.core_service.plan_distribution(
                DistributionPlanRequest(
                    input_dir=self.input_dir,
                    pdf_files=self.pdf_files,
                    case_root=self.case_root,
                    config=self.config,
                    ai_provider=self.ai_provider,
                    audit_log_path=self.audit_log_path,
                    pause_event=self.pause_event,
                    progress_cb=lambda processed, total, status: self.progress.emit(
                        processed, total, status
                    ),
                    log_cb=self.log_ready.emit,
                )
            )
            self.plan_ready.emit(response.plan)
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
        self.core_service = CoreApplicationService()

    def run(self):
        try:
            self.core_service.apply_distribution_plan(
                DistributionApplyRequest(
                    input_dir=self.input_dir,
                    case_root=self.case_root,
                    config=self.config,
                    ai_provider=self.ai_provider,
                    plan=self.plan,
                    auto_only=self.auto_only,
                    audit_log_path=self.audit_log_path,
                    pause_event=self.pause_event,
                    progress_cb=lambda processed, total, status: self.progress.emit(
                        processed, total, status
                    ),
                    log_cb=self.log_ready.emit,
                )
            )
        except Exception as e:
            log_exception(e)
            self.log_ready.emit(f"Unexpected error during apply: {e}")
        finally:
            self.finished.emit()
