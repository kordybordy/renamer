from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app_ai import extract_metadata_ai
from app_ocr import get_ocr_text
from app_text_utils import (
    apply_meta_defaults,
    apply_party_order,
    build_filename,
    defendant_from_filename,
    requirements_from_template,
)
from distribution.engine import DistributionConfig, DistributionEngine
from distribution.models import DistributionPlanItem


@dataclass
class OcrRequest:
    pdf_path: str
    enabled: bool
    char_limit: int
    dpi: int
    pages: int


@dataclass
class OcrResponse:
    text: str
    char_count: int


@dataclass
class AiMetadataRequest:
    ocr_text: str
    backend: str
    custom_elements: dict[str, str]
    turbo_mode: bool


@dataclass
class AiMetadataResponse:
    metadata: dict


@dataclass
class FilenameBuildRequest:
    meta: dict
    template_elements: list[str]
    custom_elements: dict[str, str]
    plaintiff_surname_first: bool
    defendant_surname_first: bool


@dataclass
class FilenameBuildResponse:
    requirements: dict
    defaults_applied: list[str]
    normalized_meta: dict
    filename: str


@dataclass
class RenameDocumentRequest:
    pdf_path: str
    source_name: str
    ocr: OcrRequest
    ai: AiMetadataRequest
    template_elements: list[str]
    plaintiff_surname_first: bool
    defendant_surname_first: bool


@dataclass
class RenameDocumentResponse:
    ocr_text: str
    char_count: int
    raw_meta: dict
    meta: dict
    filename: str
    requirements: dict
    defaults_applied: list[str]


@dataclass
class DistributionPlanRequest:
    input_dir: str
    pdf_files: list[str]
    case_root: str
    config: DistributionConfig
    ai_provider: str
    audit_log_path: str | None = None
    pause_event: object | None = None
    progress_cb: Callable[[int, int, str], None] | None = None
    log_cb: Callable[[str], None] | None = None


@dataclass
class DistributionPlanResponse:
    plan: list[DistributionPlanItem]


@dataclass
class DistributionApplyRequest:
    input_dir: str
    case_root: str
    config: DistributionConfig
    ai_provider: str
    plan: list[DistributionPlanItem]
    auto_only: bool
    audit_log_path: str
    pause_event: object | None = None
    progress_cb: Callable[[int, int, str], None] | None = None
    log_cb: Callable[[str], None] | None = None


class CoreApplicationService:
    def run_ocr(self, request: OcrRequest) -> OcrResponse:
        if not request.enabled:
            return OcrResponse(text="", char_count=0)
        text = get_ocr_text(
            request.pdf_path,
            request.char_limit,
            request.dpi,
            request.pages,
        )
        return OcrResponse(text=text, char_count=len(text))

    def extract_metadata(self, request: AiMetadataRequest) -> AiMetadataResponse:
        metadata = extract_metadata_ai(
            request.ocr_text,
            request.backend,
            request.custom_elements,
            request.turbo_mode,
        ) or {}
        return AiMetadataResponse(metadata=metadata)

    def generate_filename(self, request: FilenameBuildRequest) -> FilenameBuildResponse:
        requirements = requirements_from_template(request.template_elements, request.custom_elements)
        defaults_applied = [
            key for key in requirements if key not in request.meta or not request.meta.get(key)
        ]
        normalized_meta = apply_meta_defaults(request.meta, requirements)
        normalized_meta = apply_party_order(
            normalized_meta,
            plaintiff_surname_first=request.plaintiff_surname_first,
            defendant_surname_first=request.defendant_surname_first,
        )
        filename = build_filename(normalized_meta, request.template_elements)
        return FilenameBuildResponse(
            requirements=requirements,
            defaults_applied=defaults_applied,
            normalized_meta=normalized_meta,
            filename=filename,
        )

    def process_document(self, request: RenameDocumentRequest) -> RenameDocumentResponse:
        ocr_response = self.run_ocr(request.ocr)
        metadata_response = self.extract_metadata(
            AiMetadataRequest(
                ocr_text=ocr_response.text,
                backend=request.ai.backend,
                custom_elements=request.ai.custom_elements,
                turbo_mode=request.ai.turbo_mode,
            )
        )
        raw_meta = metadata_response.metadata.copy()
        if not raw_meta.get("defendant"):
            fallback_defendant = defendant_from_filename(request.source_name)
            if fallback_defendant:
                raw_meta["defendant"] = fallback_defendant

        filename_response = self.generate_filename(
            FilenameBuildRequest(
                meta=raw_meta,
                template_elements=request.template_elements,
                custom_elements=request.ai.custom_elements,
                plaintiff_surname_first=request.plaintiff_surname_first,
                defendant_surname_first=request.defendant_surname_first,
            )
        )

        return RenameDocumentResponse(
            ocr_text=ocr_response.text,
            char_count=ocr_response.char_count,
            raw_meta=raw_meta,
            meta=filename_response.normalized_meta,
            filename=filename_response.filename,
            requirements=filename_response.requirements,
            defaults_applied=filename_response.defaults_applied,
        )

    def plan_distribution(self, request: DistributionPlanRequest) -> DistributionPlanResponse:
        engine = DistributionEngine(
            input_folder=request.input_dir,
            case_root=request.case_root,
            config=request.config,
            ai_provider=request.ai_provider,
            logger=request.log_cb,
        )
        plan = engine.plan_distribution(
            request.pdf_files,
            progress_cb=request.progress_cb,
            pause_event=request.pause_event,
            audit_log_path=request.audit_log_path,
        )
        return DistributionPlanResponse(plan=plan)

    def apply_distribution_plan(self, request: DistributionApplyRequest) -> None:
        engine = DistributionEngine(
            input_folder=request.input_dir,
            case_root=request.case_root,
            config=request.config,
            ai_provider=request.ai_provider,
            logger=request.log_cb,
        )
        engine.apply_plan(
            request.plan,
            auto_only=request.auto_only,
            audit_log_path=request.audit_log_path,
            progress_cb=request.progress_cb,
            pause_event=request.pause_event,
        )
