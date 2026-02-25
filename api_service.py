from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import requests


class ApiServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserRole(str, Enum):
    FIRM_ADMIN = "firm_admin"
    ATTORNEY = "attorney"
    PARALEGAL = "paralegal"
    REVIEWER = "reviewer"
    BILLING_ADMIN = "billing_admin"


PERMISSION_MATRIX: dict[UserRole, set[str]] = {
    UserRole.FIRM_ADMIN: {"upload", "review", "rename_approval", "export"},
    UserRole.ATTORNEY: {"upload", "review", "rename_approval", "export"},
    UserRole.PARALEGAL: {"upload", "review"},
    UserRole.REVIEWER: {"review"},
    UserRole.BILLING_ADMIN: {"export"},
}


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str
    user_id: str
    role: UserRole

    def require_permission(self, permission: str) -> None:
        allowed = PERMISSION_MATRIX.get(self.role, set())
        if permission not in allowed:
            raise ApiServiceError(
                code="forbidden",
                message="User does not have permission for this operation.",
                status_code=403,
                details={"permission": permission, "role": self.role.value},
            )


class _CoreDeps:
    def __init__(self) -> None:
        try:
            from core_service import (
                AiMetadataRequest,
                CoreApplicationService,
                DistributionApplyRequest,
                DistributionPlanRequest,
                OcrRequest,
                RenameDocumentRequest,
            )
            from distribution.engine import DistributionConfig
        except Exception as exc:  # noqa: BLE001
            raise ApiServiceError(
                code="core_service_unavailable",
                message="Core renaming service is unavailable in this environment.",
                status_code=503,
                details={"reason": str(exc)},
            ) from exc

        self.AiMetadataRequest = AiMetadataRequest
        self.CoreApplicationService = CoreApplicationService
        self.DistributionApplyRequest = DistributionApplyRequest
        self.DistributionPlanRequest = DistributionPlanRequest
        self.OcrRequest = OcrRequest
        self.RenameDocumentRequest = RenameDocumentRequest
        self.DistributionConfig = DistributionConfig


@dataclass
class TenantRecord:
    tenant_id: str
    name: str
    created_at: str
    plan_name: str = "growth"
    usage: dict[str, float] = field(default_factory=dict)
    plan_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    onboarding: dict[str, Any] = field(default_factory=dict)
    notifications: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FirmUserRecord:
    tenant_id: str
    user_id: str
    email: str
    role: UserRole
    created_at: str


@dataclass
class MatterRecord:
    tenant_id: str
    matter_id: str
    name: str
    metadata: dict[str, Any]
    created_at: str
    input_dir: str


@dataclass
class DocumentRecord:
    tenant_id: str
    matter_id: str
    document_id: str
    original_name: str
    file_path: str
    created_at: str


@dataclass
class JobRecord:
    tenant_id: str
    matter_id: str
    job_id: str
    job_type: str
    status: str
    created_at: str
    updated_at: str
    webhook_url: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    uploaded_pdfs: list[str] = field(default_factory=list)
    idempotency_key: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    result_files: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    audit_path: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingJobRecord:
    tenant_id: str
    matter_id: str
    processing_job_id: str
    job_id: str
    created_at: str


@dataclass
class AuditEventRecord:
    tenant_id: str
    matter_id: str
    event_id: str
    actor_user_id: str
    action: str
    created_at: str
    details: dict[str, Any] = field(default_factory=dict)


class RenamerApiService:
    def __init__(
        self,
        storage_root: str = "api_data",
        worker_count: int = 2,
        max_retries: int = 3,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._core = None
        self._deps = None
        self._lock = threading.Lock()
        self._tenants: dict[str, TenantRecord] = {}
        self._firm_users: dict[tuple[str, str], FirmUserRecord] = {}
        self._matters: dict[tuple[str, str], MatterRecord] = {}
        self._documents: dict[tuple[str, str, str], DocumentRecord] = {}
        self._jobs: dict[tuple[str, str, str], JobRecord] = {}
        self._idempotency_index: dict[tuple[str, str, str], str] = {}
        self._dead_letters: dict[tuple[str, str, str], JobRecord] = {}
        self._job_queue: queue.Queue[str] = queue.Queue()
        self._worker_count = max(1, worker_count)
        self._max_retries = max(0, max_retries)
        self._workers: list[threading.Thread] = []
        self._start_workers()

    def _start_workers(self) -> None:
        for idx in range(self._worker_count):
            worker = threading.Thread(target=self._worker_loop, args=(idx,), daemon=True)
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self, _: int) -> None:
        while True:
            job_id = self._job_queue.get()
            try:
                self._process_queued_job(job_id)
            finally:
                self._job_queue.task_done()

    @staticmethod
    def _default_plan_limits() -> dict[str, dict[str, float]]:
        return {
            "pages_ocrd": {"soft": 10000, "hard": 12000},
            "documents_processed": {"soft": 5000, "hard": 6000},
            "ai_tokens": {"soft": 5_000_000, "hard": 6_000_000},
            "storage_bytes": {"soft": 5 * 1024**3, "hard": 6 * 1024**3},
            "active_users": {"soft": 50, "hard": 75},
        }

    @staticmethod
    def _default_usage() -> dict[str, float]:
        return {
            "pages_ocrd": 0,
            "documents_processed": 0,
            "ai_tokens": 0,
            "storage_bytes": 0,
            "active_users": 0,
        }

    def _ensure_core(self):
        if self._deps is None:
            self._deps = _CoreDeps()
        if self._core is None:
            self._core = self._deps.CoreApplicationService()
        return self._deps

    def _require_tenant_scope(self, tenant_id: str, access: AccessContext) -> None:
        if tenant_id != access.tenant_id:
            raise ApiServiceError(
                code="cross_tenant_access_denied",
                message="Cross-tenant access is not allowed.",
                status_code=403,
            )

    def _ensure_tenant(self, tenant_id: str) -> TenantRecord:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            tenant = TenantRecord(
                tenant_id=tenant_id,
                name=tenant_id,
                created_at=utc_now_iso(),
                usage=self._default_usage(),
                plan_limits=self._default_plan_limits(),
                onboarding={
                    "workspace_created": True,
                    "admin_invited": False,
                    "sso_provider": None,
                    "sso_configured": False,
                    "checklist": [
                        {"id": "workspace", "label": "Create workspace", "done": True},
                        {"id": "admin_invite", "label": "Invite first admin", "done": False},
                        {"id": "sso", "label": "Configure SSO", "done": False},
                        {"id": "sso_test", "label": "Validate SSO login", "done": False},
                    ],
                },
            )
            self._tenants[tenant_id] = tenant
        return tenant

    def _add_notification(self, tenant: TenantRecord, level: str, message: str, details: dict[str, Any]) -> None:
        tenant.notifications.append(
            {
                "id": str(uuid.uuid4()),
                "level": level,
                "message": message,
                "details": details,
                "created_at": utc_now_iso(),
            }
        )

    def _increment_usage(self, tenant: TenantRecord, metric: str, amount: float) -> None:
        if amount <= 0:
            return
        current = float(tenant.usage.get(metric, 0.0))
        limits = tenant.plan_limits.get(metric, {})
        soft = float(limits.get("soft", float("inf")))
        hard = float(limits.get("hard", float("inf")))
        next_value = current + amount
        if next_value > hard:
            self._add_notification(
                tenant,
                "hard_cap",
                f"Hard cap reached for {metric}; operation blocked.",
                {"metric": metric, "current": current, "attempted_increment": amount, "hard": hard},
            )
            raise ApiServiceError(
                code="plan_hard_cap_exceeded",
                message=f"Plan hard cap exceeded for {metric}.",
                status_code=402,
                details={"metric": metric, "hard": hard, "current": current},
            )
        tenant.usage[metric] = next_value
        if current < soft <= next_value:
            self._add_notification(
                tenant,
                "soft_cap",
                f"Soft cap reached for {metric}.",
                {"metric": metric, "value": next_value, "soft": soft},
            )

    def create_tenant_onboarding(self, workspace_name: str, admin_email: str, sso_provider: str | None) -> TenantRecord:
        tenant_id = str(uuid.uuid4())
        tenant = self._ensure_tenant(tenant_id)
        tenant.name = workspace_name
        tenant.onboarding["admin_invited"] = True
        tenant.onboarding["admin_email"] = admin_email
        tenant.onboarding["invite_sent_at"] = utc_now_iso()
        tenant.onboarding["sso_provider"] = sso_provider
        for item in tenant.onboarding["checklist"]:
            if item["id"] in {"workspace", "admin_invite"}:
                item["done"] = True
            if item["id"] == "sso" and sso_provider:
                item["done"] = True
        return tenant

    def get_onboarding(self, tenant_id: str) -> dict[str, Any]:
        tenant = self._ensure_tenant(tenant_id)
        return tenant.onboarding

    def complete_onboarding_step(self, tenant_id: str, step_id: str, done: bool) -> dict[str, Any]:
        tenant = self._ensure_tenant(tenant_id)
        for item in tenant.onboarding.get("checklist", []):
            if item.get("id") == step_id:
                item["done"] = done
                if step_id == "sso":
                    tenant.onboarding["sso_configured"] = done
                return tenant.onboarding
        raise ApiServiceError("onboarding_step_not_found", "Onboarding step was not found.", 404)

    def upsert_firm_user(self, tenant_id: str, user_id: str, email: str, role: UserRole) -> FirmUserRecord:
        tenant = self._ensure_tenant(tenant_id)
        user = FirmUserRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            email=email,
            role=role,
            created_at=utc_now_iso(),
        )
        with self._lock:
            self._firm_users[(tenant_id, user_id)] = user
            tenant.usage["active_users"] = len({uid for t_id, uid in self._firm_users if t_id == tenant_id})
        self._increment_usage(tenant, "active_users", 0)
        return user

    def _record_audit(self, tenant_id: str, matter_id: str, actor_user_id: str, action: str, details: dict[str, Any]) -> None:
        event = AuditEventRecord(
            tenant_id=tenant_id,
            matter_id=matter_id,
            event_id=str(uuid.uuid4()),
            actor_user_id=actor_user_id,
            action=action,
            details=details,
            created_at=utc_now_iso(),
        )
        with self._lock:
            self._audit_events.setdefault((tenant_id, matter_id), []).append(event)

    def create_matter(self, access: AccessContext, tenant_id: str, name: str, metadata: dict[str, Any] | None) -> MatterRecord:
        self._require_tenant_scope(tenant_id, access)
        access.require_permission("upload")
        self._ensure_tenant(tenant_id)
        matter_id = str(uuid.uuid4())
        matter_dir = self.storage_root / tenant_id / matter_id
        input_dir = matter_dir / "uploads"
        input_dir.mkdir(parents=True, exist_ok=True)
        matter = MatterRecord(
            tenant_id=tenant_id,
            matter_id=matter_id,
            name=name,
            metadata=metadata or {},
            created_at=utc_now_iso(),
            input_dir=str(input_dir),
        )
        with self._lock:
            self._matters[(tenant_id, matter_id)] = matter
        self._record_audit(tenant_id, matter_id, access.user_id, "matter_created", {"name": name})
        return matter

    def get_matter(self, access: AccessContext, tenant_id: str, matter_id: str) -> MatterRecord:
        self._require_tenant_scope(tenant_id, access)
        matter = self._matters.get((tenant_id, matter_id))
        if not matter:
            raise ApiServiceError(
                code="matter_not_found",
                message="Matter could not be found for the tenant.",
                status_code=404,
            )
        return matter

    def store_uploads(self, access: AccessContext, tenant_id: str, matter_id: str, files: list[tuple[str, bytes]]) -> list[str]:
        self._require_tenant_scope(tenant_id, access)
        access.require_permission("upload")
        matter = self.get_matter(access, tenant_id, matter_id)
        tenant = self._ensure_tenant(tenant_id)
        stored: list[str] = []
        for filename, content in files:
            lower_name = filename.lower()
            if not lower_name.endswith(".pdf"):
                raise ApiServiceError(
                    code="invalid_file_type",
                    message="Only PDF uploads are supported.",
                    status_code=400,
                    details={"filename": filename},
                )
            target = Path(matter.input_dir) / Path(filename).name
            self._increment_usage(tenant, "storage_bytes", len(content))
            target.write_bytes(content)
            stored.append(target.name)
            document = DocumentRecord(
                tenant_id=tenant_id,
                matter_id=matter_id,
                document_id=str(uuid.uuid4()),
                original_name=target.name,
                file_path=str(target),
                created_at=utc_now_iso(),
            )
            with self._lock:
                self._documents[(tenant_id, matter_id, document.document_id)] = document
        self._record_audit(tenant_id, matter_id, access.user_id, "files_uploaded", {"count": len(stored)})
        return stored

    def submit_job(
        self,
        access: AccessContext,
        tenant_id: str,
        matter_id: str,
        job_type: str,
        payload: dict[str, Any],
        webhook_url: str | None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        self._require_tenant_scope(tenant_id, access)
        required_permission = "rename_approval" if job_type == "rename" else "review"
        access.require_permission(required_permission)
        tenant = self._ensure_tenant(tenant_id)
        self._increment_usage(tenant, "documents_processed", 0)
        matter = self.get_matter(access, tenant_id, matter_id)
        uploaded_pdfs = sorted(str(path) for path in Path(matter.input_dir).glob("*.pdf"))
        if not uploaded_pdfs:
            raise ApiServiceError(
                code="no_files_uploaded",
                message="Upload at least one PDF before submitting a job.",
                status_code=400,
            )
        if job_type not in {"rename", "distribution"}:
            raise ApiServiceError(
                code="unsupported_job_type",
                message="Job type must be one of: rename, distribution.",
                status_code=400,
                details={"job_type": job_type},
            )

        clean_key = (idempotency_key or "").strip() or None
        if clean_key:
            existing = self._find_job_by_idempotency_key(tenant_id, matter_id, clean_key)
            if existing:
                return existing

        job_id = str(uuid.uuid4())
        now = utc_now_iso()
        job = JobRecord(
            tenant_id=tenant_id,
            matter_id=matter_id,
            job_id=job_id,
            job_type=job_type,
            status="queued",
            created_at=now,
            updated_at=now,
            webhook_url=webhook_url,
            payload=dict(payload),
            uploaded_pdfs=uploaded_pdfs,
            idempotency_key=clean_key,
            max_retries=self._max_retries,
        )
        with self._lock:
            self._jobs[(tenant_id, matter_id, job_id)] = job
            if clean_key:
                self._idempotency_index[(tenant_id, matter_id, clean_key)] = job_id
        self._write_audit_log(job)
        self._job_queue.put(job_id)
        return job

    def _find_job_by_idempotency_key(self, tenant_id: str, matter_id: str, key: str) -> JobRecord | None:
        with self._lock:
            job_id = self._idempotency_index.get((tenant_id, matter_id, key))
            if not job_id:
                return None
            return self._jobs.get((tenant_id, matter_id, job_id))

    def get_job(self, tenant_id: str, matter_id: str, job_id: str) -> JobRecord:
        job = self._jobs.get((tenant_id, matter_id, job_id))
        if not job:
            raise ApiServiceError(
                code="job_not_found",
                message="Job could not be found for the tenant and matter.",
                status_code=404,
            )
        return job

    def get_result_file_path(
        self,
        access: AccessContext,
        tenant_id: str,
        matter_id: str,
        job_id: str,
        result_name: str,
    ) -> str:
        self._require_tenant_scope(tenant_id, access)
        access.require_permission("export")
        job = self.get_job(access, tenant_id, matter_id, job_id)
        for item in job.result_files:
            if item.get("name") == result_name:
                path = item.get("path")
                if path and os.path.exists(path):
                    return path
        raise ApiServiceError(
            code="result_not_found",
            message="Result file not found for this job.",
            status_code=404,
            details={"result_name": result_name},
        )

    def get_audit(self, access: AccessContext, tenant_id: str, matter_id: str, job_id: str) -> dict[str, Any]:
        self._require_tenant_scope(tenant_id, access)
        access.require_permission("review")
        job = self.get_job(access, tenant_id, matter_id, job_id)
        if not job.audit_path or not os.path.isfile(job.audit_path):
            raise ApiServiceError(
                code="audit_not_ready",
                message="Audit data is not available for this job.",
                status_code=404,
            )
        with open(job.audit_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        result["events"] = [asdict(item) for item in self._audit_events.get((tenant_id, matter_id), [])]
        return result

    def trigger_callback(self, access: AccessContext, tenant_id: str, matter_id: str, job_id: str) -> dict[str, Any]:
        self._require_tenant_scope(tenant_id, access)
        access.require_permission("review")
        job = self.get_job(access, tenant_id, matter_id, job_id)
        if not job.webhook_url:
            raise ApiServiceError(
                code="webhook_not_configured",
                message="No webhook URL configured for this job.",
                status_code=400,
            )
        return self._post_webhook(job)

    def _process_queued_job(self, job_id: str) -> None:
        job = self._find_job_by_id(job_id)
        if not job:
            return
        matter = self.get_matter(job.tenant_id, job.matter_id)
        self._set_job_status(job, "running")
        try:
            if job.job_type == "rename":
                self._execute_rename_job(job, matter, job.uploaded_pdfs, job.payload)
            else:
                self._execute_distribution_job(job, matter, job.uploaded_pdfs, job.payload)

            if job.errors:
                self._set_job_status(job, "needs_review")
            else:
                self._set_job_status(job, "completed")
        except Exception as exc:  # noqa: BLE001
            self._handle_job_exception(job, exc)
        finally:
            self._write_audit_log(job)
            if job.webhook_url:
                try:
                    self._post_webhook(job)
                except Exception as exc:  # noqa: BLE001
                    job.errors.append({"code": "webhook_delivery_failed", "message": str(exc)})
                    self._write_audit_log(job)

    def _find_job_by_id(self, job_id: str) -> JobRecord | None:
        with self._lock:
            for (tenant_id, matter_id, stored_job_id), job in self._jobs.items():
                if stored_job_id == job_id:
                    return self._jobs.get((tenant_id, matter_id, stored_job_id))
        return None

    def _handle_job_exception(self, job: JobRecord, exc: Exception) -> None:
        job.failure_history.append(
            {
                "occurred_at": utc_now_iso(),
                "message": str(exc),
            }
        )
        if job.retry_count < job.max_retries:
            job.retry_count += 1
            self._set_job_status(job, "queued")
            self._job_queue.put(job.job_id)
            return

        job.errors.append({"code": "job_failed", "message": str(exc)})
        self._set_job_status(job, "failed")
        with self._lock:
            self._dead_letters[(job.tenant_id, job.matter_id, job.job_id)] = job

    def _set_job_status(self, job: JobRecord, status: str) -> None:
        with self._lock:
            job.status = status
            job.updated_at = utc_now_iso()

    def _execute_rename_job(
        self,
        job: JobRecord,
        matter: MatterRecord,
        uploaded_pdfs: list[str],
        payload: dict[str, Any],
    ) -> None:
        deps = self._ensure_core()
        backend = str(payload.get("backend") or "openai")
        template_elements = list(payload.get("template_elements") or ["plaintiff", "defendant", "letter_type"])
        custom_elements = dict(payload.get("custom_elements") or {})
        ocr_enabled = bool(payload.get("ocr_enabled", True))
        turbo_mode = bool(payload.get("turbo_mode", False))

        result_dir = Path(matter.input_dir).parent / "results" / job.job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        tenant = self._ensure_tenant(job.tenant_id)
        estimated_pages = int(payload.get("estimated_pages_per_document", 3))
        estimated_tokens = int(payload.get("ai_tokens_per_document", 1200))

        processed_count = 0
        skipped_count = 0
        for pdf_path in uploaded_pdfs:
            source_name = Path(pdf_path).name
            try:
                response = self._core.process_document(
                    deps.RenameDocumentRequest(
                        pdf_path=pdf_path,
                        source_name=source_name,
                        ocr=deps.OcrRequest(
                            pdf_path=pdf_path,
                            enabled=ocr_enabled,
                            char_limit=int(payload.get("ocr_char_limit", 6000)),
                            dpi=int(payload.get("ocr_dpi", 300)),
                            pages=int(payload.get("ocr_pages", 3)),
                        ),
                        ai=deps.AiMetadataRequest(
                            ocr_text="",
                            backend=backend,
                            custom_elements=custom_elements,
                            turbo_mode=turbo_mode,
                        ),
                        template_elements=template_elements,
                        plaintiff_surname_first=bool(payload.get("plaintiff_surname_first", False)),
                        defendant_surname_first=bool(payload.get("defendant_surname_first", False)),
                    )
                )
                target_name = f"{response.filename}.pdf"
                target_path = result_dir / target_name
                shutil.copy2(pdf_path, target_path)
                job.result_files.append({"name": target_name, "path": str(target_path), "source": source_name})
                processed_count += 1
            except Exception as exc:  # noqa: BLE001
                skipped_count += 1
                job.errors.append(
                    {
                        "code": "document_processing_failed",
                        "message": str(exc),
                        "document": source_name,
                    }
                )
            )
            target_name = f"{response.filename}.pdf"
            target_path = result_dir / target_name
            shutil.copy2(pdf_path, target_path)
            job.result_files.append({"name": target_name, "path": str(target_path), "source": source_name})
            self._increment_usage(tenant, "documents_processed", 1)
            if ocr_enabled:
                self._increment_usage(tenant, "pages_ocrd", estimated_pages)
            self._increment_usage(tenant, "ai_tokens", estimated_tokens)

        job.summary = {
            "processed": processed_count,
            "results": len(job.result_files),
            "skipped": skipped_count,
        }

    def _execute_distribution_job(
        self,
        job: JobRecord,
        matter: MatterRecord,
        uploaded_pdfs: list[str],
        payload: dict[str, Any],
    ) -> None:
        case_root = str(payload.get("case_root") or "").strip()
        if not case_root:
            raise ApiServiceError(
                code="case_root_required",
                message="Distribution jobs require 'case_root'.",
                status_code=400,
            )

        deps = self._ensure_core()
        cfg = deps.DistributionConfig(
            auto_threshold=float(payload.get("auto_threshold", 70.0)),
            gap_threshold=float(payload.get("gap_threshold", 15.0)),
            ai_threshold=float(payload.get("ai_threshold", 0.7)),
            enable_ai_tiebreaker=bool(payload.get("enable_ai_tiebreaker", False)),
            ai_provider=str(payload.get("ai_provider", "openai")),
        )

        audit_dir = Path(matter.input_dir).parent / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        plan_audit_path = audit_dir / f"{job.job_id}_plan.json"

        planned_items = []
        skipped_count = 0
        for pdf_path in uploaded_pdfs:
            source_name = Path(pdf_path).name
            try:
                item_audit_path = audit_dir / f"{job.job_id}_{Path(pdf_path).stem}_plan.json"
                plan_response = self._core.plan_distribution(
                    deps.DistributionPlanRequest(
                        input_dir=matter.input_dir,
                        pdf_files=[pdf_path],
                        case_root=case_root,
                        config=cfg,
                        ai_provider=cfg.ai_provider,
                        audit_log_path=str(item_audit_path),
                    )
                )
                planned_items.extend(plan_response.plan)
            except Exception as exc:  # noqa: BLE001
                skipped_count += 1
                job.errors.append(
                    {
                        "code": "distribution_document_failed",
                        "message": str(exc),
                        "document": source_name,
                    }
                )

        with open(plan_audit_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "job_id": job.job_id,
                    "planned_items": len(planned_items),
                    "skipped": skipped_count,
                    "errors": job.errors,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        job.summary = {
            "planned": len(planned_items),
            "auto_candidates": sum(1 for item in planned_items if item.status == "auto"),
            "skipped": skipped_count,
        }
        tenant = self._ensure_tenant(job.tenant_id)
        self._increment_usage(tenant, "documents_processed", len(uploaded_pdfs))
        if bool(payload.get("enable_ai_tiebreaker", False)):
            self._increment_usage(tenant, "ai_tokens", len(uploaded_pdfs) * int(payload.get("ai_tokens_per_document", 200)))

        if bool(payload.get("apply", False)) and planned_items:
            apply_audit_path = audit_dir / f"{job.job_id}_apply.json"
            self._core.apply_distribution_plan(
                deps.DistributionApplyRequest(
                    input_dir=matter.input_dir,
                    case_root=case_root,
                    config=cfg,
                    ai_provider=cfg.ai_provider,
                    plan=planned_items,
                    auto_only=bool(payload.get("auto_only", False)),
                    audit_log_path=str(apply_audit_path),
                )
            )
            job.summary["apply_audit"] = str(apply_audit_path)

        job.audit_path = str(plan_audit_path)

    def _write_audit_log(self, job: JobRecord) -> None:
        audit_dir = self.storage_root / job.tenant_id / job.matter_id / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{job.job_id}_job.json"
        dead_lettered = (job.tenant_id, job.matter_id, job.job_id) in self._dead_letters
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "job": asdict(job),
                    "dead_lettered": dead_lettered,
                    "written_at": utc_now_iso(),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        if not job.audit_path:
            job.audit_path = str(audit_path)

    def _post_webhook(self, job: JobRecord) -> dict[str, Any]:
        payload = {
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "matter_id": job.matter_id,
            "status": job.status,
            "summary": job.summary,
            "error_count": len(job.errors),
            "updated_at": job.updated_at,
        }
        response = requests.post(job.webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "delivered_at": utc_now_iso(),
        }

    def get_usage_summary(self, access: AccessContext, tenant_id: str) -> dict[str, Any]:
        self._require_tenant_scope(tenant_id, access)
        access.require_permission("export")
        tenant = self._ensure_tenant(tenant_id)
        return {
            "tenant_id": tenant_id,
            "plan": tenant.plan_name,
            "usage": tenant.usage,
            "limits": tenant.plan_limits,
            "notifications": tenant.notifications,
        }

    def get_invoice_summary(self, access: AccessContext, tenant_id: str) -> dict[str, Any]:
        summary = self.get_usage_summary(access, tenant_id)
        usage = summary["usage"]
        rates = {
            "pages_ocrd": 0.02,
            "documents_processed": 0.05,
            "ai_tokens": 0.000002,
            "storage_bytes": 0.0,
            "active_users": 12.0,
        }
        line_items = []
        total = 0.0
        for metric, value in usage.items():
            cost = float(value) * rates.get(metric, 0.0)
            line_items.append({"metric": metric, "quantity": value, "unit_rate": rates.get(metric, 0.0), "amount": round(cost, 2)})
            total += cost
        return {
            "tenant_id": tenant_id,
            "period": "all_time",
            "currency": "USD",
            "line_items": line_items,
            "total_amount": round(total, 2),
        }

    def get_support_dashboard(self) -> dict[str, Any]:
        tenants = []
        for tenant in self._tenants.values():
            tenants.append(
                {
                    "tenant_id": tenant.tenant_id,
                    "name": tenant.name,
                    "plan": tenant.plan_name,
                    "usage": tenant.usage,
                    "open_notifications": len(tenant.notifications),
                    "created_at": tenant.created_at,
                }
            )
        return {
            "generated_at": utc_now_iso(),
            "tenant_count": len(tenants),
            "tenants": tenants,
        }
