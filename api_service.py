from __future__ import annotations

import json
import os
import threading
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from hashlib import sha256
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import tempfile
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
class MatterRecord:
    tenant_id: str
    matter_id: str
    name: str
    metadata: dict[str, Any]
    created_at: str
    input_dir: str
    encryption_key_id: str
    key_material: str
    retention_days: int
    legal_hold: bool = False
    ai_audit_mode: str = "minimal"
    redact_fields: list[str] = field(default_factory=list)


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
    result_files: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    audit_path: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


class RenamerApiService:
    def __init__(self, storage_root: str = "api_data") -> None:
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._core = None
        self._deps = None
        self._lock = threading.Lock()
        self._matters: dict[tuple[str, str], MatterRecord] = {}
        self._jobs: dict[tuple[str, str, str], JobRecord] = {}

    def _derive_matter_key(self, matter: MatterRecord) -> bytes:
        return sha256(f"{matter.tenant_id}:{matter.matter_id}:{matter.key_material}".encode("utf-8")).digest()

    def _encrypt_bytes(self, matter: MatterRecord, payload: bytes, aad: str) -> bytes:
        key = self._derive_matter_key(matter)
        aad_key = sha256(aad.encode("utf-8")).digest()
        out = bytes(value ^ key[idx % len(key)] ^ aad_key[idx % len(aad_key)] for idx, value in enumerate(payload))
        return urlsafe_b64encode(out)

    def _decrypt_bytes(self, matter: MatterRecord, payload: bytes, aad: str) -> bytes:
        raw = urlsafe_b64decode(payload)
        key = self._derive_matter_key(matter)
        aad_key = sha256(aad.encode("utf-8")).digest()
        return bytes(value ^ key[idx % len(key)] ^ aad_key[idx % len(aad_key)] for idx, value in enumerate(raw))

    def _write_encrypted_file(self, matter: MatterRecord, target: Path, payload: bytes) -> None:
        encrypted = self._encrypt_bytes(matter, payload, target.name)
        target.write_bytes(encrypted)

    def _read_encrypted_file(self, matter: MatterRecord, target: Path) -> bytes:
        return self._decrypt_bytes(matter, target.read_bytes(), target.name)

    def _apply_redaction(self, payload: dict[str, Any], redact_fields: list[str]) -> dict[str, Any]:
        if not redact_fields:
            return payload
        redacted = dict(payload)
        for key in redact_fields:
            if key in redacted:
                redacted[key] = "[REDACTED]"
        return redacted

    def _append_audit_event(self, matter: MatterRecord, event_type: str, details: dict[str, Any] | None = None) -> None:
        audit_dir = Path(matter.input_dir).parent / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / "events.jsonl"
        previous_hash = ""
        if audit_path.exists():
            lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                previous_hash = json.loads(lines[-1]).get("entry_hash", "")
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": utc_now_iso(),
            "tenant_id": matter.tenant_id,
            "matter_id": matter.matter_id,
            "event_type": event_type,
            "previous_hash": previous_hash,
            "details": details or {},
        }
        event_hash = sha256(json.dumps(event, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        event["entry_hash"] = event_hash
        with open(audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _ensure_core(self):
        if self._deps is None:
            self._deps = _CoreDeps()
        if self._core is None:
            self._core = self._deps.CoreApplicationService()
        return self._deps

    def create_matter(self, tenant_id: str, name: str, metadata: dict[str, Any] | None) -> MatterRecord:
        matter_id = str(uuid.uuid4())
        matter_dir = self.storage_root / tenant_id / matter_id
        input_dir = matter_dir / "uploads"
        input_dir.mkdir(parents=True, exist_ok=True)
        metadata = metadata or {}
        security = dict(metadata.get("security") or {})
        retention = dict(metadata.get("retention") or {})
        ai_audit = dict(metadata.get("ai_audit") or {})
        matter = MatterRecord(
            tenant_id=tenant_id,
            matter_id=matter_id,
            name=name,
            metadata=metadata,
            created_at=utc_now_iso(),
            input_dir=str(input_dir),
            encryption_key_id=str(security.get("kms_key_id") or f"kms://tenant/{tenant_id}/matter/{matter_id}"),
            key_material=str(uuid.uuid4()),
            retention_days=max(1, int(retention.get("ocr_ai_days", 30))),
            legal_hold=bool(retention.get("legal_hold", False)),
            ai_audit_mode=str(ai_audit.get("mode", "minimal")),
            redact_fields=list(ai_audit.get("redact_fields") or []),
        )
        with self._lock:
            self._matters[(tenant_id, matter_id)] = matter
        self._append_audit_event(
            matter,
            "matter_created",
            {
                "name": matter.name,
                "encryption": {"provider": "object-storage+kms", "kms_key_id": matter.encryption_key_id},
                "retention_days": matter.retention_days,
                "legal_hold": matter.legal_hold,
            },
        )
        return matter

    def get_matter(self, tenant_id: str, matter_id: str) -> MatterRecord:
        matter = self._matters.get((tenant_id, matter_id))
        if not matter:
            raise ApiServiceError(
                code="matter_not_found",
                message="Matter could not be found for the tenant.",
                status_code=404,
            )
        return matter

    def store_uploads(self, tenant_id: str, matter_id: str, files: list[tuple[str, bytes]]) -> list[str]:
        matter = self.get_matter(tenant_id, matter_id)
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
            self._write_encrypted_file(matter, target, content)
            stored.append(target.name)
            self._append_audit_event(
                matter,
                "upload",
                {
                    "filename": target.name,
                    "bytes": len(content),
                    "encryption": {"at_rest": True, "kms_key_id": matter.encryption_key_id},
                },
            )
        return stored

    def delete_upload(self, tenant_id: str, matter_id: str, filename: str) -> None:
        matter = self.get_matter(tenant_id, matter_id)
        target = Path(matter.input_dir) / Path(filename).name
        if not target.exists():
            raise ApiServiceError(code="file_not_found", message="The requested file does not exist.", status_code=404)
        target.unlink()
        self._append_audit_event(matter, "delete", {"filename": target.name})

    def submit_job(
        self,
        tenant_id: str,
        matter_id: str,
        job_type: str,
        payload: dict[str, Any],
        webhook_url: str | None,
    ) -> JobRecord:
        matter = self.get_matter(tenant_id, matter_id)
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

        job_id = str(uuid.uuid4())
        job = JobRecord(
            tenant_id=tenant_id,
            matter_id=matter_id,
            job_id=job_id,
            job_type=job_type,
            status="queued",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            webhook_url=webhook_url,
        )
        with self._lock:
            self._jobs[(tenant_id, matter_id, job_id)] = job

        worker = threading.Thread(
            target=self._run_job,
            args=(job, matter, uploaded_pdfs, payload),
            daemon=True,
        )
        worker.start()
        return job

    def get_job(self, tenant_id: str, matter_id: str, job_id: str) -> JobRecord:
        job = self._jobs.get((tenant_id, matter_id, job_id))
        if not job:
            raise ApiServiceError(
                code="job_not_found",
                message="Job could not be found for the tenant and matter.",
                status_code=404,
            )
        return job

    def get_result_file_path(self, tenant_id: str, matter_id: str, job_id: str, result_name: str) -> str:
        job = self.get_job(tenant_id, matter_id, job_id)
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

    def read_result_file(self, tenant_id: str, matter_id: str, job_id: str, result_name: str) -> bytes:
        matter = self.get_matter(tenant_id, matter_id)
        path = Path(self.get_result_file_path(tenant_id, matter_id, job_id, result_name))
        if not path.exists():
            raise ApiServiceError(code="result_not_found", message="Result file not found.", status_code=404)
        payload = self._read_encrypted_file(matter, path)
        self._append_audit_event(matter, "download", {"filename": result_name, "job_id": job_id})
        return payload

    def get_audit(self, tenant_id: str, matter_id: str, job_id: str) -> dict[str, Any]:
        job = self.get_job(tenant_id, matter_id, job_id)
        if not job.audit_path or not os.path.isfile(job.audit_path):
            raise ApiServiceError(
                code="audit_not_ready",
                message="Audit data is not available for this job.",
                status_code=404,
            )
        with open(job.audit_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def trigger_callback(self, tenant_id: str, matter_id: str, job_id: str) -> dict[str, Any]:
        job = self.get_job(tenant_id, matter_id, job_id)
        if not job.webhook_url:
            raise ApiServiceError(
                code="webhook_not_configured",
                message="No webhook URL configured for this job.",
                status_code=400,
            )
        return self._post_webhook(job)

    def set_legal_hold(self, tenant_id: str, matter_id: str, enabled: bool) -> MatterRecord:
        matter = self.get_matter(tenant_id, matter_id)
        matter.legal_hold = enabled
        self._append_audit_event(matter, "legal_hold_updated", {"enabled": enabled})
        return matter

    def export_matter_audit_report(self, tenant_id: str, matter_id: str) -> str:
        matter = self.get_matter(tenant_id, matter_id)
        audit_dir = Path(matter.input_dir).parent / "audit"
        events_path = audit_dir / "events.jsonl"
        report_path = audit_dir / "audit_report.json"
        events: list[dict[str, Any]] = []
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        report = {
            "tenant_id": tenant_id,
            "matter_id": matter_id,
            "generated_at": utc_now_iso(),
            "legal_hold": matter.legal_hold,
            "event_count": len(events),
            "events": events,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(report_path)

    def _run_job(self, job: JobRecord, matter: MatterRecord, uploaded_pdfs: list[str], payload: dict[str, Any]) -> None:
        self._set_job_status(job, "running")
        work_dir = Path(tempfile.mkdtemp(prefix=f"renamer_{job.job_id}_"))
        decrypted_uploads: list[str] = []
        try:
            for encrypted_pdf in uploaded_pdfs:
                source_path = Path(encrypted_pdf)
                decrypted_path = work_dir / source_path.name
                decrypted_path.write_bytes(self._read_encrypted_file(matter, source_path))
                decrypted_uploads.append(str(decrypted_path))
            if job.job_type == "rename":
                self._execute_rename_job(job, matter, decrypted_uploads, payload)
            else:
                self._execute_distribution_job(job, matter, decrypted_uploads, payload)
            self._set_job_status(job, "completed")
        except Exception as exc:  # noqa: BLE001
            job.errors.append({"code": "job_failed", "message": str(exc)})
            self._set_job_status(job, "failed")
        finally:
            for path in work_dir.glob("*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
            work_dir.rmdir()
            self._purge_expired_artifacts(matter)
            self._write_audit_log(job)
            if job.webhook_url:
                try:
                    self._post_webhook(job)
                except Exception as exc:  # noqa: BLE001
                    job.errors.append({"code": "webhook_delivery_failed", "message": str(exc)})

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

        for pdf_path in uploaded_pdfs:
            source_name = Path(pdf_path).name
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
            requested = {"source": source_name, "backend": backend, "turbo_mode": turbo_mode}
            self._append_audit_event(
                matter,
                "ai_extraction_request",
                self._apply_redaction(requested, matter.redact_fields),
            )
            override = str(payload.get("filename_overrides", {}).get(source_name, "")).strip()
            target_name = f"{override or response.filename}.pdf"
            target_path = result_dir / target_name
            pdf_payload = Path(pdf_path).read_bytes()
            self._write_encrypted_file(matter, target_path, pdf_payload)
            job.result_files.append({"name": target_name, "path": str(target_path), "source": source_name})
            self._append_audit_event(
                matter,
                "ai_extraction_response",
                self._apply_redaction({"source": source_name, "generated_filename": response.filename}, matter.redact_fields),
            )
            decision_event = "filename_override" if override else "filename_approval"
            self._append_audit_event(
                matter,
                decision_event,
                {"source": source_name, "final_filename": target_name},
            )

        job.summary = {
            "processed": len(uploaded_pdfs),
            "results": len(job.result_files),
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

        plan_response = self._core.plan_distribution(
            deps.DistributionPlanRequest(
                input_dir=matter.input_dir,
                pdf_files=uploaded_pdfs,
                case_root=case_root,
                config=cfg,
                ai_provider=cfg.ai_provider,
                audit_log_path=str(plan_audit_path),
            )
        )
        self._append_audit_event(matter, "distribution_plan_generated", {"planned": len(plan_response.plan)})

        job.summary = {
            "planned": len(plan_response.plan),
            "auto_candidates": sum(1 for item in plan_response.plan if item.status == "auto"),
        }

        if bool(payload.get("apply", False)):
            apply_audit_path = audit_dir / f"{job.job_id}_apply.json"
            self._core.apply_distribution_plan(
                deps.DistributionApplyRequest(
                    input_dir=matter.input_dir,
                    case_root=case_root,
                    config=cfg,
                    ai_provider=cfg.ai_provider,
                    plan=plan_response.plan,
                    auto_only=bool(payload.get("auto_only", False)),
                    audit_log_path=str(apply_audit_path),
                )
            )
            job.summary["apply_audit"] = str(apply_audit_path)
            self._append_audit_event(matter, "distribution_apply", {"auto_only": bool(payload.get("auto_only", False))})

        job.audit_path = str(plan_audit_path)

    def _write_audit_log(self, job: JobRecord) -> None:
        audit_dir = self.storage_root / job.tenant_id / job.matter_id / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{job.job_id}_job.json"
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "job": asdict(job),
                    "written_at": utc_now_iso(),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        if not job.audit_path:
            job.audit_path = str(audit_path)

    def _purge_expired_artifacts(self, matter: MatterRecord) -> None:
        if matter.legal_hold:
            return
        artifacts_dir = Path(matter.input_dir).parent / "artifacts"
        if not artifacts_dir.exists():
            return
        cutoff = datetime.now(timezone.utc).timestamp() - (matter.retention_days * 86400)
        for artifact in artifacts_dir.glob("*.json"):
            if artifact.stat().st_mtime < cutoff:
                artifact.unlink(missing_ok=True)
                self._append_audit_event(matter, "retention_purge", {"artifact": artifact.name})

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
