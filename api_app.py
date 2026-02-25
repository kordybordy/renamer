from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, File, Path, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api_service import ApiServiceError, RenamerApiService

app = FastAPI(
    title="Renamer API",
    version="1.0.0",
    description=(
        "Tenant-scoped API for creating matters, uploading PDF files, submitting "
        "rename/distribution jobs, polling status, receiving webhook callbacks, "
        "and downloading job artifacts."
    ),
)

service = RenamerApiService()


class ErrorResponse(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="User-safe error message")
    details: dict[str, Any] = Field(default_factory=dict)


class MatterCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatterCreateResponse(BaseModel):
    tenant_id: str
    matter_id: str
    name: str
    metadata: dict[str, Any]
    created_at: str
    encryption: dict[str, Any]
    retention: dict[str, Any]


class UploadResponse(BaseModel):
    uploaded: list[str]


class JobSubmitRequest(BaseModel):
    job_type: Literal["rename", "distribution"]
    options: dict[str, Any] = Field(default_factory=dict)
    webhook_url: str | None = None


class JobSubmitResponse(BaseModel):
    tenant_id: str
    matter_id: str
    job_id: str
    job_type: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    tenant_id: str
    matter_id: str
    job_id: str
    job_type: str
    status: str
    created_at: str
    updated_at: str
    result_files: list[dict[str, str]]
    errors: list[dict[str, str]]
    summary: dict[str, Any]


class WebhookRetryResponse(BaseModel):
    status_code: int
    delivered_at: str


class LegalHoldRequest(BaseModel):
    enabled: bool


class LegalHoldResponse(BaseModel):
    tenant_id: str
    matter_id: str
    legal_hold: bool


@app.exception_handler(ApiServiceError)
async def handle_api_service_error(_: Request, exc: ApiServiceError):
    body = ErrorResponse(code=exc.code, message=exc.message, details=exc.details).model_dump()
    return JSONResponse(status_code=exc.status_code, content=body)


@app.middleware("http")
async def enforce_tls(request: Request, call_next):
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if request.url.scheme != "https" and forwarded_proto.lower() != "https":
        return JSONResponse(
            status_code=400,
            content={"code": "tls_required", "message": "TLS is required for all API traffic.", "details": {}},
        )
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.post(
    "/api/v1/tenants/{tenant_id}/matters",
    response_model=MatterCreateResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["matters"],
)
def create_matter(tenant_id: str, request: MatterCreateRequest):
    matter = service.create_matter(tenant_id, request.name, request.metadata)
    return MatterCreateResponse(
        tenant_id=matter.tenant_id,
        matter_id=matter.matter_id,
        name=matter.name,
        metadata=matter.metadata,
        created_at=matter.created_at,
        encryption={"at_rest": True, "kms_key_id": matter.encryption_key_id},
        retention={"ocr_ai_days": matter.retention_days, "legal_hold": matter.legal_hold},
    )


@app.post(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/files",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["files"],
)
async def upload_files(
    tenant_id: str,
    matter_id: str,
    files: list[UploadFile] = File(...),
):
    file_payload: list[tuple[str, bytes]] = []
    for file in files:
        file_payload.append((file.filename or "uploaded.pdf", await file.read()))
    uploaded = service.store_uploads(tenant_id, matter_id, file_payload)
    return UploadResponse(uploaded=uploaded)


@app.post(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/jobs",
    response_model=JobSubmitResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["jobs"],
)
def submit_job(tenant_id: str, matter_id: str, request: JobSubmitRequest):
    job = service.submit_job(
        tenant_id=tenant_id,
        matter_id=matter_id,
        job_type=request.job_type,
        payload=request.options,
        webhook_url=request.webhook_url,
    )
    return JobSubmitResponse(
        tenant_id=tenant_id,
        matter_id=matter_id,
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
    )


@app.get(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["jobs"],
)
def get_job_status(tenant_id: str, matter_id: str, job_id: str):
    job = service.get_job(tenant_id, matter_id, job_id)
    return JobStatusResponse(
        tenant_id=job.tenant_id,
        matter_id=job.matter_id,
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result_files=job.result_files,
        errors=job.errors,
        summary=job.summary,
    )


@app.post(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/jobs/{job_id}/callbacks",
    response_model=WebhookRetryResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["jobs"],
)
def retry_webhook(tenant_id: str, matter_id: str, job_id: str):
    result = service.trigger_callback(tenant_id, matter_id, job_id)
    return WebhookRetryResponse(**result)


@app.get(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/jobs/{job_id}/results/{result_name}",
    responses={404: {"model": ErrorResponse}},
    tags=["results"],
)
def download_result(
    tenant_id: str,
    matter_id: str,
    job_id: str,
    result_name: str = Path(..., description="File name returned in job result_files"),
):
    payload = service.read_result_file(tenant_id, matter_id, job_id, result_name)
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{result_name}"'},
    )


@app.get(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/jobs/{job_id}/audit",
    responses={404: {"model": ErrorResponse}},
    tags=["results"],
)
def get_audit(tenant_id: str, matter_id: str, job_id: str):
    return service.get_audit(tenant_id, matter_id, job_id)


@app.delete(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/files/{file_name}",
    responses={404: {"model": ErrorResponse}},
    tags=["files"],
)
def delete_upload(
    tenant_id: str,
    matter_id: str,
    file_name: str = Path(..., description="Uploaded file name"),
):
    service.delete_upload(tenant_id, matter_id, file_name)
    return {"deleted": file_name}


@app.post(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/legal-hold",
    response_model=LegalHoldResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["matters"],
)
def set_legal_hold(tenant_id: str, matter_id: str, request: LegalHoldRequest):
    matter = service.set_legal_hold(tenant_id, matter_id, request.enabled)
    return LegalHoldResponse(tenant_id=tenant_id, matter_id=matter_id, legal_hold=matter.legal_hold)


@app.get(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/audit-report",
    responses={404: {"model": ErrorResponse}},
    tags=["results"],
)
def download_audit_report(tenant_id: str, matter_id: str):
    path = service.export_matter_audit_report(tenant_id, matter_id)
    with open(path, "rb") as handle:
        payload = handle.read()
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="audit_report.json"'},
    )
