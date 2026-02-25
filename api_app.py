from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Literal

from fastapi import Cookie, Depends, FastAPI, File, Header, Path, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr, Field

from api_service import AccessContext, ApiServiceError, RenamerApiService, UserRole

app = FastAPI(
    title="Renamer API",
    version="1.1.0",
    description=(
        "Tenant-scoped API for creating matters, uploading PDF files, submitting "
        "rename/distribution jobs, polling status, receiving webhook callbacks, "
        "and downloading job artifacts."
    ),
)

service = RenamerApiService()
JWT_SECRET = os.getenv("RENAMER_JWT_SECRET", "renamer-dev-secret")
JWT_AUDIENCE = "renamer-api"
JWT_ISSUER = "renamer-auth"
SESSION_COOKIE = "renamer_session_id"
_session_store: dict[str, dict[str, str]] = {}


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _jwt_encode(payload: dict[str, Any]) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def _jwt_decode(token: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise ApiServiceError("invalid_token", "Token format is invalid.", 401) from exc
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ApiServiceError("invalid_token", "Token signature validation failed.", 401)
    payload = json.loads(_b64url_decode(payload_b64))
    now = int(time.time())
    if int(payload.get("exp", 0)) < now:
        raise ApiServiceError("token_expired", "Token has expired.", 401)
    if payload.get("aud") != JWT_AUDIENCE or payload.get("iss") != JWT_ISSUER:
        raise ApiServiceError("invalid_token", "Token audience/issuer mismatch.", 401)
    return payload


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


class UploadResponse(BaseModel):
    uploaded: list[str]


class JobSubmitRequest(BaseModel):
    job_type: Literal["rename", "distribution"]
    options: dict[str, Any] = Field(default_factory=dict)
    webhook_url: str | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


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


class LoginRequest(BaseModel):
    tenant_id: str
    user_id: str
    email: EmailStr
    role: UserRole
    provider: Literal["local", "saml", "oidc"] = "local"


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    session_id: str




class OnboardingCreateRequest(BaseModel):
    workspace_name: str = Field(min_length=1, max_length=200)
    admin_email: EmailStr
    sso_provider: Literal["saml", "oidc"] | None = None


class OnboardingResponse(BaseModel):
    tenant_id: str
    workspace_name: str
    onboarding: dict[str, Any]


class OnboardingStepUpdateRequest(BaseModel):
    done: bool


class UsageSummaryResponse(BaseModel):
    tenant_id: str
    plan: str
    usage: dict[str, float]
    limits: dict[str, dict[str, float]]
    notifications: list[dict[str, Any]]


class InvoiceSummaryResponse(BaseModel):
    tenant_id: str
    period: str
    currency: str
    line_items: list[dict[str, Any]]
    total_amount: float


class SupportDashboardResponse(BaseModel):
    generated_at: str
    tenant_count: int
    tenants: list[dict[str, Any]]
class SsoProviderConfig(BaseModel):
    provider: Literal["saml", "oidc"]
    display_name: str
    metadata_url: str
    client_id: str | None = None


class SsoStartResponse(BaseModel):
    provider: str
    authorization_url: str
    state: str


class SsoCallbackRequest(BaseModel):
    state: str
    tenant_id: str
    user_id: str
    email: EmailStr
    role: UserRole


SSO_PROVIDERS: dict[str, SsoProviderConfig] = {
    "saml": SsoProviderConfig(provider="saml", display_name="Corporate SAML", metadata_url="https://example.com/saml/metadata"),
    "oidc": SsoProviderConfig(provider="oidc", display_name="Corporate OIDC", metadata_url="https://example.com/.well-known/openid-configuration", client_id="renamer-api"),
}


def _build_token(*, tenant_id: str, user_id: str, role: UserRole, email: str) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role.value,
        "email": email,
        "iat": now,
        "exp": now + 3600,
        "aud": JWT_AUDIENCE,
        "iss": JWT_ISSUER,
    }
    return _jwt_encode(payload)


def _parse_access_context(payload: dict[str, Any]) -> AccessContext:
    tenant_id = str(payload.get("tenant_id") or "")
    user_id = str(payload.get("sub") or "")
    role_value = str(payload.get("role") or "")
    if not tenant_id or not user_id or not role_value:
        raise ApiServiceError("invalid_token", "Missing token claims.", 401)
    try:
        role = UserRole(role_value)
    except ValueError as exc:
        raise ApiServiceError("invalid_token", "Unsupported role claim.", 401) from exc
    return AccessContext(tenant_id=tenant_id, user_id=user_id, role=role)


def _store_session(access: AccessContext, email: str) -> str:
    session_id = str(uuid.uuid4())
    _session_store[session_id] = {
        "tenant_id": access.tenant_id,
        "user_id": access.user_id,
        "role": access.role.value,
        "email": email,
    }
    return session_id


def get_access_context(
    authorization: str | None = Header(default=None),
    session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> AccessContext:
    if authorization and authorization.lower().startswith("bearer "):
        payload = _jwt_decode(authorization.split(" ", 1)[1].strip())
        return _parse_access_context(payload)
    if session_id:
        session = _session_store.get(session_id)
        if not session:
            raise ApiServiceError("session_not_found", "Session is invalid or expired.", 401)
        payload = {
            "tenant_id": session["tenant_id"],
            "sub": session["user_id"],
            "role": session["role"],
        }
        return _parse_access_context(payload)
    raise ApiServiceError("authentication_required", "Provide JWT bearer token or session cookie.", 401)


@app.exception_handler(ApiServiceError)
async def handle_api_service_error(_: Request, exc: ApiServiceError):
    body = ErrorResponse(code=exc.code, message=exc.message, details=exc.details).model_dump()
    return JSONResponse(status_code=exc.status_code, content=body)


@app.post("/api/v1/auth/token", response_model=LoginResponse, tags=["auth"])
def issue_token(request: LoginRequest):
    service.upsert_firm_user(request.tenant_id, request.user_id, str(request.email), request.role)
    token = _build_token(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        role=request.role,
        email=str(request.email),
    )
    access = AccessContext(tenant_id=request.tenant_id, user_id=request.user_id, role=request.role)
    session_id = _store_session(access, str(request.email))
    return LoginResponse(access_token=token, expires_in=3600, session_id=session_id)


@app.get("/api/v1/auth/sso/providers", response_model=list[SsoProviderConfig], tags=["auth"])
def list_sso_providers():
    return list(SSO_PROVIDERS.values())


@app.get("/api/v1/auth/sso/{provider}/start", response_model=SsoStartResponse, tags=["auth"])
def start_sso(provider: str):
    config = SSO_PROVIDERS.get(provider)
    if not config:
        raise ApiServiceError("provider_not_found", "Unknown SSO provider.", 404)
    state = str(uuid.uuid4())
    return SsoStartResponse(
        provider=provider,
        state=state,
        authorization_url=f"{config.metadata_url}?state={state}",
    )


@app.post("/api/v1/auth/sso/{provider}/callback", response_model=LoginResponse, tags=["auth"])
def complete_sso(provider: str, request: SsoCallbackRequest):
    if provider not in SSO_PROVIDERS:
        raise ApiServiceError("provider_not_found", "Unknown SSO provider.", 404)
    service.upsert_firm_user(request.tenant_id, request.user_id, str(request.email), request.role)
    token = _build_token(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        role=request.role,
        email=str(request.email),
    )
    access = AccessContext(tenant_id=request.tenant_id, user_id=request.user_id, role=request.role)
    session_id = _store_session(access, str(request.email))
    return LoginResponse(access_token=token, expires_in=3600, session_id=session_id)


@app.post("/api/v1/tenants/onboarding", response_model=OnboardingResponse, tags=["tenants"])
def create_tenant_onboarding(request: OnboardingCreateRequest):
    tenant = service.create_tenant_onboarding(
        workspace_name=request.workspace_name,
        admin_email=str(request.admin_email),
        sso_provider=request.sso_provider,
    )
    return OnboardingResponse(
        tenant_id=tenant.tenant_id,
        workspace_name=tenant.name,
        onboarding=tenant.onboarding,
    )


@app.get("/api/v1/tenants/{tenant_id}/onboarding", response_model=OnboardingResponse, tags=["tenants"])
def get_tenant_onboarding(tenant_id: str, access: AccessContext = Depends(get_access_context)):
    service._require_tenant_scope(tenant_id, access)
    onboarding = service.get_onboarding(tenant_id)
    tenant = service._ensure_tenant(tenant_id)
    return OnboardingResponse(tenant_id=tenant_id, workspace_name=tenant.name, onboarding=onboarding)


@app.patch("/api/v1/tenants/{tenant_id}/onboarding/{step_id}", response_model=OnboardingResponse, tags=["tenants"])
def update_tenant_onboarding_step(
    tenant_id: str,
    step_id: str,
    request: OnboardingStepUpdateRequest,
    access: AccessContext = Depends(get_access_context),
):
    service._require_tenant_scope(tenant_id, access)
    onboarding = service.complete_onboarding_step(tenant_id, step_id, request.done)
    tenant = service._ensure_tenant(tenant_id)
    return OnboardingResponse(tenant_id=tenant_id, workspace_name=tenant.name, onboarding=onboarding)


@app.get("/api/v1/tenants/{tenant_id}/usage", response_model=UsageSummaryResponse, tags=["billing"])
def get_usage_summary(tenant_id: str, access: AccessContext = Depends(get_access_context)):
    return UsageSummaryResponse(**service.get_usage_summary(access, tenant_id))


@app.get("/api/v1/tenants/{tenant_id}/usage/invoice", response_model=InvoiceSummaryResponse, tags=["billing"])
def get_invoice_summary(tenant_id: str, access: AccessContext = Depends(get_access_context)):
    return InvoiceSummaryResponse(**service.get_invoice_summary(access, tenant_id))


@app.get("/api/v1/internal/support/dashboard", response_model=SupportDashboardResponse, tags=["internal"])
def get_support_dashboard(_: str = Header(default="", alias="X-Internal-Token")):
    return SupportDashboardResponse(**service.get_support_dashboard())


@app.post(
    "/api/v1/tenants/{tenant_id}/matters",
    response_model=MatterCreateResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["matters"],
)
def create_matter(tenant_id: str, request: MatterCreateRequest, access: AccessContext = Depends(get_access_context)):
    matter = service.create_matter(access, tenant_id, request.name, request.metadata)
    return MatterCreateResponse(
        tenant_id=matter.tenant_id,
        matter_id=matter.matter_id,
        name=matter.name,
        metadata=matter.metadata,
        created_at=matter.created_at,
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
    access: AccessContext = Depends(get_access_context),
):
    file_payload: list[tuple[str, bytes]] = []
    for file in files:
        file_payload.append((file.filename or "uploaded.pdf", await file.read()))
    uploaded = service.store_uploads(access, tenant_id, matter_id, file_payload)
    return UploadResponse(uploaded=uploaded)


@app.post(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/jobs",
    response_model=JobSubmitResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["jobs"],
)
def submit_job(tenant_id: str, matter_id: str, request: JobSubmitRequest, access: AccessContext = Depends(get_access_context)):
    job = service.submit_job(
        access=access,
        tenant_id=tenant_id,
        matter_id=matter_id,
        job_type=request.job_type,
        payload=request.options,
        webhook_url=request.webhook_url,
        idempotency_key=request.idempotency_key,
    )
    return JobSubmitResponse(
        tenant_id=job.tenant_id,
        matter_id=job.matter_id,
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
def get_job_status(tenant_id: str, matter_id: str, job_id: str, access: AccessContext = Depends(get_access_context)):
    job = service.get_job(access, tenant_id, matter_id, job_id)
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
def retry_webhook(tenant_id: str, matter_id: str, job_id: str, access: AccessContext = Depends(get_access_context)):
    result = service.trigger_callback(access, tenant_id, matter_id, job_id)
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
    access: AccessContext = Depends(get_access_context),
):
    path = service.get_result_file_path(access, tenant_id, matter_id, job_id, result_name)
    return FileResponse(path=path, filename=result_name, media_type="application/pdf")


@app.get(
    "/api/v1/tenants/{tenant_id}/matters/{matter_id}/jobs/{job_id}/audit",
    responses={404: {"model": ErrorResponse}},
    tags=["results"],
)
def get_audit(tenant_id: str, matter_id: str, job_id: str, access: AccessContext = Depends(get_access_context)):
    return service.get_audit(access, tenant_id, matter_id, job_id)
