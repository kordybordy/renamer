import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from ai_service import (
    OpenAIKeyMissingError,
    AICallResult,
    append_ai_telemetry,
    call_ollama_chat_result,
    call_openai_chat_result,
    read_telemetry_totals,
)
from app_constants import BASE_SYSTEM_PROMPT, FILENAME_RULES
from app_logging import log_exception, log_info
from app_text_utils import clean_party_name, normalize_person_to_given_surname


@dataclass
class FallbackStep:
    provider: str
    model: str


@dataclass
class TenantAIPolicy:
    tenant_id: str = "default"
    allowed_providers: set[str] = field(default_factory=lambda: {"openai", "ollama"})
    allowed_models: dict[str, set[str]] = field(
        default_factory=lambda: {
            "openai": {"gpt-5-nano", "gpt-4.1-mini"},
            "ollama": {"qwen2.5:7b"},
        }
    )
    max_tokens_per_request: int = 0
    max_cost_per_request_usd: float = 0.0
    daily_token_quota: int = 0
    daily_budget_usd: float = 0.0
    fallback_chain: list[FallbackStep] = field(default_factory=list)


def _today_yyyymmdd() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _load_policy_config(policy_config: dict | None, tenant_id: str) -> TenantAIPolicy:
    raw = policy_config
    if raw is None:
        env_payload = os.environ.get("AI_POLICY_CONFIG", "").strip()
        if env_payload:
            try:
                raw = json.loads(env_payload)
            except json.JSONDecodeError:
                raw = None
    if raw is None:
        return TenantAIPolicy(tenant_id=tenant_id)

    tenants = raw.get("tenants", {}) if isinstance(raw, dict) else {}
    tenant_payload = tenants.get(tenant_id, {}) if isinstance(tenants, dict) else {}
    if not isinstance(tenant_payload, dict):
        tenant_payload = {}

    allowed_providers = set(tenant_payload.get("allowed_providers") or ["openai", "ollama"])
    models_payload = tenant_payload.get("allowed_models") or {}
    allowed_models: dict[str, set[str]] = {}
    if isinstance(models_payload, dict):
        for provider, models in models_payload.items():
            if isinstance(models, list):
                allowed_models[provider] = {m for m in models if isinstance(m, str) and m.strip()}

    if not allowed_models:
        allowed_models = {
            "openai": {"gpt-5-nano", "gpt-4.1-mini"},
            "ollama": {"qwen2.5:7b"},
        }

    fallback_chain: list[FallbackStep] = []
    chain_payload = tenant_payload.get("fallback_chain") or []
    if isinstance(chain_payload, list):
        for item in chain_payload:
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or "").strip().lower()
            model = str(item.get("model") or "").strip()
            if provider and model:
                fallback_chain.append(FallbackStep(provider=provider, model=model))

    return TenantAIPolicy(
        tenant_id=tenant_id,
        allowed_providers={p.lower() for p in allowed_providers if isinstance(p, str)},
        allowed_models=allowed_models,
        max_tokens_per_request=max(0, int(tenant_payload.get("max_tokens_per_request") or 0)),
        max_cost_per_request_usd=max(0.0, float(tenant_payload.get("max_cost_per_request_usd") or 0.0)),
        daily_token_quota=max(0, int(tenant_payload.get("daily_token_quota") or 0)),
        daily_budget_usd=max(0.0, float(tenant_payload.get("daily_budget_usd") or 0.0)),
        fallback_chain=fallback_chain,
    )


def _deterministic_metadata_fallback(ocr_text: str, custom_keys: list[str]) -> dict:
    meta: dict[str, str] = {}
    case_match = re.search(r"\b([A-Z]{1,4}\s*\d{1,6}/\d{2,4})\b", ocr_text)
    if case_match:
        meta["case_number"] = case_match.group(1).replace("  ", " ").strip()
    if re.search(r"\b(wezwanie|summons|nakaz)\b", ocr_text, re.IGNORECASE):
        meta["letter_type"] = "wezwanie"
    elif re.search(r"\b(pozew|claim)\b", ocr_text, re.IGNORECASE):
        meta["letter_type"] = "pozew"

    for key in custom_keys:
        if key not in meta:
            meta[key] = ""
    return {k: v for k, v in meta.items() if isinstance(v, str) and v.strip()}


def build_system_prompt(custom_elements: dict[str, str]) -> str:
    extras = ""
    if custom_elements:
        extra_lines = [f'    "{name}": "string"' for name in custom_elements]
        extras = '"custom": {\n' + ",\n".join(extra_lines) + "\n  }"
        details = "\n".join(
            [f'- {name}: {desc or "Return a concise string"}' for name, desc in custom_elements.items()]
        )
        guidance = f"\nCustom fields to add under \"custom\" (as strings):\n{details}\n"
    else:
        guidance = ""
    prompt = BASE_SYSTEM_PROMPT
    if extras:
        prompt = prompt.replace('"custom": {}', extras, 1)
    return prompt + guidance


def extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")
    snippet = raw[start : end + 1]
    return json.loads(snippet)


def parse_json_content(content: str, source: str) -> dict:
    """Parse JSON content from AI responses, stripping code fences if present."""

    raw = (content or "").strip()

    def attempt_parse(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    fence_match = re.search(r"```(?:json)?\\s*(.*?)\\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    parsed = attempt_parse(raw)
    if parsed is None:
        try:
            parsed = extract_json_object(raw)
        except Exception:
            parsed = None

    if parsed is None:
        snippet = raw[:120]
        raise ValueError(
            f"{source} did not return valid JSON. Received: '{snippet or 'empty response'}'"
        )
    return parsed


def parse_ai_metadata(raw: str, custom_keys: list[str]) -> dict:
    """Convert AI JSON into the meta structure expected by the app."""

    try:
        data = parse_json_content(raw, "AI response")
    except Exception as e:
        log_info(f"[AI] JSON parse failed: {e}. Raw response: {raw[:500]}")
        return {}

    meta: dict[str, str] = {}

    def prepare_party(key: str):
        values = data.get(key)
        if not isinstance(values, list):
            return
        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            name = clean_party_name(value)
            name = normalize_person_to_given_surname(name) or name
            if not name:
                continue
            lower_name = name.lower()
            if "dwf poland jamka" in lower_name:
                continue
            if FILENAME_RULES.get("remove_raiffeisen") and "raiffeisen" in lower_name:
                continue
            cleaned.append(name)
        if FILENAME_RULES.get("primary_party_only"):
            cleaned = cleaned[:1]
        max_items = FILENAME_RULES.get("max_parties", len(cleaned))
        if cleaned:
            meta[key] = ", ".join(cleaned[:max_items])

    prepare_party("plaintiff")
    prepare_party("defendant")

    lt = data.get("letter_type")
    if isinstance(lt, str) and lt.strip():
        meta["letter_type"] = lt.strip()

    case_numbers = data.get("case_numbers")
    if isinstance(case_numbers, list) and case_numbers:
        first_case = next((c for c in case_numbers if isinstance(c, str) and c.strip()), "")
        if first_case:
            meta["case_number"] = first_case.strip()

    custom_payload = data.get("custom", {}) if isinstance(data.get("custom"), dict) else {}
    for key in custom_keys:
        val = custom_payload.get(key)
        if not val:
            val = data.get(key)
        if isinstance(val, str) and val.strip():
            meta[key] = val.strip()

    return meta


def _allowed_by_policy(policy: TenantAIPolicy, provider: str, model: str) -> bool:
    if policy.allowed_providers and provider not in policy.allowed_providers:
        return False
    allowed_models = policy.allowed_models.get(provider, set())
    if allowed_models and model not in allowed_models:
        return False
    return True


def _policy_quota_exceeded(policy: TenantAIPolicy, tenant_id: str) -> tuple[bool, str]:
    if not policy.daily_token_quota and not policy.daily_budget_usd:
        return False, ""
    totals = read_telemetry_totals(tenant_id=tenant_id, date_yyyymmdd=_today_yyyymmdd())
    if policy.daily_token_quota and totals["tokens"] >= policy.daily_token_quota:
        return True, f"daily token quota exceeded ({totals['tokens']:.0f}/{policy.daily_token_quota})"
    if policy.daily_budget_usd and totals["cost_usd"] >= policy.daily_budget_usd:
        return True, f"daily budget exceeded (${totals['cost_usd']:.4f}/${policy.daily_budget_usd:.4f})"
    return False, ""


def _build_fallback_chain(backend: str, policy: TenantAIPolicy) -> list[FallbackStep]:
    if policy.fallback_chain:
        chain = list(policy.fallback_chain)
    elif backend == "openai":
        chain = [FallbackStep("openai", "gpt-5-nano"), FallbackStep("openai", "gpt-4.1-mini")]
    elif backend == "ollama":
        chain = [FallbackStep("ollama", "qwen2.5:7b")]
    elif backend == "auto":
        chain = [
            FallbackStep("ollama", "qwen2.5:7b"),
            FallbackStep("openai", "gpt-5-nano"),
            FallbackStep("openai", "gpt-4.1-mini"),
        ]
    else:
        chain = [FallbackStep("openai", "gpt-5-nano")]

    filtered: list[FallbackStep] = []
    for step in chain:
        if _allowed_by_policy(policy, step.provider, step.model):
            filtered.append(step)
    return filtered


def _record_telemetry(
    *,
    tenant_id: str,
    job_id: str,
    source: str,
    success: bool,
    error: str,
    call_result: AICallResult | None,
) -> None:
    payload = {
        "tenant_id": tenant_id,
        "job_id": job_id,
        "date": _today_yyyymmdd(),
        "source": source,
        "success": success,
        "error": error,
    }
    if call_result:
        payload.update(
            {
                "provider": call_result.provider,
                "model": call_result.model,
                "prompt_id": call_result.prompt_id,
                "prompt_version": call_result.prompt_version,
                "request_id": call_result.request_id,
                "prompt_tokens": call_result.prompt_tokens,
                "completion_tokens": call_result.completion_tokens,
                "total_tokens": call_result.total_tokens,
                "estimated_cost_usd": call_result.estimated_cost_usd,
            }
        )
    append_ai_telemetry(payload)


def _invoke_provider(
    *,
    step: FallbackStep,
    prompt: str,
    ocr_text: str,
    max_tokens: int,
    prompt_id: str,
    prompt_version: str,
) -> AICallResult:
    if step.provider == "ollama":
        return call_ollama_chat_result(
            prompt=f"{prompt}\n\n{ocr_text}",
            model=step.model,
            timeout=120,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )
    return call_openai_chat_result(
        system_prompt=prompt,
        user_prompt=ocr_text,
        model=step.model,
        fallback_model=step.model,
        temperature=None,
        fallback_temperature=0.0,
        max_tokens=max_tokens if max_tokens > 0 else None,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        log_info=lambda message: log_info(f"[AI] {message}"),
    )


def extract_metadata_ai(
    ocr_text: str,
    backend: str,
    custom_elements: dict[str, str],
    turbo: bool = False,
    tenant_id: str = "default",
    job_id: str = "",
    policy_config: dict | None = None,
    prompt_id: str = "metadata-extraction",
    prompt_version: str = "v1",
) -> dict:
    """Use AI backend to extract metadata; returns empty dict on failure."""

    del turbo  # Turbo mode bypassed here to enforce policy and deterministic fallbacks.

    if not ocr_text.strip():
        return {}

    policy = _load_policy_config(policy_config, tenant_id)
    prompt = build_system_prompt(custom_elements)
    chain = _build_fallback_chain(backend, policy)

    if not chain:
        log_info(f"[AI] No allowed provider/model chain for tenant '{tenant_id}'.")
        meta = _deterministic_metadata_fallback(ocr_text, list(custom_elements.keys()))
        _record_telemetry(
            tenant_id=tenant_id,
            job_id=job_id,
            source="deterministic-fallback",
            success=bool(meta),
            error="policy denied all provider/model pairs",
            call_result=None,
        )
        return meta

    quota_hit, quota_reason = _policy_quota_exceeded(policy, tenant_id)
    if quota_hit:
        log_info(f"[AI] Skipping provider calls for tenant '{tenant_id}': {quota_reason}")
        meta = _deterministic_metadata_fallback(ocr_text, list(custom_elements.keys()))
        _record_telemetry(
            tenant_id=tenant_id,
            job_id=job_id,
            source="deterministic-fallback",
            success=bool(meta),
            error=quota_reason,
            call_result=None,
        )
        return meta

    for step in chain:
        call_result: AICallResult | None = None
        try:
            call_result = _invoke_provider(
                step=step,
                prompt=prompt,
                ocr_text=ocr_text,
                max_tokens=policy.max_tokens_per_request,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
            )

            if policy.max_cost_per_request_usd and call_result.estimated_cost_usd > policy.max_cost_per_request_usd:
                msg = (
                    f"cost cap exceeded for {step.provider}/{step.model}: "
                    f"${call_result.estimated_cost_usd:.4f} > ${policy.max_cost_per_request_usd:.4f}"
                )
                _record_telemetry(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    source="ai-call",
                    success=False,
                    error=msg,
                    call_result=call_result,
                )
                log_info(f"[AI] {msg}")
                continue

            meta = parse_ai_metadata(call_result.content, list(custom_elements.keys()))
            _record_telemetry(
                tenant_id=tenant_id,
                job_id=job_id,
                source="ai-call",
                success=bool(meta),
                error="" if meta else "empty metadata",
                call_result=call_result,
            )
            if meta:
                log_info(f"[AI] metadata extracted using {step.provider}/{step.model}")
                return meta
        except OpenAIKeyMissingError as exc:
            log_info(f"[AI] {exc}")
            _record_telemetry(
                tenant_id=tenant_id,
                job_id=job_id,
                source="ai-call",
                success=False,
                error=str(exc),
                call_result=call_result,
            )
        except Exception as e:
            log_exception(e)
            _record_telemetry(
                tenant_id=tenant_id,
                job_id=job_id,
                source="ai-call",
                success=False,
                error=str(e),
                call_result=call_result,
            )

    meta = _deterministic_metadata_fallback(ocr_text, list(custom_elements.keys()))
    _record_telemetry(
        tenant_id=tenant_id,
        job_id=job_id,
        source="deterministic-fallback",
        success=bool(meta),
        error="provider chain exhausted",
        call_result=None,
    )
    return meta
