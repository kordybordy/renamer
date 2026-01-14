from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .ai_tiebreaker import choose_best_candidate
from .indexer import build_folder_index
from .models import DocumentMeta, DistributionPlanItem, MatchCandidate
from .scorer import (
    DEFAULT_STOPWORDS,
    ScoreSummary,
    extract_surnames_from_parties,
    score_document,
    tokenize,
)
from .safety import resolve_destination_path, safe_copy


@dataclass
class DistributionConfig:
    auto_threshold: float = 70.0
    gap_threshold: float = 15.0
    ai_threshold: float = 0.7
    top_k: int = 15
    tie_epsilon: float = 5.0
    ai_max_candidates: int = 5
    stopwords: list[str] = None
    unmatched_policy: str = "leave"

    def normalized_stopwords(self) -> list[str]:
        if self.stopwords is None:
            return DEFAULT_STOPWORDS[:]
        return self.stopwords


class DistributionEngine:
    def __init__(
        self,
        *,
        input_folder: str,
        case_root: str,
        config: DistributionConfig,
        ai_provider: str,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.input_folder = input_folder
        self.case_root = case_root
        self.config = config
        self.ai_provider = ai_provider
        self.logger = logger

    def log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def build_index(self) -> list:
        return build_folder_index(self.case_root, self.config.normalized_stopwords())

    def _load_sidecar_meta(self, pdf_path: str) -> dict:
        base = os.path.splitext(pdf_path)[0]
        candidates = [f"{base}.json", f"{pdf_path}.json"]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        return json.load(handle)
                except Exception:
                    return {}
        return {}

    def _parse_case_numbers(self, text: str) -> list[str]:
        if not text:
            return []
        pattern = re.compile(r"\b[IVX]{1,4}\s*[A-Z]{1,4}\s*\d{1,4}/\d{2,4}\b", re.IGNORECASE)
        return [match.group(0).strip() for match in pattern.finditer(text)]

    def _parse_parties_from_filename(self, filename: str) -> list[str]:
        base = os.path.splitext(os.path.basename(filename))[0]
        base = re.sub(r"[_]+", " ", base)
        base = re.sub(r"\s+", " ", base).strip()
        if not base:
            return []
        head = re.split(r"\s+-\s+", base, maxsplit=1)[0]
        parts = [part.strip() for part in re.split(r"[;,]", head) if part.strip()]
        return parts

    def _filter_opposing_parties(self, names: list[str]) -> list[str]:
        filtered: list[str] = []
        for name in names:
            lower = name.lower()
            if "dwf poland jamka" in lower:
                continue
            if "raiffeisen" in lower:
                continue
            if name not in filtered:
                filtered.append(name)
        return filtered

    def build_document_meta(self, pdf_path: str, filename: str) -> DocumentMeta:
        sidecar = self._load_sidecar_meta(pdf_path)
        plaintiffs = []
        defendants = []
        case_numbers: list[str] = []
        letter_type = ""
        extraction_source = "unknown"

        if sidecar:
            raw_plaintiffs = sidecar.get("plaintiff") or sidecar.get("plaintiffs") or []
            raw_defendants = sidecar.get("defendant") or sidecar.get("defendants") or []
            if isinstance(raw_plaintiffs, str):
                plaintiffs = [p.strip() for p in raw_plaintiffs.split(",") if p.strip()]
            elif isinstance(raw_plaintiffs, list):
                plaintiffs = [str(p).strip() for p in raw_plaintiffs if str(p).strip()]
            if isinstance(raw_defendants, str):
                defendants = [d.strip() for d in raw_defendants.split(",") if d.strip()]
            elif isinstance(raw_defendants, list):
                defendants = [str(d).strip() for d in raw_defendants if str(d).strip()]
            case_numbers = sidecar.get("case_numbers") or []
            if isinstance(case_numbers, str):
                case_numbers = [case_numbers]
            case_numbers = [str(c).strip() for c in case_numbers if str(c).strip()]
            letter_type = str(sidecar.get("letter_type") or "").strip()
            extraction_source = "meta_json"

        if not plaintiffs and not defendants:
            parties = self._parse_parties_from_filename(filename)
            if parties:
                defendants = parties
                extraction_source = "filename"

        if not case_numbers:
            case_numbers = self._parse_case_numbers(filename)

        combined_parties = plaintiffs + defendants
        opposing_parties = self._filter_opposing_parties(combined_parties or defendants or plaintiffs)

        return DocumentMeta(
            source_path=pdf_path,
            file_name=filename,
            plaintiffs=plaintiffs,
            defendants=defendants,
            opposing_parties=opposing_parties,
            case_numbers=case_numbers,
            letter_type=letter_type,
            raw_text_excerpt=None,
            extraction_source=extraction_source,
        )

    def _ai_tiebreak(
        self,
        doc: DocumentMeta,
        candidates: list[MatchCandidate],
    ) -> tuple[int | None, float, str]:
        shortlist = candidates[: self.config.ai_max_candidates]
        doc_summary = {
            "opposing_parties": doc.opposing_parties,
            "case_numbers": doc.case_numbers,
            "letter_type": doc.letter_type,
        }
        candidate_summary = [
            {
                "folder_name": cand.folder.folder_name,
                "score": cand.score,
                "reason": "; ".join(cand.reasons[:2]),
            }
            for cand in shortlist
        ]

        providers = [self.ai_provider]
        if self.ai_provider == "auto":
            providers = ["ollama", "openai"]

        for provider in providers:
            result = choose_best_candidate(
                doc_summary=doc_summary,
                candidates=candidate_summary,
                provider=provider,
            )
            if result is None:
                continue
            best_index = result.get("best_index")
            confidence = float(result.get("confidence") or 0.0)
            reason = str(result.get("reason") or "").strip()
            if best_index is None:
                return None, confidence, reason or "AI declined to choose"
            if not isinstance(best_index, int):
                return None, confidence, "AI returned invalid index"
            if best_index < 0 or best_index >= len(shortlist):
                return None, confidence, "AI index out of range"
            return best_index, confidence, reason or "AI tie-breaker"

        return None, 0.0, "AI unavailable"

    def _decision_for_score(self, score_summary: ScoreSummary) -> tuple[bool, bool]:
        best = score_summary.best_score
        second = score_summary.second_score
        auto = best >= self.config.auto_threshold and (best - second) >= self.config.gap_threshold
        tie = abs(best - second) <= self.config.tie_epsilon
        return auto, tie

    def plan_distribution(
        self,
        pdf_files: list[str],
        *,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> list[DistributionPlanItem]:
        folder_index = self.build_index()
        plan: list[DistributionPlanItem] = []
        total = len(pdf_files)
        processed = 0
        for filename in pdf_files:
            pdf_path = os.path.join(self.input_folder, filename)
            doc = self.build_document_meta(pdf_path, filename)
            score_summary = score_document(
                doc,
                folder_index,
                self.config.normalized_stopwords(),
                self.config.top_k,
            )

            candidates = score_summary.candidates
            best = candidates[0] if candidates else None
            if best:
                self.log(
                    f"Scored: {len(candidates)} candidates for {filename} "
                    f"(best={best.folder.folder_name} score={best.score:.1f})"
                )
            else:
                self.log(f"Scored: 0 candidates for {filename}")

            decision = "UNMATCHED"
            confidence = 0.0
            reason = "No candidates"
            chosen_folder: str | None = None
            dest_path: str | None = None

            if candidates:
                auto, tie = self._decision_for_score(score_summary)
                if auto and not tie and best:
                    stopwords = self.config.normalized_stopwords()
                    doc_surnames = extract_surnames_from_parties(doc.opposing_parties, stopwords)
                    doc_tokens = set(tokenize(" ".join(doc.opposing_parties), stopwords))
                    token_overlap = doc_tokens & best.folder.tokens
                    surname_overlap = doc_surnames & best.folder.surnames
                    non_surname_overlap = token_overlap - surname_overlap
                    extreme_score = 95.0
                    if not non_surname_overlap and best.score < extreme_score:
                        auto = False
                        self.log(
                            "Auto-accept blocked: passes threshold but lacks given-name overlap"
                        )
                if auto and not tie:
                    decision = "AUTO"
                    confidence = min(1.0, best.score / 100.0)
                    reason = "High confidence deterministic match"
                    chosen_folder = best.folder.folder_path
                else:
                    best_index, ai_confidence, ai_reason = self._ai_tiebreak(doc, candidates)
                    if best_index is not None and ai_confidence >= self.config.ai_threshold:
                        decision = "AI"
                        confidence = ai_confidence
                        reason = ai_reason or "AI tie-breaker"
                        chosen_folder = candidates[best_index].folder.folder_path
                        self.log(
                            f"AI picked folder #{best_index + 1} confidence={ai_confidence:.2f}"
                        )
                    else:
                        decision = "ASK"
                        confidence = ai_confidence
                        reason = ai_reason or "Ambiguous match; needs confirmation"

            if chosen_folder:
                try:
                    dest_path, _ = resolve_destination_path(pdf_path, chosen_folder)
                except Exception:
                    dest_path = None

            plan.append(
                DistributionPlanItem(
                    source_pdf=pdf_path,
                    chosen_folder=chosen_folder,
                    candidates=candidates,
                    decision=decision,
                    confidence=confidence,
                    reason=reason,
                    dest_path=dest_path,
                )
            )
            processed += 1
            if progress_cb:
                progress_cb(processed, total, f"Planned {processed}/{total}")

        return plan

    def apply_plan(
        self,
        plan: list[DistributionPlanItem],
        *,
        auto_only: bool,
        audit_log_path: str,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> None:
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        total = len(plan)
        processed = 0
        for item in plan:
            decision = item.decision
            if auto_only and decision not in ("AUTO", "AI"):
                self._append_audit_log(audit_log_path, item, "SKIPPED")
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue

            target_folder = item.chosen_folder
            if not target_folder and self.config.unmatched_policy == "unmatched_folder":
                target_folder = os.path.join(self.case_root, "UNMATCHED")

            if not target_folder:
                self.log(f"UNMATCHED -> {os.path.basename(item.source_pdf)}")
                self._append_audit_log(audit_log_path, item, "SKIPPED")
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue

            try:
                dest_path, result = safe_copy(item.source_pdf, target_folder)
                item.dest_path = dest_path
                self.log(f"{result} -> {os.path.basename(target_folder)}")
                self._append_audit_log(audit_log_path, item, result)
            except Exception as exc:
                self.log(f"FAILED -> {os.path.basename(target_folder)} ({exc})")
                self._append_audit_log(audit_log_path, item, "FAILED")
            processed += 1
            if progress_cb:
                progress_cb(processed, total, f"Applied {processed}/{total}")

    def _append_audit_log(self, path: str, item: DistributionPlanItem, result: str) -> None:
        top_candidates = ", ".join(
            f"{cand.folder.folder_name}:{cand.score:.1f}" for cand in item.candidates[:3]
        )
        entry = (
            f"{datetime.now().isoformat()}\t"
            f"src={item.source_pdf}\t"
            f"chosen={item.chosen_folder or ''}\t"
            f"decision={item.decision}\t"
            f"confidence={item.confidence:.2f}\t"
            f"top3={top_candidates}\t"
            f"result={result}"
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(entry + "\n")
