from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .ai_tiebreaker import choose_best_candidate
from .indexer import build_folder_index
from .models import DocumentMeta, DistributionPlanItem, MatchCandidate
from .scorer import (
    COMMON_FIRST_NAMES,
    DEFAULT_STOPWORDS,
    ScoreSummary,
    ensure_document_cache,
    similarity_ratio_normalized,
    score_document,
)
from .safety import resolve_destination_path, safe_copy, safe_move


@dataclass
class DistributionConfig:
    auto_threshold: float = 70.0
    gap_threshold: float = 15.0
    ai_threshold: float = 0.7
    enable_ai_tiebreaker: bool = False
    ai_provider: str = "openai"
    top_k: int = 15
    stage2_k: int = 80
    candidate_pool_limit: int = 200
    fast_mode: bool = True
    tie_epsilon: float = 5.0
    ai_max_candidates: int = 15
    ollama_base_url: str | None = None
    ollama_model: str = "qwen2.5:7b"
    stopwords: list[str] = None
    unassigned_action: str = "leave"
    unassigned_include_ask: bool = False
    dest_exists_policy: str = "skip"
    apply_during_planning: bool = False
    max_workers: int | None = None
    create_unresolved_folders: bool = False
    create_unresolved_scope: str = "unmatched_only"
    create_unresolved_include_case_number: bool = False
    create_unresolved_dry_run: bool = False

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
        self.ai_provider = ai_provider or config.ai_provider
        self.logger = logger

    def log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def build_index(self):
        return build_folder_index(self.case_root, self.config.normalized_stopwords())

    def _is_multi_defendant_folder(self, folder: MatchCandidate | None) -> bool:
        if not folder:
            return False
        folder_meta = folder.folder
        if "_" not in folder_meta.folder_name:
            return False
        if len(folder_meta.surnames) < 2:
            return False
        return not any(token in COMMON_FIRST_NAMES for token in folder_meta.tokens)

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
    ) -> tuple[str | None, float, str]:
        shortlist = candidates[: self.config.ai_max_candidates]
        if not shortlist:
            return None, 0.0, "No candidates for AI"
        doc_summary = {
            "opposing_parties": doc.opposing_parties,
            "case_numbers": doc.case_numbers,
            "letter_type": doc.letter_type,
            "people_tokens": sorted(doc.tokens),
            "people_pairs": [" ".join(pair) for pair in sorted(doc.person_pairs)],
        }
        candidate_summary = [
            {
                "candidate_id": f"cand_{idx + 1}",
                "folder_name": cand.folder.folder_name,
                "match_name": cand.folder.match_name,
                "people_tokens": sorted(cand.folder.tokens),
                "people_pairs": [" ".join(pair) for pair in sorted(cand.folder.person_pairs)],
                "case_numbers": sorted(cand.folder.case_numbers),
                "score": cand.score,
            }
            for idx, cand in enumerate(shortlist)
        ]

        providers = [self.ai_provider]
        if self.ai_provider == "auto":
            providers = ["ollama", "openai"]

        for provider in providers:
            result = choose_best_candidate(
                doc_summary=doc_summary,
                candidates=candidate_summary,
                provider=provider,
                ollama_base_url=self.config.ollama_base_url,
                ollama_model=self.config.ollama_model,
            )
            if result is None:
                logging.getLogger(__name__).debug(
                    "AI tiebreaker failed to return JSON (provider=%s)", provider
                )
                continue
            chosen_id = str(result.get("chosen_id") or "").strip()
            chosen_folder = str(result.get("chosen_folder") or "").strip()
            confidence = float(result.get("confidence") or 0.0)
            reason = str(result.get("reason") or "").strip()
            if not chosen_id and not chosen_folder:
                return None, confidence, reason or "AI declined to choose"
            match = None
            if chosen_id:
                match = next(
                    (
                        cand
                        for idx, cand in enumerate(shortlist)
                        if f"cand_{idx + 1}" == chosen_id
                    ),
                    None,
                )
            if match is None and chosen_folder:
                match = next(
                    (
                        cand
                        for cand in shortlist
                        if cand.folder.folder_name == chosen_folder
                    ),
                    None,
                )
            if not match:
                logger = logging.getLogger(__name__)
                logger.debug(
                    "AI tiebreaker returned unknown folder: id=%s folder=%s (provider=%s)",
                    chosen_id,
                    chosen_folder,
                    provider,
                )
                return None, confidence, "AI returned unknown folder"
            return match.folder.folder_path, confidence, reason or "AI tie-breaker"

        self.log("[AI][WARN] Tiebreaker unavailable; check AI configuration or logs.")
        return None, 0.0, "AI unavailable"

    def _is_ambiguous(self, score_summary: ScoreSummary, doc: DocumentMeta) -> tuple[bool, str]:
        if not score_summary.candidates:
            return False, "no candidates"
        if len(score_summary.candidates) < 3:
            return False, "not enough candidates"
        best = score_summary.best_score
        second = score_summary.second_score
        min_score = max(25.0, self.config.auto_threshold * 0.5)
        if best < min_score:
            return False, "score below floor"
        gap = best - second
        if gap < self.config.gap_threshold or abs(gap) <= self.config.tie_epsilon:
            return True, "low gap"
        top = score_summary.candidates[:3]
        if len(top) >= 2 and doc.person_pairs:
            overlaps = [
                len(doc.person_pairs & candidate.folder.person_pairs)
                for candidate in top
            ]
            overlaps_sorted = sorted(overlaps, reverse=True)
            if overlaps_sorted[0] > 0 and overlaps_sorted[0] - overlaps_sorted[1] <= 1:
                return True, "overlap tie"
        if top and doc.surnames:
            best_candidate = top[0]
            doc_given = doc.tokens - doc.surnames
            best_given = best_candidate.folder.tokens - best_candidate.folder.surnames
            best_surname_overlap = doc.surnames & best_candidate.folder.surnames
            best_given_overlap = doc_given & best_given
            if best_surname_overlap and not best_given_overlap:
                for candidate in top[1:]:
                    candidate_given = candidate.folder.tokens - candidate.folder.surnames
                    candidate_surname_overlap = doc.surnames & candidate.folder.surnames
                    candidate_given_overlap = doc_given & candidate_given
                    if candidate_surname_overlap and candidate_given_overlap:
                        return True, "given-name conflict on shared surname"
        if top and top[0].folder.match_name != top[0].folder.folder_name:
            if gap <= (self.config.gap_threshold + self.config.tie_epsilon):
                return True, "suffix noise on folder name"
        return False, "no ambiguity trigger"

    def _sanitize_folder_component(self, text: str) -> str:
        cleaned = (text or "").replace("/", "_")
        cleaned = re.sub(r'[<>:"\\|?*]', "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.replace(" ", "_")
        return cleaned.strip("_")

    def _format_party_name(self, party: str) -> str:
        tokens = [tok for tok in re.split(r"\s+", party.strip()) if tok]
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0].upper()
        surname = tokens[-1].upper()
        given = tokens[0].upper()
        return f"{surname}_{given}"

    def _build_created_folder_name(self, doc: DocumentMeta) -> str:
        parties = doc.opposing_parties or []
        formatted = []
        for party in parties:
            formatted_name = self._format_party_name(party)
            safe_name = self._sanitize_folder_component(formatted_name)
            if safe_name:
                formatted.append(safe_name)
        if formatted:
            base = "_".join(formatted)
        else:
            base = ""
        if self.config.create_unresolved_include_case_number and doc.case_numbers:
            case_tag = self._sanitize_folder_component(str(doc.case_numbers[0]))
            if case_tag:
                base = f"{base}_{case_tag}" if base else case_tag
        base = base.strip("_")
        if not base or base.upper() == "UNASSIGNED":
            digest = hashlib.sha1(doc.file_name.encode("utf-8")).hexdigest()[:8]
            base = f"UNRESOLVED_{digest}"
        return base

    def _find_existing_case_folder(self, folder_name: str) -> str | None:
        if not folder_name:
            return None
        target_lower = folder_name.lower()
        try:
            for name in os.listdir(self.case_root):
                if name.lower() == target_lower:
                    candidate_path = os.path.join(self.case_root, name)
                    if os.path.isdir(candidate_path):
                        return candidate_path
        except FileNotFoundError:
            return None
        return None

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
        pause_event: threading.Event | None = None,
        audit_log_path: str | None = None,
    ) -> list[DistributionPlanItem]:
        folder_index = self.build_index()
        plan: list[DistributionPlanItem | None] = [None] * len(pdf_files)
        total = len(pdf_files)
        processed = 0
        if audit_log_path:
            os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        stopwords = self.config.normalized_stopwords()
        def build_plan_item(filename: str) -> DistributionPlanItem:
            if pause_event is not None:
                pause_event.wait()
            pdf_path = os.path.join(self.input_folder, filename)
            doc = self.build_document_meta(pdf_path, filename)
            ensure_document_cache(doc, stopwords)
            score_summary = score_document(
                doc,
                folder_index,
                stopwords,
                self.config.top_k,
                stage2_k=self.config.stage2_k,
                candidate_pool_limit=self.config.candidate_pool_limit,
                fast_mode=self.config.fast_mode,
            )

            candidates = score_summary.candidates
            best = candidates[0] if candidates else None
            auto_block_reason = ""
            if best:
                self.log(
                    f"Scored {len(candidates)} candidates for {filename}: "
                    f"best={best.folder.folder_name} score={best.score:.1f}"
                )
            else:
                self.log(f"Scored 0 candidates for {filename}")
                folder_count = len(folder_index.folders) if hasattr(folder_index, "folders") else len(folder_index)
                if folder_count == 0:
                    self.log("No candidates: folder index contained 0 folders")
            if not doc.tokens and not doc.person_pairs:
                self.log("No candidates: document party parsing returned empty")
            if candidates and score_summary.best_score <= 1.0:
                self.log("Low-score candidates: best score near zero; check parsing")

            decision = "UNMATCHED"
            confidence = 0.0
            reason = "No candidates"
            chosen_folder: str | None = None
            dest_path: str | None = None
            result: str | None = None

            if candidates:
                auto, tie = self._decision_for_score(score_summary)
                force_ask = False
                auto_block_reason = ""
                if auto and not tie and best:
                    token_overlap = doc.tokens & best.folder.tokens
                    surname_overlap = doc.surnames & best.folder.surnames
                    non_surname_overlap = token_overlap - surname_overlap
                    pair_overlap = doc.person_pairs & best.folder.person_pairs
                    overlap_count = len(token_overlap)
                    full_name_match = bool(pair_overlap) or (
                        overlap_count >= 2 and bool(non_surname_overlap)
                    )
                    surname_only = bool(token_overlap) and (
                        overlap_count == 1 or token_overlap <= surname_overlap
                    )
                    score_gap = score_summary.best_score - score_summary.second_score
                    multi_defendant_folder = self._is_multi_defendant_folder(best)
                    allow_surname_exception = (
                        surname_only
                        and not multi_defendant_folder
                        and best.score >= 90.0
                        and score_gap >= 20.0
                    )

                    if full_name_match:
                        self.log(
                            "Auto-accepted: full-name overlap for "
                            f"{filename} -> {best.folder.folder_name} "
                            f"(score={best.score:.1f}, overlap={overlap_count}, gap={score_gap:.1f})"
                        )
                    elif allow_surname_exception:
                        self.log(
                            "Auto-accepted: surname-only exception for "
                            f"{filename} -> {best.folder.folder_name} "
                            f"(score={best.score:.1f}, overlap={overlap_count}, gap={score_gap:.1f})"
                        )
                    else:
                        auto = False
                        force_ask = True
                        auto_block_reason = "surname-only overlap"
                        if surname_only:
                            reason_detail = (
                                f"overlap={overlap_count}, non_surname={len(non_surname_overlap)}, "
                                f"multi_defendant={multi_defendant_folder}"
                            )
                            self.log(
                                "Auto-accept blocked: surname-only overlap for "
                                f"{filename} -> {best.folder.folder_name} "
                                f"(score={best.score:.1f}, overlap={overlap_count}, reason={reason_detail})"
                            )

                if auto and not tie:
                    decision = "AUTO"
                    confidence = min(1.0, best.score / 100.0)
                    reason = "High confidence deterministic match"
                    chosen_folder = best.folder.folder_path
                else:
                    if force_ask:
                        decision = "ASK"
                        confidence = 0.0
                        reason = "Ambiguous match; needs confirmation"
                    else:
                        if not auto and auto_block_reason:
                            self.log(
                                f"Auto-accept blocked: {auto_block_reason} "
                                f"(score={score_summary.best_score:.1f}, gap={score_summary.best_score - score_summary.second_score:.1f})"
                            )
                        if score_summary.best_score < self.config.auto_threshold:
                            auto_block_reason = "low score"
                        elif score_summary.best_score - score_summary.second_score < self.config.gap_threshold:
                            auto_block_reason = "low gap"
                        decision = "ASK"
                        confidence = 0.0
                        reason = "Ambiguous match; needs confirmation"
                    ai_chosen, ai_confidence, ai_reason = (None, 0.0, "")
                    attempted_ai = False
                    ambiguous, ambiguity_reason = self._is_ambiguous(score_summary, doc)
                    if self.config.enable_ai_tiebreaker and ambiguous:
                        attempted_ai = True
                        self.log(
                            "AI_TIEBREAK triggered: "
                            f"src={filename} best={best.folder.folder_name if best else 'n/a'} "
                            f"gap={score_summary.best_score - score_summary.second_score:.1f} "
                            f"candidates={min(len(candidates), self.config.ai_max_candidates)} "
                            f"reason={ambiguity_reason}"
                        )
                        ai_chosen, ai_confidence, ai_reason = self._ai_tiebreak(
                            doc, candidates
                        )
                    elif self.config.enable_ai_tiebreaker and not ambiguous:
                        self.log(
                            f"AI_TIEBREAK skipped: src={filename} reason={ambiguity_reason}"
                        )
                    if ai_chosen and ai_confidence >= self.config.ai_threshold:
                        decision = "AI"
                        confidence = ai_confidence
                        reason = ai_reason or "AI tie-breaker"
                        chosen_folder = ai_chosen
                        self.log(
                            "AI_TIEBREAK chosen: "
                            f"folder={os.path.basename(ai_chosen)} conf={ai_confidence:.2f}"
                        )
                    elif ai_chosen:
                        decision = "ASK"
                        confidence = ai_confidence
                        reason = ai_reason or "AI suggestion (low confidence)"
                        chosen_folder = ai_chosen
                        self.log(
                            "AI_TIEBREAK chosen: "
                            f"folder={os.path.basename(ai_chosen)} conf={ai_confidence:.2f} (below threshold)"
                        )
                    else:
                        if attempted_ai and ai_reason:
                            logging.getLogger(__name__).debug(
                                "AI tiebreaker skipped: %s (file=%s)", ai_reason, filename
                            )
                            if ai_reason == "AI unavailable":
                                self.log(f"AI unavailable for {filename}")
                            else:
                                self.log(
                                    f"AI_TIEBREAK skipped: src={filename} reason={ai_reason}"
                                )

            if decision in ("ASK", "UNMATCHED"):
                doc_token_list = sorted(doc.tokens)
                doc_pair_list = sorted(doc.person_pairs)
                pool_size = score_summary.candidate_pool_size
                if decision == "ASK" and (confidence <= 0.0 or score_summary.best_score <= 0.0):
                    any_token_overlap = any(
                        doc.tokens & candidate.folder.tokens for candidate in candidates
                    )
                    any_pair_overlap = any(
                        doc.person_pairs & candidate.folder.person_pairs for candidate in candidates
                    )
                    self.log(
                        "ASK=0 diagnostics: "
                        f"file={filename} tokens={doc_token_list} pairs={len(doc_pair_list)} "
                        f"pool={pool_size} any_token_overlap={any_token_overlap} "
                        f"any_pair_overlap={any_pair_overlap}"
                    )
                if candidates:
                    best_candidate = candidates[0]
                    top_pair_overlap = doc.person_pairs & best_candidate.folder.person_pairs
                    top_token_overlap = doc.tokens & best_candidate.folder.tokens
                    top_similarity = similarity_ratio_normalized(
                        doc.normalized_opposing, best_candidate.folder.normalized_name
                    )
                    logger = logging.getLogger(__name__)
                    logger.debug(
                        "Match diagnostics: decision=%s file=%s tokens=%s pairs=%d pool=%d best=%s "
                        "score=%.1f pair_matches=%d token_overlap=%d similarity=%.2f auto_block=%s",
                        decision,
                        filename,
                        doc_token_list,
                        len(doc_pair_list),
                        pool_size,
                        best_candidate.folder.folder_name,
                        best_candidate.score,
                        len(top_pair_overlap),
                        len(top_token_overlap),
                        top_similarity,
                        auto_block_reason or "n/a",
                    )
                    self.log(
                        f"Match diagnostics: decision={decision} file={filename} "
                        f"best={best_candidate.folder.folder_name} score={best_candidate.score:.1f} "
                        f"auto_block={auto_block_reason or 'n/a'}"
                    )
                else:
                    logger = logging.getLogger(__name__)
                    logger.debug(
                        "Match diagnostics: decision=%s file=%s tokens=%s pairs=%d pool=%d best=None "
                        "score=0.0 auto_block=%s",
                        decision,
                        filename,
                        doc_token_list,
                        len(doc_pair_list),
                        pool_size,
                        auto_block_reason or "n/a",
                    )
                    self.log(
                        f"Match diagnostics: decision={decision} file={filename} "
                        f"best=None auto_block={auto_block_reason or 'n/a'}"
                    )
                logger = logging.getLogger(__name__)
                if candidates:
                    top_three = candidates[:3]
                    for idx, candidate in enumerate(top_three, start=1):
                        pair_overlap = doc.person_pairs & candidate.folder.person_pairs
                        token_overlap = doc.tokens & candidate.folder.tokens
                        logger.debug(
                            "Debug match #%d file=%s folder=%s tokens=%s pairs=%s pair_overlap=%s token_overlap=%s",
                            idx,
                            filename,
                            candidate.folder.folder_name,
                            sorted(candidate.folder.tokens),
                            sorted(candidate.folder.person_pairs),
                            sorted(pair_overlap),
                            sorted(token_overlap),
                        )

            if chosen_folder:
                try:
                    dest_path, _ = resolve_destination_path(
                        pdf_path,
                        chosen_folder,
                        exists_policy=self.config.dest_exists_policy,
                    )
                except Exception:
                    dest_path = None
            if (
                self.config.apply_during_planning
                and decision == "AUTO"
                and chosen_folder
                and audit_log_path
            ):
                result, applied_dest = self._apply_during_planning(
                    pdf_path,
                    chosen_folder,
                    audit_log_path,
                    dest_path,
                    decision,
                    confidence,
                    reason,
                    candidates,
                )
                if applied_dest:
                    dest_path = applied_dest

            return DistributionPlanItem(
                source_pdf=pdf_path,
                chosen_folder=chosen_folder,
                candidates=candidates,
                decision=decision,
                confidence=confidence,
                reason=reason,
                dest_path=dest_path,
                result=result,
            )

        max_workers = self.config.max_workers
        if max_workers is None:
            cpu_count = os.cpu_count() or 2
            max_workers = min(4, cpu_count)

        if max_workers < 2 or total < 2 or self.config.apply_during_planning:
            for idx, filename in enumerate(pdf_files):
                plan[idx] = build_plan_item(filename)
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Planned {processed}/{total}")
            return [item for item in plan if item]

        progress_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(build_plan_item, filename): idx
                for idx, filename in enumerate(pdf_files)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                plan[idx] = future.result()
                with progress_lock:
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total, f"Planned {processed}/{total}")

        return [item for item in plan if item]

    def _apply_during_planning(
        self,
        source_pdf: str,
        chosen_folder: str,
        audit_log_path: str,
        dest_path: str | None,
        decision: str,
        confidence: float,
        reason: str,
        candidates: list[MatchCandidate],
    ) -> tuple[str, str | None]:
        case_root_abs = os.path.abspath(self.case_root)
        target_abs = os.path.abspath(chosen_folder)
        if os.path.commonpath([case_root_abs, target_abs]) != case_root_abs:
            result = "SKIPPED (safety_guard)"
            self.log(f"{result} -> {os.path.basename(source_pdf)}")
            self._append_audit_log(
                audit_log_path,
                DistributionPlanItem(
                    source_pdf=source_pdf,
                    chosen_folder=chosen_folder,
                    candidates=candidates,
                    decision=decision,
                    confidence=confidence,
                    reason=reason,
                    dest_path=dest_path,
                ),
                result,
            )
            return result, dest_path
        try:
            cleanup_target: str | None = None
            dest_path, copy_result = safe_copy(
                source_pdf,
                chosen_folder,
                exists_policy=self.config.dest_exists_policy,
            )
            cleanup_target = dest_path
            if copy_result == "SKIPPED":
                result = "SKIPPED (dest_exists)"
            else:
                result = "COPIED (planning)"
            self.log(f"{result} -> {os.path.basename(chosen_folder)}")
            self._append_audit_log(
                audit_log_path,
                DistributionPlanItem(
                    source_pdf=source_pdf,
                    chosen_folder=chosen_folder,
                    candidates=candidates,
                    decision=decision,
                    confidence=confidence,
                    reason=reason,
                    dest_path=dest_path,
                ),
                result,
            )
            return result, dest_path
        except Exception as exc:
            result = f"FAILED ({exc})"
            if cleanup_target and os.path.exists(cleanup_target):
                try:
                    os.remove(cleanup_target)
                    self.log(
                        f"[WARN] Removed incomplete file -> {os.path.basename(cleanup_target)}"
                    )
                except Exception as cleanup_exc:
                    logging.getLogger(__name__).exception(
                        "Failed to remove incomplete file %s: %s",
                        cleanup_target,
                        cleanup_exc,
                    )
            self.log(
                f"{result} -> {os.path.basename(source_pdf)} -> {os.path.basename(chosen_folder)}"
            )
            self._append_audit_log(
                audit_log_path,
                DistributionPlanItem(
                    source_pdf=source_pdf,
                    chosen_folder=chosen_folder,
                    candidates=candidates,
                    decision=decision,
                    confidence=confidence,
                    reason=reason,
                    dest_path=dest_path,
                ),
                "FAILED",
            )
            return "FAILED", dest_path

    def apply_plan(
        self,
        plan: list[DistributionPlanItem],
        *,
        auto_only: bool,
        audit_log_path: str,
        progress_cb: Callable[[int, int, str], None] | None = None,
        pause_event: threading.Event | None = None,
    ) -> None:
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        total = len(plan)
        processed = 0
        copied_count = 0
        skipped_count = 0
        dest_exists_count = 0
        unmatched_count = 0
        ask_unresolved_count = 0
        for item in plan:
            if pause_event is not None:
                pause_event.wait()
            if item.result and (
                item.result.startswith("COPIED")
                or item.result.startswith("SKIPPED (dest_exists)")
            ):
                self._append_audit_log(audit_log_path, item, "SKIPPED (already_applied)")
                self.log(
                    f"SKIPPED (already_applied) -> {os.path.basename(item.source_pdf)}"
                )
                skipped_count += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue
            decision = item.decision
            unassigned_action = self.config.unassigned_action
            send_ask = self.config.unassigned_include_ask
            should_unassign = (
                decision == "UNMATCHED"
                or (decision == "ASK" and send_ask and not item.chosen_folder)
            )
            should_create_unresolved = (
                self.config.create_unresolved_folders
                and (
                    decision == "UNMATCHED"
                    or (
                        decision == "ASK"
                        and not item.chosen_folder
                        and self.config.create_unresolved_scope == "unmatched_and_ask"
                    )
                )
            )
            allow_unassigned = should_unassign and unassigned_action in ("copy", "move")
            if auto_only and decision not in ("AUTO", "AI") and not (
                decision == "ASK" and item.chosen_folder
            ) and not allow_unassigned and not should_create_unresolved:
                self._append_audit_log(audit_log_path, item, "SKIPPED (auto_apply_only)")
                self.log(f"SKIPPED (auto_apply_only) -> {os.path.basename(item.source_pdf)}")
                skipped_count += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue

            target_folder = item.chosen_folder
            if (
                not target_folder
                and should_unassign
                and unassigned_action in ("copy", "move")
                and not should_create_unresolved
            ):
                target_folder = os.path.join(self.case_root, "UNASSIGNED")

            if (
                decision == "ASK"
                and not item.chosen_folder
                and not should_create_unresolved
                and not (should_unassign and unassigned_action in ("copy", "move"))
            ):
                self._append_audit_log(audit_log_path, item, "SKIPPED (unresolved_ask)")
                self.log(f"SKIPPED (unresolved_ask) -> {os.path.basename(item.source_pdf)}")
                skipped_count += 1
                ask_unresolved_count += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue

            if (
                decision == "UNMATCHED"
                and not should_create_unresolved
                and not (should_unassign and unassigned_action in ("copy", "move"))
            ):
                self._append_audit_log(audit_log_path, item, "SKIPPED (unmatched)")
                self.log(f"SKIPPED (unmatched) -> {os.path.basename(item.source_pdf)}")
                skipped_count += 1
                unmatched_count += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue

            if should_create_unresolved and not item.chosen_folder:
                doc = self.build_document_meta(
                    item.source_pdf, os.path.basename(item.source_pdf)
                )
                folder_name = self._build_created_folder_name(doc)
                if not folder_name:
                    self.log(
                        f"SKIPPED (invalid_folder_name) -> {os.path.basename(item.source_pdf)}"
                    )
                    self._append_audit_log(
                        audit_log_path, item, "SKIPPED (invalid_folder_name)"
                    )
                    skipped_count += 1
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total, f"Applied {processed}/{total}")
                    continue
                existing_path = self._find_existing_case_folder(folder_name)
                target_folder = existing_path or os.path.join(self.case_root, folder_name)
                case_root_abs = os.path.abspath(self.case_root)
                target_abs = os.path.abspath(target_folder)
                if os.path.commonpath([case_root_abs, target_abs]) != case_root_abs:
                    self.log(
                        f"SKIPPED (safety_guard) -> {os.path.basename(item.source_pdf)}"
                    )
                    self._append_audit_log(audit_log_path, item, "SKIPPED (safety_guard)")
                    skipped_count += 1
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total, f"Applied {processed}/{total}")
                    continue
                if self.config.create_unresolved_dry_run:
                    self.log(f"DRY_RUN create folder -> {target_folder}")
                    self._append_audit_log(
                        audit_log_path, item, f"DRY_RUN (create_folder -> {target_folder})"
                    )
                    skipped_count += 1
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total, f"Applied {processed}/{total}")
                    continue
                if not existing_path:
                    os.makedirs(target_folder, exist_ok=True)
                    self.log(f"CREATED_FOLDER -> {target_folder}")
                    self._append_audit_log(
                        audit_log_path, item, f"CREATED_FOLDER -> {target_folder}"
                    )
                else:
                    self.log(f"CREATED_FOLDER skipped (exists) -> {target_folder}")
                item.chosen_folder = target_folder

            if not target_folder:
                self.log(f"SKIPPED (invalid_target) -> {os.path.basename(item.source_pdf)}")
                self._append_audit_log(audit_log_path, item, "SKIPPED (invalid_target)")
                skipped_count += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue

            case_root_abs = os.path.abspath(self.case_root)
            target_abs = os.path.abspath(target_folder)
            if os.path.commonpath([case_root_abs, target_abs]) != case_root_abs:
                self.log(f"SKIPPED (safety_guard) -> {os.path.basename(item.source_pdf)}")
                self._append_audit_log(audit_log_path, item, "SKIPPED (safety_guard)")
                skipped_count += 1
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, f"Applied {processed}/{total}")
                continue

            dest_path = None
            try:
                if should_unassign and unassigned_action == "move":
                    dest_path, result = safe_move(
                        item.source_pdf,
                        target_folder,
                        exists_policy=self.config.dest_exists_policy,
                    )
                else:
                    dest_path, result = safe_copy(
                        item.source_pdf,
                        target_folder,
                        exists_policy=self.config.dest_exists_policy,
                    )
                item.dest_path = dest_path
                if result == "SKIPPED":
                    result = "SKIPPED (dest_exists)"
                    skipped_count += 1
                    dest_exists_count += 1
                else:
                    copied_count += 1
                result_label = result
                if should_unassign and os.path.basename(target_folder) == "UNASSIGNED":
                    result_label = f"{result} (unassigned)"
                elif should_create_unresolved and target_folder:
                    result_label = f"{result} (created_folder)"
                self.log(f"{result_label} -> {os.path.basename(target_folder)}")
                self._append_audit_log(audit_log_path, item, result_label)
            except Exception as exc:
                if dest_path and os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                        self.log(
                            f"[WARN] Removed incomplete file -> {os.path.basename(dest_path)}"
                        )
                    except Exception as cleanup_exc:
                        logging.getLogger(__name__).exception(
                            "Failed to remove incomplete file %s: %s",
                            dest_path,
                            cleanup_exc,
                        )
                self.log(
                    f"FAILED -> {os.path.basename(item.source_pdf)} -> {os.path.basename(target_folder)} ({exc})"
                )
                self._append_audit_log(audit_log_path, item, "FAILED")
            processed += 1
            if progress_cb:
                progress_cb(processed, total, f"Applied {processed}/{total}")
        self.log(
            "Apply summary: "
            f"COPIED={copied_count} SKIPPED={skipped_count} "
            f"DEST_EXISTS={dest_exists_count} UNMATCHED={unmatched_count} "
            f"ASK_UNRESOLVED={ask_unresolved_count}"
        )

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
