from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from .models import DocumentMeta, FolderMeta, MatchCandidate


DEFAULT_STOPWORDS = [
    "sp",
    "spółka",
    "spolka",
    "sa",
    "s.a",
    "ag",
    "zoo",
    "z.o.o",
    "oddzial",
    "w",
    "polsce",
    "bank",
    "international",
]

COMMON_FIRST_NAMES_RAW = {
    "Jan",
    "Anna",
    "Piotr",
    "Katarzyna",
    "Pawel",
    "Paweł",
    "Agnieszka",
    "Tomasz",
    "Magdalena",
    "Marek",
    "Joanna",
    "Krzysztof",
    "Barbara",
    "Andrzej",
    "Monika",
    "Michal",
    "Michał",
    "Ewa",
    "Marcin",
    "Zofia",
    "Adam",
    "Aleksandra",
    "Karol",
    "Natalia",
    "Jakub",
    "Iwona",
    "Rafal",
    "Rafał",
    "Julia",
    "Dariusz",
    "Oliwia",
    "Patryk",
    "Weronika",
    "Damian",
    "Elzbieta",
    "Elżbieta",
    "Grzegorz",
    "Marta",
    "Mateusz",
    "Dorota",
    "Sebastian",
    "Kinga",
    "Lukasz",
    "Łukasz",
    "Beata",
}


logger = logging.getLogger(__name__)


@dataclass
class ScoreSummary:
    candidates: list[MatchCandidate]
    best_score: float
    second_score: float


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    cleaned = strip_diacritics(text or "")
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[\-_,.;:()\[\]{}<>!?/\\]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


COMMON_FIRST_NAMES = {normalize_text(name) for name in COMMON_FIRST_NAMES_RAW if name}


def normalize_stopwords(stopwords: Iterable[str]) -> set[str]:
    return {normalize_text(word) for word in stopwords if word}


def tokenize(text: str, stopwords: Iterable[str]) -> list[str]:
    normalized = normalize_text(text)
    tokens = [tok for tok in normalized.split() if tok]
    blocked = normalize_stopwords(stopwords)
    return [tok for tok in tokens if tok not in blocked]


def split_party_segments(text: str) -> list[str]:
    raw = text or ""
    raw = raw.replace("&", ",")
    raw = re.sub(r"\s+i\s+", ",", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+oraz\s+", ",", raw, flags=re.IGNORECASE)
    parts = re.split(r"[;,]", raw)
    return [part.strip() for part in parts if part.strip()]


def last_non_stopword(tokens: list[str], stopwords: set[str]) -> str:
    for token in reversed(tokens):
        if token and token not in stopwords:
            return token
    return ""


def choose_surname(tokens: list[str], stopwords: set[str]) -> str:
    filtered = [token for token in tokens if token and token not in stopwords]
    if not filtered:
        return ""
    if len(filtered) == 2:
        first, second = filtered
        first_is_name = first in COMMON_FIRST_NAMES
        second_is_name = second in COMMON_FIRST_NAMES
        if first_is_name and not second_is_name:
            return second
        if second_is_name and not first_is_name:
            return first
        return second
    for token in reversed(filtered):
        if token not in COMMON_FIRST_NAMES:
            return token
    return filtered[-1]


def extract_surnames_from_parties(parties: Iterable[str], stopwords: Iterable[str]) -> set[str]:
    blocked = normalize_stopwords(stopwords)
    surnames: set[str] = set()
    for name in parties:
        segments = split_party_segments(name)
        if not segments:
            segments = [name]
        for segment in segments:
            tokens = [tok for tok in normalize_text(segment).split() if tok]
            surname = choose_surname(tokens, blocked)
            if surname:
                surnames.add(surname)
    return surnames


def extract_surnames_from_folder(folder_name: str, stopwords: Iterable[str]) -> set[str]:
    blocked = normalize_stopwords(stopwords)
    surnames: set[str] = set()
    segments = split_party_segments(folder_name)
    if not segments:
        segments = [folder_name]
    for segment in segments:
        tokens = [tok for tok in normalize_text(segment).split() if tok]
        candidates = [
            token
            for token in tokens
            if token and token not in blocked and token not in COMMON_FIRST_NAMES
        ]
        if candidates:
            surnames.update(candidates)
        else:
            fallback = last_non_stopword(tokens, blocked)
            if fallback:
                surnames.add(fallback)
    return surnames


def similarity_ratio(a: str, b: str) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def score_document_to_folder(
    doc: DocumentMeta,
    folder: FolderMeta,
    stopwords: Iterable[str],
) -> MatchCandidate:
    reasons: list[str] = []
    score = 0.0

    if doc.case_numbers and folder.case_numbers:
        overlap_cases = set(doc.case_numbers) & set(folder.case_numbers)
        if overlap_cases:
            score += 100.0
            reasons.append(f"Case number match: {', '.join(sorted(overlap_cases))}")

    doc_surnames = extract_surnames_from_parties(doc.opposing_parties, stopwords)
    surname_overlap = doc_surnames & folder.surnames
    if surname_overlap:
        score += 30.0 * len(surname_overlap)
        reasons.append(f"Surname overlap: {', '.join(sorted(surname_overlap))}")

    doc_tokens = set(tokenize(" ".join(doc.opposing_parties), stopwords))
    token_overlap = doc_tokens & folder.tokens
    non_surname_overlap = token_overlap - surname_overlap
    if token_overlap:
        score += 4.0 * len(token_overlap)
        reasons.append(f"Token overlap: {', '.join(sorted(token_overlap))}")
    if surname_overlap and non_surname_overlap:
        # Ensure surname+given-name beats surname-only matches without inflating scale.
        score += 20.0
        reasons.append("Bonus: surname and given name overlap")

    ratio = similarity_ratio(" ".join(doc.opposing_parties), folder.folder_name)
    if ratio:
        score += ratio * 20.0
        reasons.append(f"Name similarity: {ratio:.2f}")

    if len(folder.surnames) == 1 and len(doc_surnames) >= 2 and len(surname_overlap) == 1:
        score -= 10.0
        reasons.append("Penalty: folder has single surname while doc has multiple")

    if len(doc_surnames) >= 2 and not token_overlap:
        score -= 5.0
        reasons.append("Penalty: no meaningful token overlap for multi-party doc")

    logger.debug(
        "Score doc=%s folder=%s doc_surnames=%s folder_surnames=%s doc_tokens=%s folder_tokens=%s "
        "token_overlap=%d non_surname_overlap=%d score=%.1f",
        doc.file_name,
        folder.folder_name,
        sorted(doc_surnames),
        sorted(folder.surnames),
        sorted(doc_tokens),
        sorted(folder.tokens),
        len(token_overlap),
        len(non_surname_overlap),
        score,
    )

    return MatchCandidate(folder=folder, score=score, reasons=reasons)


def score_document(
    doc: DocumentMeta,
    folders: Iterable[FolderMeta],
    stopwords: Iterable[str],
    top_k: int,
) -> ScoreSummary:
    candidates = [score_document_to_folder(doc, folder, stopwords) for folder in folders]
    candidates.sort(key=lambda cand: (cand.score, cand.folder.folder_name.lower()), reverse=True)
    top_candidates = candidates[: max(1, top_k)]
    best_score = top_candidates[0].score if top_candidates else 0.0
    second_score = top_candidates[1].score if len(top_candidates) > 1 else 0.0
    return ScoreSummary(candidates=top_candidates, best_score=best_score, second_score=second_score)
