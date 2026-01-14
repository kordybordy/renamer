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


def split_parties(text: str) -> list[str]:
    raw = text or ""
    raw = raw.replace("&", ",")
    raw = re.sub(r"\s+(?:i|oraz)\s+", ",", raw, flags=re.IGNORECASE)
    parts = [part.strip() for part in re.split(r"[;,]", raw) if part.strip()]
    segments: list[str] = []
    for part in parts:
        segments.extend(_split_party_segment(part))
    return segments


def _split_party_segment(segment: str) -> list[str]:
    cleaned = segment.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    tokens = [tok for tok in normalize_text(cleaned).split() if tok]
    if len(tokens) <= 3:
        return [" ".join(tokens)]
    groups: list[list[str]] = []
    idx = 0
    remaining = len(tokens)
    while remaining > 3:
        groups.append(tokens[idx : idx + 2])
        idx += 2
        remaining -= 2
    if remaining:
        groups.append(tokens[idx:])
    if groups and len(groups[-1]) == 1 and len(groups) > 1:
        groups[-2].extend(groups[-1])
        groups = groups[:-1]
    return [" ".join(group) for group in groups if len(group) >= 2]


def extract_tokens_from_parties(parties: Iterable[str], stopwords: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for name in parties:
        segments = split_parties(name)
        if not segments:
            segments = [name]
        for segment in segments:
            tokens.update(tokenize(segment, stopwords))
    return tokens


def extract_tokens(text: str, stopwords: Iterable[str]) -> set[str]:
    return extract_tokens_from_parties([text], stopwords)


def _unique_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _pick_person_pair(tokens: list[str]) -> tuple[str, str] | None:
    tokens = _unique_tokens(tokens)
    if len(tokens) < 2:
        return None
    if len(tokens) == 2:
        return tuple(sorted(tokens))
    first_name_tokens = [token for token in tokens if token in COMMON_FIRST_NAMES]
    if first_name_tokens:
        given = max(first_name_tokens, key=len)
        others = [token for token in tokens if token != given]
        if not others:
            return None
        surname = max(others, key=len)
        return tuple(sorted([given, surname]))
    shortest = min(tokens, key=len)
    longest = max(tokens, key=len)
    if shortest == longest:
        return tuple(sorted(tokens)[:2])
    return tuple(sorted([shortest, longest]))


def extract_person_pairs(text: str, stopwords: Iterable[str]) -> set[tuple[str, str]]:
    return extract_person_pairs_from_parties([text], stopwords)


def extract_person_pairs_from_parties(
    parties: Iterable[str], stopwords: Iterable[str]
) -> set[tuple[str, str]]:
    blocked = normalize_stopwords(stopwords)
    pairs: set[tuple[str, str]] = set()
    for name in parties:
        segments = split_parties(name)
        if not segments:
            segments = [name]
        for segment in segments:
            tokens = [
                tok
                for tok in normalize_text(segment).split()
                if tok and tok not in blocked
            ]
            pair = _pick_person_pair(tokens)
            if pair:
                pairs.add(pair)
    return pairs


def extract_surnames_from_parties(parties: Iterable[str], stopwords: Iterable[str]) -> set[str]:
    blocked = normalize_stopwords(stopwords)
    surnames: set[str] = set()
    for name in parties:
        segments = split_parties(name)
        if not segments:
            segments = [name]
        for segment in segments:
            tokens = [tok for tok in normalize_text(segment).split() if tok]
            filtered = [tok for tok in tokens if tok and tok not in blocked]
            if not filtered:
                continue
            candidates = [tok for tok in filtered if tok not in COMMON_FIRST_NAMES]
            surnames.update(candidates or filtered)
    return surnames


def extract_surnames_from_folder(folder_name: str, stopwords: Iterable[str]) -> set[str]:
    blocked = normalize_stopwords(stopwords)
    surnames: set[str] = set()
    segments = split_parties(folder_name)
    if not segments:
        segments = [folder_name]
    for segment in segments:
        tokens = [tok for tok in normalize_text(segment).split() if tok]
        filtered = [token for token in tokens if token and token not in blocked]
        if not filtered:
            continue
        candidates = [token for token in filtered if token not in COMMON_FIRST_NAMES]
        surnames.update(candidates or filtered)
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

    doc_person_pairs = extract_person_pairs_from_parties(doc.opposing_parties, stopwords)
    pair_overlap = doc_person_pairs & folder.person_pairs
    if pair_overlap:
        score += 80.0 * len(pair_overlap)
        formatted_pairs = [f"{left}+{right}" for left, right in sorted(pair_overlap)]
        reasons.append(f"Full person match: {', '.join(formatted_pairs)}")

    doc_surnames = extract_surnames_from_parties(doc.opposing_parties, stopwords)
    surname_overlap = doc_surnames & folder.surnames
    if surname_overlap:
        score += 15.0 * len(surname_overlap)
        reasons.append(f"Surname overlap: {', '.join(sorted(surname_overlap))}")

    doc_tokens = extract_tokens_from_parties(doc.opposing_parties, stopwords)
    token_overlap = doc_tokens & folder.tokens
    non_surname_overlap = token_overlap - surname_overlap
    if token_overlap:
        score += 4.0 * len(token_overlap)
        reasons.append(f"Token overlap: {', '.join(sorted(token_overlap))}")
    if surname_overlap and non_surname_overlap:
        score += 10.0
        reasons.append("Bonus: surname and given name overlap")

    ratio = similarity_ratio(" ".join(doc.opposing_parties), folder.folder_name)
    if ratio:
        score += ratio * 20.0
        reasons.append(f"Name similarity: {ratio:.2f}")

    mismatch_tokens = _mismatched_surname_tokens(doc_person_pairs, folder.person_pairs)
    if mismatch_tokens:
        score -= 25.0 * len(mismatch_tokens)
        reasons.append(
            f"Penalty: surname with mismatched given name ({', '.join(mismatch_tokens)})"
        )

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


def _mismatched_surname_tokens(
    doc_pairs: set[tuple[str, str]], folder_pairs: set[tuple[str, str]]
) -> list[str]:
    if not doc_pairs or not folder_pairs:
        return []
    doc_map: dict[str, set[str]] = {}
    folder_map: dict[str, set[str]] = {}
    for left, right in doc_pairs:
        doc_map.setdefault(left, set()).add(right)
        doc_map.setdefault(right, set()).add(left)
    for left, right in folder_pairs:
        folder_map.setdefault(left, set()).add(right)
        folder_map.setdefault(right, set()).add(left)
    mismatched: list[str] = []
    for token in sorted(set(doc_map) & set(folder_map)):
        if token in COMMON_FIRST_NAMES:
            continue
        doc_others = doc_map[token]
        folder_others = folder_map[token]
        if doc_others and folder_others and not (doc_others & folder_others):
            mismatched.append(token)
    return mismatched


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
