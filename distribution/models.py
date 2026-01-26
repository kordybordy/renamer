from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DocumentMeta:
    source_path: str
    file_name: str
    plaintiffs: list[str]
    defendants: list[str]
    opposing_parties: list[str]
    case_numbers: list[str]
    letter_type: str
    raw_text_excerpt: str | None
    extraction_source: Literal["meta_json", "filename", "ai", "unknown"]
    tokens: set[str] = field(default_factory=set)
    surnames: set[str] = field(default_factory=set)
    person_pairs: set[tuple[str, str]] = field(default_factory=set)
    normalized_opposing: str = ""
    cache_ready: bool = False


@dataclass
class FolderMeta:
    folder_path: str
    folder_name: str
    match_name: str
    normalized_name: str
    tokens: set[str] = field(default_factory=set)
    surnames: set[str] = field(default_factory=set)
    person_pairs: set[tuple[str, str]] = field(default_factory=set)
    case_numbers: set[str] = field(default_factory=set)


@dataclass
class FolderIndex:
    folders: list[FolderMeta]
    token_to_folders: dict[str, list[int]] = field(default_factory=dict)
    pair_to_folders: dict[tuple[str, str], list[int]] = field(default_factory=dict)


@dataclass
class MatchCandidate:
    folder: FolderMeta
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class DistributionPlanItem:
    source_pdf: str
    chosen_folder: str | None
    candidates: list[MatchCandidate]
    decision: Literal["AUTO", "AI", "ASK", "UNMATCHED"]
    confidence: float
    reason: str
    dest_path: str | None
