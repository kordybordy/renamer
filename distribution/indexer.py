from __future__ import annotations

import os
from typing import Iterable

from .models import FolderIndex, FolderMeta
from .scorer import (
    extract_person_pairs,
    extract_surnames_from_folder,
    extract_tokens,
    normalize_text,
    normalize_stopwords,
    strip_folder_suffix,
)


def build_folder_index(case_root: str, stopwords: Iterable[str]) -> FolderIndex:
    entries: list[FolderMeta] = []
    token_to_folders: dict[str, list[int]] = {}
    pair_to_folders: dict[tuple[str, str], list[int]] = {}
    blocked = normalize_stopwords(stopwords)
    for name in os.listdir(case_root):
        full_path = os.path.join(case_root, name)
        if not os.path.isdir(full_path):
            continue
        match_name = strip_folder_suffix(name)
        normalized = normalize_text(match_name)
        tokens = extract_tokens(match_name, blocked)
        surnames = extract_surnames_from_folder(match_name, blocked)
        person_pairs = extract_person_pairs(match_name, blocked)
        folder_meta = FolderMeta(
            folder_path=full_path,
            folder_name=name,
            match_name=match_name,
            normalized_name=normalized,
            tokens=tokens,
            surnames=surnames,
            person_pairs=person_pairs,
            case_numbers=set(),
        )
        entries.append(folder_meta)
        folder_id = len(entries) - 1
        for token in tokens:
            token_to_folders.setdefault(token, []).append(folder_id)
        for pair in person_pairs:
            pair_to_folders.setdefault(pair, []).append(folder_id)
    return FolderIndex(
        folders=entries,
        token_to_folders=token_to_folders,
        pair_to_folders=pair_to_folders,
    )
