from __future__ import annotations

import os
from typing import Iterable

from .models import FolderMeta
from .scorer import extract_surnames_from_folder, normalize_text, normalize_stopwords, tokenize


def build_folder_index(case_root: str, stopwords: Iterable[str]) -> list[FolderMeta]:
    entries: list[FolderMeta] = []
    blocked = normalize_stopwords(stopwords)
    for name in os.listdir(case_root):
        full_path = os.path.join(case_root, name)
        if not os.path.isdir(full_path):
            continue
        normalized = normalize_text(name)
        tokens = {tok for tok in tokenize(name, blocked) if tok}
        surnames = extract_surnames_from_folder(name, blocked)
        entries.append(
            FolderMeta(
                folder_path=full_path,
                folder_name=name,
                normalized_name=normalized,
                tokens=tokens,
                surnames=surnames,
                case_numbers=set(),
            )
        )
    return entries
