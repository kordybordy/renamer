import os
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from text_utils import normalize_polish


@dataclass
class CaseFolderInfo:
    path: str
    tokens: List[str]
    full: str


class DistributionManager:
    def __init__(self, normalizer: Callable[[str], str] = normalize_polish):
        self.normalizer = normalizer

    def build_case_index(self, case_root: str) -> List[CaseFolderInfo]:
        entries: List[CaseFolderInfo] = []
        for name in os.listdir(case_root):
            full_path = os.path.join(case_root, name)
            if not os.path.isdir(full_path):
                continue
            normalized = self.normalizer(name)
            tokens = [tok for tok in normalized.split(" ") if tok]
            entries.append(CaseFolderInfo(path=full_path, tokens=tokens, full=normalized))
        return entries

    def _defendant_tokens(self, defendant: str) -> Tuple[List[str], str]:
        normalized = self.normalizer(defendant)
        tokens = [tok for tok in normalized.split(" ") if tok]
        surname = tokens[-1] if tokens else ""
        return tokens, surname

    def find_matches(self, defendants: List[str], case_index: List[CaseFolderInfo]) -> List[CaseFolderInfo]:
        normalized_defendants = []
        for defendant in defendants:
            tokens, surname = self._defendant_tokens(defendant)
            if tokens:
                normalized_defendants.append((tokens, surname))

        matches: Dict[str, CaseFolderInfo] = {}
        for folder in case_index:
            folder_token_set = set(folder.tokens)
            for tokens, surname in normalized_defendants:
                if surname and surname in folder_token_set:
                    matches.setdefault(folder.path, folder)
                    break
                if surname and folder_token_set.issuperset(tokens):
                    matches.setdefault(folder.path, folder)
                    break
        return list(matches.values())

    def copy_pdf(self, source_path: str, target_dir: str, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        candidate = filename
        counter = 1
        os.makedirs(target_dir, exist_ok=True)
        while os.path.exists(os.path.join(target_dir, candidate)):
            candidate = f"{base} ({counter}){ext}"
            counter += 1
        shutil.copy2(source_path, os.path.join(target_dir, candidate))
        return candidate
