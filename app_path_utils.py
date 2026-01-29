import os
import re
from typing import List


ROOT_PATH = os.path.abspath(os.sep)
HOME_PATH = os.path.abspath(os.path.expanduser("~"))
MIN_PATH_LENGTH = 10


def _is_drive_root(path: str) -> bool:
    normalized = os.path.abspath(path)
    if normalized == ROOT_PATH:
        return True
    if os.name == "nt":
        return bool(re.match(r"^[a-zA-Z]:\\\\?$", normalized))
    return False


def _is_parent_path(parent: str, child: str) -> bool:
    try:
        common = os.path.commonpath([parent, child])
    except ValueError:
        return False
    return common == parent


def validate_path_set(paths: List[tuple[str, str]]) -> tuple[bool, str]:
    normalized: list[tuple[str, str]] = []
    for label, raw_path in paths:
        if not raw_path:
            return False, f"{label} is empty"
        abs_path = os.path.abspath(os.path.expanduser(raw_path))
        if len(abs_path) <= MIN_PATH_LENGTH:
            return False, f"{label} is too short to be safe: {abs_path}"
        if _is_drive_root(abs_path):
            return False, f"{label} points to a drive root: {abs_path}"
        if abs_path == HOME_PATH:
            return False, f"{label} points to the user home directory: {abs_path}"
        normalized.append((label, abs_path))

    for idx, (label_a, path_a) in enumerate(normalized):
        for label_b, path_b in normalized[idx + 1 :]:
            if path_a == path_b:
                return False, f"{label_a} and {label_b} reference the same path: {path_a}"
            if _is_parent_path(path_a, path_b):
                return False, f"{label_a} is a parent of {label_b}: {path_a} -> {path_b}"
            if _is_parent_path(path_b, path_a):
                return False, f"{label_a} is a child of {label_b}: {path_a} -> {path_b}"
    return True, ""
