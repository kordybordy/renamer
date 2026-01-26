from __future__ import annotations

import hashlib
import os
import shutil
from typing import Iterable


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path or ""))


def _is_drive_root(path: str) -> bool:
    drive, tail = os.path.splitdrive(path)
    if not drive:
        return False
    return tail in ("/", "\\", "")


def _common_path(paths: Iterable[str]) -> str:
    return os.path.commonpath([_normalize_path(p) for p in paths])


def _is_subpath(candidate: str, base: str) -> bool:
    candidate_norm = _normalize_path(candidate)
    base_norm = _normalize_path(base)
    try:
        return os.path.commonpath([candidate_norm, base_norm]) == base_norm
    except ValueError:
        return False


def validate_distribution_paths(
    input_folder: str,
    case_root: str,
    *,
    allow_home_case_root: bool,
) -> tuple[bool, str]:
    input_norm = _normalize_path(input_folder)
    case_norm = _normalize_path(case_root)

    if not input_norm or not case_norm:
        return False, "Input folder and case root must be provided."

    if _is_drive_root(input_norm) or _is_drive_root(case_norm):
        return False, "Drive roots are not allowed for safety."

    if input_norm == case_norm:
        return False, "Input folder and case root must be different."

    if _is_subpath(case_norm, input_norm):
        return False, "Case root cannot be inside the input folder."

    if _is_subpath(input_norm, case_norm):
        return False, "Input folder cannot be inside the case root."

    home_dir = _normalize_path(os.path.expanduser("~"))
    if not allow_home_case_root and _is_subpath(case_norm, home_dir):
        return False, "Case root inside the user home requires explicit confirmation."

    return True, ""


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_destination_path(
    source_path: str,
    destination_dir: str,
    *,
    exists_policy: str = "rename",
) -> tuple[str, str]:
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    os.makedirs(destination_dir, exist_ok=True)

    base_name = os.path.basename(source_path)
    base, ext = os.path.splitext(base_name)
    candidate = base_name
    counter = 1

    while True:
        dest_path = os.path.join(destination_dir, candidate)
        if not os.path.exists(dest_path):
            return dest_path, "planned"
        if exists_policy == "skip":
            return dest_path, "skip_existing"
        if exists_policy == "overwrite":
            return dest_path, "overwrite"
        if os.path.getsize(dest_path) == os.path.getsize(source_path):
            try:
                if _file_hash(dest_path) == _file_hash(source_path):
                    return dest_path, "skip_existing"
            except OSError:
                pass
        candidate = f"{base} ({counter}){ext}"
        counter += 1


def safe_copy(
    source_path: str, destination_dir: str, *, exists_policy: str = "rename"
) -> tuple[str, str]:
    dest_path, status = resolve_destination_path(
        source_path, destination_dir, exists_policy=exists_policy
    )
    if status == "skip_existing":
        return dest_path, "SKIPPED"
    if status == "overwrite" and os.path.exists(dest_path):
        if os.path.abspath(source_path) == os.path.abspath(dest_path):
            raise ValueError("Destination cannot equal source.")
        os.remove(dest_path)

    if os.path.abspath(source_path) == os.path.abspath(dest_path):
        raise ValueError("Destination cannot equal source.")

    shutil.copy2(source_path, dest_path)
    return dest_path, "OVERWROTE" if status == "overwrite" else "COPIED"


def safe_move(
    source_path: str, destination_dir: str, *, exists_policy: str = "rename"
) -> tuple[str, str]:
    dest_path, status = resolve_destination_path(
        source_path, destination_dir, exists_policy=exists_policy
    )
    if status == "skip_existing":
        return dest_path, "SKIPPED"
    if status == "overwrite" and os.path.exists(dest_path):
        if os.path.abspath(source_path) == os.path.abspath(dest_path):
            raise ValueError("Destination cannot equal source.")
        os.remove(dest_path)

    if os.path.abspath(source_path) == os.path.abspath(dest_path):
        raise ValueError("Destination cannot equal source.")

    shutil.move(source_path, dest_path)
    return dest_path, "OVERWROTE" if status == "overwrite" else "MOVED"
